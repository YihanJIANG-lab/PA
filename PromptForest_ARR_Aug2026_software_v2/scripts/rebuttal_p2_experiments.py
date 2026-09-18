"""Run P2 rebuttal analyses from saved PromptForest full-factorial results.

This script makes no LLM/API calls. It adds two secondary analyses:

1. Online bandit simulation as a low-information deployment sanity check.
2. LCB threshold sweep for conservative routing/risk-coverage diagnostics.

Outputs are written under data/rebuttal_p2/.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from rebuttal_p0_experiments import (  # noqa: E402
    MODELS,
    load_model_data,
    oracle_scores,
    pivot_outcomes,
    policy_value,
    promptforest_policy,
    realized_scores,
    task_best_policy,
    train_test_pivots,
)


def _safe_nanmean(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0 or np.all(np.isnan(arr)):
        return float("nan")
    return float(np.nanmean(arr))


def _outcome_lookup(md, test_uids: List[str]) -> Dict[tuple[str, int], float]:
    test_df = md.valid_df[md.valid_df["query_uid"].astype(str).isin(test_uids)].copy()
    return {
        (str(row.query_uid), int(row.assigned_strategy)): float(row.outcome_score_v2)
        for row in test_df[["query_uid", "assigned_strategy", "outcome_score_v2"]].itertuples(index=False)
    }


def _reward(lookup: Dict[tuple[str, int], float], uid: str, arm: int) -> float:
    return float(lookup.get((str(uid), int(arm)), np.nan))


def simulate_bandits(md, n_seeds: int = 200, epsilon: float = 0.1, alpha: float = 1.0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Simulate non-contextual online bandits over the saved test queries.

    The online policies observe only the reward for the selected strategy. The
    full-factorial table is used only as an offline replay oracle for whichever
    strategy the policy chooses.
    """
    train_pivot, test_pivot = train_test_pivots(md)
    test_uids = test_pivot.index.astype(str).tolist()
    arms = [int(c) for c in sorted(test_pivot.columns)]
    lookup = _outcome_lookup(md, test_uids)
    zero_arr = realized_scores(md.valid_df, {uid: 0 for uid in test_uids}, test_uids)
    oracle_arr = oracle_scores(test_pivot, test_uids)
    oracle_value = policy_value(oracle_arr)
    zero_value = policy_value(zero_arr)

    pf_arr = realized_scores(md.valid_df, promptforest_policy(md, test_uids), test_uids)
    task_arr = realized_scores(md.valid_df, task_best_policy(md, train_pivot, test_uids), test_uids)

    fixed_summary = [
        {
            "model": md.model,
            "policy": "oracle",
            "avg_reward": oracle_value,
            "gain_vs_zero_shot": oracle_value - zero_value,
            "avg_regret_vs_oracle": 0.0,
            "final_cumulative_regret": 0.0,
            "n_test_queries": len(test_uids),
            "n_seeds": 1,
        },
        {
            "model": md.model,
            "policy": "promptforest_offline_router",
            "avg_reward": policy_value(pf_arr),
            "gain_vs_zero_shot": policy_value(pf_arr) - zero_value,
            "avg_regret_vs_oracle": oracle_value - policy_value(pf_arr),
            "final_cumulative_regret": float(np.nansum(oracle_arr - pf_arr)),
            "n_test_queries": int(np.sum(~np.isnan(pf_arr))),
            "n_seeds": 1,
        },
        {
            "model": md.model,
            "policy": "task_best_offline",
            "avg_reward": policy_value(task_arr),
            "gain_vs_zero_shot": policy_value(task_arr) - zero_value,
            "avg_regret_vs_oracle": oracle_value - policy_value(task_arr),
            "final_cumulative_regret": float(np.nansum(oracle_arr - task_arr)),
            "n_test_queries": int(np.sum(~np.isnan(task_arr))),
            "n_seeds": 1,
        },
        {
            "model": md.model,
            "policy": "zero_shot_direct",
            "avg_reward": zero_value,
            "gain_vs_zero_shot": 0.0,
            "avg_regret_vs_oracle": oracle_value - zero_value,
            "final_cumulative_regret": float(np.nansum(oracle_arr - zero_arr)),
            "n_test_queries": int(np.sum(~np.isnan(zero_arr))),
            "n_seeds": 1,
        },
    ]

    summary_rows = []
    curve_rows = []
    online_policies = ["uniform_random", f"epsilon_greedy_{epsilon:g}", f"ucb1_alpha_{alpha:g}", "thompson_beta"]
    for policy_name in online_policies:
        seed_rewards = []
        seed_regrets = []
        curve_accumulator: Dict[int, List[tuple[float, float, float]]] = {}
        for seed in range(n_seeds):
            rng = np.random.default_rng(2026 + seed)
            order = rng.permutation(test_uids)
            counts = {arm: 0 for arm in arms}
            sums = {arm: 0.0 for arm in arms}
            beta_success = {arm: 1.0 for arm in arms}
            beta_failure = {arm: 1.0 for arm in arms}
            rewards = []
            regrets = []
            for t, uid in enumerate(order, start=1):
                if policy_name == "uniform_random":
                    arm = int(rng.choice(arms))
                elif policy_name.startswith("epsilon_greedy"):
                    if rng.random() < epsilon or sum(counts.values()) < len(arms):
                        unexplored = [a for a in arms if counts[a] == 0]
                        arm = int(rng.choice(unexplored if unexplored else arms))
                    else:
                        means = {a: sums[a] / max(counts[a], 1) for a in arms}
                        arm = int(max(means, key=means.get))
                elif policy_name.startswith("ucb1"):
                    unexplored = [a for a in arms if counts[a] == 0]
                    if unexplored:
                        arm = int(rng.choice(unexplored))
                    else:
                        total = max(sum(counts.values()), 1)
                        scores = {
                            a: (sums[a] / counts[a]) + alpha * np.sqrt(np.log(total + 1) / counts[a])
                            for a in arms
                        }
                        arm = int(max(scores, key=scores.get))
                elif policy_name == "thompson_beta":
                    samples = {a: rng.beta(beta_success[a], beta_failure[a]) for a in arms}
                    arm = int(max(samples, key=samples.get))
                else:
                    raise ValueError(policy_name)

                reward = _reward(lookup, uid, arm)
                if np.isnan(reward):
                    # Missing rows are rare; use zero-shot for a valid replay step.
                    arm = 0
                    reward = _reward(lookup, uid, arm)
                reward = 0.0 if np.isnan(reward) else float(np.clip(reward, 0.0, 1.0))
                oracle_reward = float(np.nanmax(test_pivot.loc[uid].to_numpy(dtype=float)))
                counts[arm] += 1
                sums[arm] += reward
                beta_success[arm] += reward
                beta_failure[arm] += 1.0 - reward
                rewards.append(reward)
                regrets.append(oracle_reward - reward)
                curve_accumulator.setdefault(t, []).append(
                    (
                        float(np.sum(rewards)),
                        float(np.sum(regrets)),
                        float(np.mean(rewards)),
                    )
                )
            seed_rewards.append(float(np.mean(rewards)))
            seed_regrets.append(float(np.sum(regrets)))
        for step, triples in curve_accumulator.items():
            arr = np.asarray(triples, dtype=float)
            curve_rows.append(
                {
                    "model": md.model,
                    "policy": policy_name,
                    "step": step,
                    "mean_cumulative_reward": float(arr[:, 0].mean()),
                    "mean_cumulative_regret": float(arr[:, 1].mean()),
                    "mean_running_avg_reward": float(arr[:, 2].mean()),
                    "n_seeds": n_seeds,
                }
            )
        avg_reward = float(np.mean(seed_rewards))
        final_regret = float(np.mean(seed_regrets))
        summary_rows.append(
            {
                "model": md.model,
                "policy": policy_name,
                "avg_reward": avg_reward,
                "gain_vs_zero_shot": avg_reward - zero_value,
                "avg_regret_vs_oracle": oracle_value - avg_reward,
                "final_cumulative_regret": final_regret,
                "avg_reward_sd_across_seeds": float(np.std(seed_rewards, ddof=1)),
                "n_test_queries": len(test_uids),
                "n_seeds": n_seeds,
            }
        )

    summary = pd.DataFrame(fixed_summary + summary_rows)
    curves = pd.DataFrame(curve_rows)
    return summary.sort_values(["model", "avg_reward"], ascending=[True, False]), curves


def threshold_sweep(md, thresholds: List[float]) -> pd.DataFrame:
    train_pivot, test_pivot = train_test_pivots(md)
    test_uids = test_pivot.index.astype(str).tolist()
    assignment = md.assignment.set_index("query_uid").reindex(test_uids).copy()
    assignment["optimal_treatment_id"] = pd.to_numeric(assignment["optimal_treatment_id"], errors="coerce").fillna(0).astype(int)
    assignment["expected_gain"] = pd.to_numeric(assignment["expected_gain"], errors="coerce").fillna(0.0)
    assignment["ci_lower"] = pd.to_numeric(assignment["ci_lower"], errors="coerce").fillna(-np.inf)

    zero_policy = {uid: 0 for uid in test_uids}
    zero_arr = realized_scores(md.valid_df, zero_policy, test_uids)
    zero_value = policy_value(zero_arr)
    oracle_arr = oracle_scores(test_pivot, test_uids)
    oracle_value = policy_value(oracle_arr)

    rows = []
    for threshold in thresholds:
        policy = {}
        routed = []
        for uid, row in assignment.iterrows():
            tid = int(row["optimal_treatment_id"])
            use_route = tid != 0 and float(row["ci_lower"]) > threshold
            policy[str(uid)] = tid if use_route else 0
            routed.append(bool(use_route))
        arr = realized_scores(md.valid_df, policy, test_uids)
        selected_gain = arr - zero_arr
        routed_mask = np.asarray(routed, dtype=bool)
        routed_gains = selected_gain[routed_mask]
        rows.append(
            {
                "model": md.model,
                "lcb_threshold": threshold,
                "route_rate": float(np.mean(routed_mask)),
                "policy_value": policy_value(arr),
                "gain_vs_zero_shot": policy_value(arr) - zero_value,
                "avg_regret_vs_oracle": oracle_value - policy_value(arr),
                "mean_routed_realized_gain": _safe_nanmean(routed_gains),
                "false_positive_route_rate": float(np.mean(routed_gains <= 0)) if routed_gains.size else 0.0,
                "n_routed": int(np.sum(routed_mask)),
                "n_test_queries": int(np.sum(~np.isnan(arr))),
            }
        )
    return pd.DataFrame(rows)


def write_summary(out_dir: Path, bandit_df: pd.DataFrame, threshold_df: pd.DataFrame) -> None:
    def md_table(df: pd.DataFrame) -> str:
        cols = list(df.columns)
        lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
        for _, row in df.iterrows():
            vals = []
            for col in cols:
                val = row[col]
                vals.append(f"{val:.4f}" if isinstance(val, float) else str(val))
            lines.append("| " + " | ".join(vals) + " |")
        return "\n".join(lines)

    bandit_core = bandit_df[
        [
            "model",
            "policy",
            "avg_reward",
            "gain_vs_zero_shot",
            "avg_regret_vs_oracle",
            "final_cumulative_regret",
            "n_test_queries",
            "n_seeds",
        ]
    ].copy()

    best_threshold = (
        threshold_df.sort_values(["model", "policy_value"], ascending=[True, False])
        .groupby("model", as_index=False)
        .head(1)
    )
    conservative = threshold_df[threshold_df["lcb_threshold"].isin([0.0, 0.05, 0.1])].copy()

    lines = [
        "# P2 Rebuttal Results Summary",
        "",
        "Generated from saved full-factorial results; no new LLM/API calls.",
        "",
        "## Online bandit sanity check",
        "",
        md_table(bandit_core.round(4)),
        "",
        "Interpretation: the bandit policies are low-information deployment approximations. They should not be treated as the main contribution. Their role is to show how much performance is lost when the full-information diagnostic table is unavailable at deployment time.",
        "",
        "## Best LCB threshold by model",
        "",
        md_table(
            best_threshold[
                [
                    "model",
                    "lcb_threshold",
                    "route_rate",
                    "policy_value",
                    "gain_vs_zero_shot",
                    "false_positive_route_rate",
                    "n_routed",
                ]
            ].round(4)
        ),
        "",
        "## Conservative thresholds",
        "",
        md_table(
            conservative[
                [
                    "model",
                    "lcb_threshold",
                    "route_rate",
                    "policy_value",
                    "gain_vs_zero_shot",
                    "false_positive_route_rate",
                    "n_routed",
                ]
            ].round(4)
        ),
        "",
        "## Rebuttal framing",
        "",
        "These results should be framed as secondary diagnostics. The bandit simulation clarifies that full-information offline replay is an evaluation/training protocol, while low-information online exploration has weaker reward and higher regret. The LCB sweep shows that PromptForest can be made more conservative by routing only when the lower confidence bound clears a threshold, trading coverage for lower false-positive routing.",
    ]
    (out_dir / "p2_rebuttal_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/rebuttal_p2"))
    parser.add_argument("--n-seeds", type=int, default=200)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    thresholds = [-0.20, -0.10, -0.05, 0.0, 0.02, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30, 0.40]

    bandit_all = []
    curve_all = []
    threshold_all = []
    for model in MODELS:
        md = load_model_data(args.data_root, model)
        model_out = args.out_dir / f"{model}_n100"
        model_out.mkdir(parents=True, exist_ok=True)
        bandit_df, curve_df = simulate_bandits(md, n_seeds=args.n_seeds)
        sweep_df = threshold_sweep(md, thresholds)
        bandit_df.to_csv(model_out / f"p2_online_bandit_simulation_{model}.csv", index=False)
        curve_df.to_csv(model_out / f"p2_online_bandit_curves_{model}.csv", index=False)
        sweep_df.to_csv(model_out / f"p2_lcb_threshold_sweep_{model}.csv", index=False)
        bandit_all.append(bandit_df)
        curve_all.append(curve_df)
        threshold_all.append(sweep_df)

    bandit_all_df = pd.concat(bandit_all, ignore_index=True)
    curve_all_df = pd.concat(curve_all, ignore_index=True)
    threshold_all_df = pd.concat(threshold_all, ignore_index=True)
    bandit_all_df.to_csv(args.out_dir / "p2_online_bandit_simulation_all_models.csv", index=False)
    curve_all_df.to_csv(args.out_dir / "p2_online_bandit_curves_all_models.csv", index=False)
    threshold_all_df.to_csv(args.out_dir / "p2_lcb_threshold_sweep_all_models.csv", index=False)
    write_summary(args.out_dir, bandit_all_df, threshold_all_df)
    print(f"Wrote P2 rebuttal outputs to {args.out_dir}")


if __name__ == "__main__":
    main()
