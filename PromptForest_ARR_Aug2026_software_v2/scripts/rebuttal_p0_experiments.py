"""Run P0 rebuttal analyses from saved full-factorial PromptForest results.

This script does not make any LLM/API calls. It rebuilds:
  1. Direct supervised outcome-difference baselines vs PromptForest.
  2. Corrected two-sided bootstrap significance tables with query/task clusters.
  3. Deployment-time token/cost analysis with input/output/caching scenarios.

Outputs are written under data/rebuttal_p0/.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


MODELS = ["gpt-5.5", "deepseek-v4-pro", "claude-sonnet-4.6"]
OUTCOME_COL = "outcome_score_v2"
ALPHA = 0.05


@dataclass
class ModelData:
    model: str
    result_dir: Path
    full_df: pd.DataFrame
    valid_df: pd.DataFrame
    train_uids: List[str]
    test_uids: List[str]
    features: pd.DataFrame
    feature_cols: List[str]
    assignment: pd.DataFrame


def model_dir(data_root: Path, model: str) -> Path:
    return data_root / "results" / f"{model}_n100"


def first_existing(paths: Iterable[Path]) -> Path:
    for path in paths:
        if path.exists():
            return path
    raise FileNotFoundError("None of the candidate files exists: " + ", ".join(map(str, paths)))


def read_feature_cols(processed_root: Path) -> List[str]:
    return (processed_root / "01_feature_columns.txt").read_text(encoding="utf-8").splitlines()


def load_model_data(data_root: Path, model: str) -> ModelData:
    processed = data_root / "processed"
    rdir = model_dir(data_root, model)
    results_path = first_existing(
        [
            rdir / f"02_experiment_results_{model}_n100_scored_v2.csv",
            rdir / f"02_experiment_results_{model}_n100_scored_v3.csv",
            rdir / f"02_experiment_results_{model}_n100_light_v2.csv",
        ]
    )
    full_df = pd.read_csv(results_path)
    if "is_valid_outcome" in full_df.columns:
        valid_df = full_df[full_df["is_valid_outcome"].astype(bool)].copy()
    else:
        valid_df = full_df.copy()
    valid_df[OUTCOME_COL] = pd.to_numeric(valid_df[OUTCOME_COL], errors="coerce")
    valid_df = valid_df.dropna(subset=[OUTCOME_COL, "query_uid", "assigned_strategy"]).copy()
    valid_df["assigned_strategy"] = valid_df["assigned_strategy"].astype(int)

    features = pd.read_pickle(processed / "01_featured_queries.pkl")
    features["query_uid"] = features["query_uid"].astype(str)
    feature_cols = read_feature_cols(processed)
    needed_cols = ["query_uid", "dataset_name", "task_type"] + feature_cols
    feat = features[needed_cols].drop_duplicates("query_uid")
    merged = valid_df.merge(feat, on=["query_uid", "dataset_name", "task_type"], how="left", validate="many_to_one")
    uids = merged[["query_uid", "dataset_name"]].drop_duplicates()
    train_uid_df, test_uid_df = train_test_split(
        uids,
        test_size=0.25,
        random_state=42,
        stratify=uids["dataset_name"] if uids["dataset_name"].nunique() > 1 else None,
    )
    train_uids = train_uid_df["query_uid"].astype(str).tolist()
    test_uids = test_uid_df["query_uid"].astype(str).tolist()
    assignment = pd.read_csv(rdir / f"03_optimal_assignment_v3_{model}.csv")
    assignment["query_uid"] = assignment["query_uid"].astype(str)
    assignment["optimal_treatment_id"] = assignment["optimal_treatment_id"].astype(int)
    # The saved assignment file is the authoritative Step-5/Step-6 test set.
    # For GPT, a few test queries have one missing strategy outcome; the original
    # paper tables still keep the 275-query assignment set rather than using a
    # complete-case test pivot.
    if len(assignment) > 0:
        test_uids = assignment["query_uid"].astype(str).tolist()

    return ModelData(model, rdir, full_df, valid_df, train_uids, test_uids, features, feature_cols, assignment)


def pivot_outcomes(df: pd.DataFrame) -> pd.DataFrame:
    pivot = df.pivot_table(
        index="query_uid",
        columns="assigned_strategy",
        values=OUTCOME_COL,
        aggfunc="mean",
    )
    return pivot


def feature_matrix(features: pd.DataFrame, uids: List[str], feature_cols: List[str]) -> pd.DataFrame:
    f = features.set_index("query_uid").reindex(uids)
    missing = [c for c in feature_cols if c not in f.columns]
    if missing:
        raise KeyError(f"Missing feature columns: {missing[:10]}")
    X = f[feature_cols].apply(pd.to_numeric, errors="coerce")
    return X.fillna(0.0)


def realized_scores(df: pd.DataFrame, policy: Dict[str, int], uids: List[str]) -> np.ndarray:
    lookup = {
        (str(row.query_uid), int(row.assigned_strategy)): float(getattr(row, OUTCOME_COL))
        for row in df[["query_uid", "assigned_strategy", OUTCOME_COL]].itertuples(index=False)
    }
    out = []
    for uid in uids:
        tid = int(policy.get(uid, 0))
        out.append(lookup.get((uid, tid), np.nan))
    return np.asarray(out, dtype=float)


def fixed_strategy_scores(df: pd.DataFrame, uids: List[str], tid: int) -> np.ndarray:
    return realized_scores(df, {uid: tid for uid in uids}, uids)


def random_scores(pivot: pd.DataFrame, uids: List[str]) -> np.ndarray:
    return pivot.reindex(uids).mean(axis=1).to_numpy(dtype=float)


def oracle_scores(pivot: pd.DataFrame, uids: List[str]) -> np.ndarray:
    return pivot.reindex(uids).max(axis=1).to_numpy(dtype=float)


def policy_value(arr: np.ndarray) -> float:
    return float(np.nanmean(arr))


def train_test_pivots(md: ModelData) -> Tuple[pd.DataFrame, pd.DataFrame]:
    pivot = pivot_outcomes(md.valid_df)
    train_pivot = pivot.reindex(md.train_uids).dropna(axis=0, how="any")
    test_pivot = pivot.reindex(md.test_uids)
    return train_pivot, test_pivot


def global_best_policy(train_pivot: pd.DataFrame, test_uids: List[str], require_positive: bool = False) -> Dict[str, int]:
    base = float(train_pivot[0].mean())
    means = {int(c): float(train_pivot[c].mean()) for c in train_pivot.columns if int(c) != 0}
    if not means:
        best_tid = 0
    else:
        best_tid = max(means, key=means.get)
        if require_positive and means[best_tid] - base <= 0:
            best_tid = 0
    return {uid: int(best_tid) for uid in test_uids}


def task_best_policy(md: ModelData, train_pivot: pd.DataFrame, test_uids: List[str]) -> Dict[str, int]:
    meta = md.features.set_index("query_uid")[["task_type"]]
    policy = {}
    global_policy = global_best_policy(train_pivot, test_uids)
    global_tid = next(iter(global_policy.values())) if global_policy else 0
    for task, task_meta in meta.reindex(train_pivot.index).groupby("task_type"):
        task_uids = task_meta.index.tolist()
        sub = train_pivot.reindex(task_uids).dropna(axis=0, how="any")
        if len(sub) == 0:
            continue
        means = {int(c): float(sub[c].mean()) for c in sub.columns}
        best_tid = max(means, key=means.get)
        for uid in meta[(meta["task_type"] == task) & (meta.index.isin(test_uids))].index:
            policy[str(uid)] = int(best_tid)
    for uid in test_uids:
        policy.setdefault(uid, int(global_tid))
    return policy


def promptforest_policy(md: ModelData, test_uids: List[str]) -> Dict[str, int]:
    amap = dict(zip(md.assignment["query_uid"], md.assignment["optimal_treatment_id"]))
    return {uid: int(amap.get(uid, 0)) for uid in test_uids}


def fit_direct_diff_baselines(md: ModelData, train_pivot: pd.DataFrame, test_pivot: pd.DataFrame) -> Dict[str, Dict[str, int]]:
    train_uids = train_pivot.index.astype(str).tolist()
    test_uids = test_pivot.index.astype(str).tolist()
    X_train = feature_matrix(md.features, train_uids, md.feature_cols).to_numpy()
    X_test = feature_matrix(md.features, test_uids, md.feature_cols).to_numpy()
    treatment_ids = [int(c) for c in train_pivot.columns if int(c) != 0]
    y_diff = np.column_stack(
        [(train_pivot[tid].to_numpy(dtype=float) - train_pivot[0].to_numpy(dtype=float)) for tid in treatment_ids]
    )

    policies: Dict[str, Dict[str, int]] = {}

    per_strategy_preds = []
    for j, tid in enumerate(treatment_ids):
        model = RandomForestRegressor(
            n_estimators=500,
            max_depth=None,
            min_samples_leaf=3,
            random_state=42 + tid,
            n_jobs=1,
        )
        model.fit(X_train, y_diff[:, j])
        per_strategy_preds.append(model.predict(X_test))
    pred = np.column_stack(per_strategy_preds)
    policies["direct_diff_rf_per_strategy"] = choose_by_predicted_gain(test_uids, treatment_ids, pred)

    mo = RandomForestRegressor(
        n_estimators=500,
        max_depth=None,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=1,
    )
    mo.fit(X_train, y_diff)
    policies["direct_diff_rf_multioutput"] = choose_by_predicted_gain(test_uids, treatment_ids, mo.predict(X_test))

    ridge = make_pipeline(StandardScaler(with_mean=True), MultiOutputRegressor(Ridge(alpha=1.0)))
    ridge.fit(X_train, y_diff)
    policies["direct_diff_ridge_multioutput"] = choose_by_predicted_gain(test_uids, treatment_ids, ridge.predict(X_test))

    # Strategy-specific outcome regressors, including T0. This is a direct supervised policy,
    # not a treatment-effect estimator.
    outcome_preds = []
    all_tids = [int(c) for c in train_pivot.columns]
    for tid in all_tids:
        model = RandomForestRegressor(
            n_estimators=500,
            max_depth=None,
            min_samples_leaf=3,
            random_state=100 + tid,
            n_jobs=1,
        )
        model.fit(X_train, train_pivot[tid].to_numpy(dtype=float))
        outcome_preds.append(model.predict(X_test))
    outcome_pred = np.column_stack(outcome_preds)
    best_idx = np.argmax(outcome_pred, axis=1)
    policies["strategy_specific_outcome_rf"] = {
        uid: int(all_tids[idx]) for uid, idx in zip(test_uids, best_idx)
    }
    return policies


def choose_by_predicted_gain(uids: List[str], treatment_ids: List[int], pred: np.ndarray) -> Dict[str, int]:
    best_idx = np.argmax(pred, axis=1)
    best_gain = pred[np.arange(len(uids)), best_idx]
    policy = {}
    for uid, idx, gain in zip(uids, best_idx, best_gain):
        policy[uid] = int(treatment_ids[idx]) if float(gain) > 0 else 0
    return policy


def p_adjust_holm(p_values: np.ndarray) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    running = 0.0
    m = len(p)
    for rank, idx in enumerate(order):
        val = (m - rank) * p[idx]
        running = max(running, val)
        adjusted[idx] = min(running, 1.0)
    return adjusted


def p_adjust_bh(p_values: np.ndarray) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)[::-1]
    adjusted = np.empty_like(p)
    running = 1.0
    m = len(p)
    for rank_from_high, idx in enumerate(order):
        rank = m - rank_from_high
        val = p[idx] * m / rank
        running = min(running, val)
        adjusted[idx] = min(running, 1.0)
    return adjusted


def bootstrap_diff(
    pf: np.ndarray,
    baseline: np.ndarray,
    clusters: np.ndarray | None = None,
    n_bootstrap: int = 10000,
    seed: int = 42,
) -> Tuple[float, float, float, float]:
    rng = np.random.RandomState(seed)
    pf = np.asarray(pf, dtype=float)
    baseline = np.asarray(baseline, dtype=float)
    diff = pf - baseline
    if clusters is None:
        boot = []
        n = len(diff)
        for _ in range(n_bootstrap):
            idx = rng.choice(n, size=n, replace=True)
            boot.append(float(np.nanmean(diff[idx])))
    else:
        clusters = np.asarray(clusters)
        unique = np.array(pd.Series(clusters).dropna().unique())
        boot = []
        for _ in range(n_bootstrap):
            sampled = rng.choice(unique, size=len(unique), replace=True)
            vals = []
            for c in sampled:
                vals.extend(diff[clusters == c].tolist())
            boot.append(float(np.nanmean(vals)))
    boot_arr = np.asarray(boot, dtype=float)
    ci_lo = float(np.nanpercentile(boot_arr, 2.5))
    ci_hi = float(np.nanpercentile(boot_arr, 97.5))
    p_left = float(np.nanmean(boot_arr <= 0))
    p_right = float(np.nanmean(boot_arr >= 0))
    p_two = min(1.0, 2.0 * min(p_left, p_right))
    return float(np.nanmean(diff)), ci_lo, ci_hi, p_two


def build_policy_arrays(md: ModelData, n_bootstrap: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_pivot, test_pivot = train_test_pivots(md)
    test_uids = test_pivot.index.astype(str).tolist()
    test_df = md.valid_df[md.valid_df["query_uid"].astype(str).isin(test_uids)].copy()
    meta = md.features.set_index("query_uid").reindex(test_uids)

    policies = {
        "zero_shot_direct": {uid: 0 for uid in test_uids},
        "random": None,
        "global_best": global_best_policy(train_pivot, test_uids, require_positive=False),
        "pairwise_ate_selector": global_best_policy(train_pivot, test_uids, require_positive=True),
        "task_best": task_best_policy(md, train_pivot, test_uids),
        "promptforest": promptforest_policy(md, test_uids),
    }
    policies.update(fit_direct_diff_baselines(md, train_pivot, test_pivot))

    arrays: Dict[str, np.ndarray] = {}
    for name, policy in policies.items():
        if name == "random":
            arrays[name] = random_scores(test_pivot, test_uids)
        else:
            arrays[name] = realized_scores(test_df, policy, test_uids)
    arrays["oracle"] = oracle_scores(test_pivot, test_uids)

    baseline_value = policy_value(arrays["zero_shot_direct"])
    policy_rows = []
    for name, arr in arrays.items():
        policy_rows.append(
            {
                "model": md.model,
                "policy": name,
                "value": policy_value(arr),
                "gain_vs_zero_shot": policy_value(arr) - baseline_value,
                "n_queries": int(np.sum(~np.isnan(arr))),
            }
        )
    policy_df = pd.DataFrame(policy_rows).sort_values("value", ascending=False)

    pf = arrays["promptforest"]
    sig_rows = []
    for name, arr in arrays.items():
        if name in {"promptforest", "oracle"}:
            continue
        diff, ci_lo, ci_hi, p_two = bootstrap_diff(pf, arr, n_bootstrap=n_bootstrap)
        task_diff, task_lo, task_hi, task_p = bootstrap_diff(
            pf,
            arr,
            clusters=meta["dataset_name"].to_numpy(),
            n_bootstrap=n_bootstrap,
            seed=123,
        )
        sig_rows.append(
            {
                "model": md.model,
                "baseline": name,
                "pf_mean": policy_value(pf),
                "baseline_mean": policy_value(arr),
                "diff_mean": diff,
                "query_cluster_ci_lower": ci_lo,
                "query_cluster_ci_upper": ci_hi,
                "p_two_sided": p_two,
                "task_cluster_ci_lower": task_lo,
                "task_cluster_ci_upper": task_hi,
                "task_cluster_p_two_sided": task_p,
            }
        )
    sig_df = pd.DataFrame(sig_rows)
    sig_df["p_holm"] = p_adjust_holm(sig_df["p_two_sided"].to_numpy())
    sig_df["p_bh_fdr"] = p_adjust_bh(sig_df["p_two_sided"].to_numpy())
    sig_df["significant_holm"] = (
        (sig_df["p_holm"] < ALPHA)
        & ((sig_df["query_cluster_ci_lower"] > 0) | (sig_df["query_cluster_ci_upper"] < 0))
    )
    sig_df["significant_task_cluster"] = (
        (sig_df["task_cluster_p_two_sided"] < ALPHA)
        & ((sig_df["task_cluster_ci_lower"] > 0) | (sig_df["task_cluster_ci_upper"] < 0))
    )

    per_query = pd.DataFrame({"query_uid": test_uids, "dataset_name": meta["dataset_name"].values})
    for name, arr in arrays.items():
        per_query[name] = arr
    return policy_df, sig_df, per_query


TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def approx_tokens(text: object) -> int:
    if pd.isna(text):
        return 0
    s = str(text)
    if not s:
        return 0
    # Conservative fallback without provider-specific tokenizers.
    return max(1, len(TOKEN_RE.findall(s)))


def deployment_cost_table(md: ModelData, per_query: pd.DataFrame) -> pd.DataFrame:
    df = md.full_df.copy()
    if "prompt" not in df.columns or "response" not in df.columns:
        return pd.DataFrame(
            [
                {
                    "model": md.model,
                    "note": "Full prompt/response text unavailable; token-level cost analysis skipped.",
                }
            ]
        )
    df["query_uid"] = df["query_uid"].astype(str)
    df["assigned_strategy"] = df["assigned_strategy"].astype(int)
    df["input_tokens_approx"] = df["prompt"].map(approx_tokens)
    df["output_tokens_approx"] = df["response"].map(approx_tokens)
    pf_policy = dict(zip(md.assignment["query_uid"].astype(str), md.assignment["optimal_treatment_id"].astype(int)))
    test_uids = per_query["query_uid"].astype(str).tolist()

    strategy_means = (
        df[df["query_uid"].isin(test_uids)]
        .groupby(["assigned_strategy", "strategy_name"], dropna=False)[
            ["input_tokens_approx", "output_tokens_approx", "latency"]
        ]
        .mean()
        .reset_index()
    )
    rows = []

    def selected_rows(policy_name: str, selector: Dict[str, int] | None) -> pd.DataFrame:
        if selector is None:
            return df[df["query_uid"].isin(test_uids)].copy()
        keys = pd.DataFrame({"query_uid": test_uids, "assigned_strategy": [selector.get(uid, 0) for uid in test_uids]})
        return keys.merge(df, on=["query_uid", "assigned_strategy"], how="left")

    train_pivot, _ = train_test_pivots(md)
    policies = {
        "Zero-Shot Direct": {uid: 0 for uid in test_uids},
        "PromptForest": pf_policy,
        "Task-Best": task_best_policy(md, train_pivot, test_uids),
        "Random": None,
    }

    # Normalized price units: input=1, output=10 reflects common output-token premium;
    # cached input is reported with 0.25x sensitivity. This avoids baking unstable vendor prices.
    for policy_name, selector in policies.items():
        sel = selected_rows(policy_name, selector)
        in_tok = float(sel["input_tokens_approx"].mean())
        out_tok = float(sel["output_tokens_approx"].mean())
        latency = float(sel["latency"].mean()) if "latency" in sel.columns else np.nan
        no_cache = in_tok + 10.0 * out_tok
        cached = 0.25 * in_tok + 10.0 * out_tok
        rows.append(
            {
                "model": md.model,
                "policy": policy_name,
                "avg_input_tokens_approx": in_tok,
                "avg_output_tokens_approx": out_tok,
                "avg_latency_sec": latency,
                "normalized_cost_no_cache": no_cache,
                "normalized_cost_cached_input_25pct": cached,
            }
        )
    out = pd.DataFrame(rows)
    z_no = float(out.loc[out["policy"] == "Zero-Shot Direct", "normalized_cost_no_cache"].iloc[0])
    z_cached = float(out.loc[out["policy"] == "Zero-Shot Direct", "normalized_cost_cached_input_25pct"].iloc[0])
    out["cost_multiplier_no_cache_vs_zsd"] = out["normalized_cost_no_cache"] / z_no
    out["cost_multiplier_cached_vs_zsd"] = out["normalized_cost_cached_input_25pct"] / z_cached
    out["pricing_note"] = "normalized units: input=1, output=10, cached input=0.25; deployment invokes one selected strategy"

    strategy_means.insert(0, "model", md.model)
    return out, strategy_means


def write_markdown_summary(out_dir: Path, all_policy: pd.DataFrame, all_sig: pd.DataFrame, all_cost: pd.DataFrame) -> None:
    def md_table(df: pd.DataFrame) -> str:
        df = df.copy()
        cols = list(df.columns)
        lines = [
            "| " + " | ".join(cols) + " |",
            "| " + " | ".join(["---"] * len(cols)) + " |",
        ]
        for _, row in df.iterrows():
            vals = []
            for col in cols:
                val = row[col]
                if isinstance(val, float):
                    vals.append(f"{val:.4f}")
                else:
                    vals.append(str(val))
            lines.append("| " + " | ".join(vals) + " |")
        return "\n".join(lines)

    policy_wide = all_policy.pivot(index="policy", columns="model", values="value").round(4).reset_index()
    sig_display = all_sig[
        [
            "model",
            "baseline",
            "diff_mean",
            "query_cluster_ci_lower",
            "query_cluster_ci_upper",
            "p_two_sided",
            "p_holm",
            "p_bh_fdr",
            "significant_holm",
            "task_cluster_ci_lower",
            "task_cluster_ci_upper",
            "task_cluster_p_two_sided",
            "significant_task_cluster",
        ]
    ].round(4)
    cost_display = all_cost[
        [
            "model",
            "policy",
            "avg_input_tokens_approx",
            "avg_output_tokens_approx",
            "cost_multiplier_no_cache_vs_zsd",
            "cost_multiplier_cached_vs_zsd",
        ]
    ].round(3)
    lines = [
        "# P0 Rebuttal Results Summary",
        "",
        "Generated from saved full-factorial results; no new LLM/API calls.",
        "",
        "## Direct supervised baselines",
        "",
        md_table(policy_wide),
        "",
        "## Corrected significance, PromptForest vs baselines",
        "",
        md_table(sig_display),
        "",
        "## Deployment cost sensitivity",
        "",
        md_table(cost_display),
        "",
    ]
    (out_dir / "p0_rebuttal_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--out", default="data/rebuttal_p0")
    parser.add_argument("--bootstrap", type=int, default=10000)
    args = parser.parse_args()

    data_root = Path(args.data_root)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_policy = []
    all_sig = []
    all_per_query = []
    all_cost = []
    all_strategy_cost = []
    for model in MODELS:
        print(f"[P0] Processing {model}")
        md = load_model_data(data_root, model)
        policy_df, sig_df, per_query = build_policy_arrays(md, n_bootstrap=args.bootstrap)
        cost_df, strategy_cost_df = deployment_cost_table(md, per_query)

        mdir = out_dir / f"{model}_n100"
        mdir.mkdir(parents=True, exist_ok=True)
        policy_df.to_csv(mdir / f"p0_direct_supervised_policy_values_{model}.csv", index=False)
        sig_df.to_csv(mdir / f"p0_corrected_significance_{model}.csv", index=False)
        per_query.to_csv(mdir / f"p0_per_query_policy_values_{model}.csv", index=False)
        cost_df.to_csv(mdir / f"p0_deployment_cost_{model}.csv", index=False)
        strategy_cost_df.to_csv(mdir / f"p0_strategy_token_summary_{model}.csv", index=False)

        all_policy.append(policy_df)
        all_sig.append(sig_df)
        all_per_query.append(per_query.assign(model=model))
        all_cost.append(cost_df)
        all_strategy_cost.append(strategy_cost_df)

    policy_all = pd.concat(all_policy, ignore_index=True)
    sig_all = pd.concat(all_sig, ignore_index=True)
    per_query_all = pd.concat(all_per_query, ignore_index=True)
    cost_all = pd.concat(all_cost, ignore_index=True)
    strategy_cost_all = pd.concat(all_strategy_cost, ignore_index=True)

    policy_all.to_csv(out_dir / "p0_direct_supervised_policy_values_all_models.csv", index=False)
    sig_all.to_csv(out_dir / "p0_corrected_significance_all_models.csv", index=False)
    per_query_all.to_csv(out_dir / "p0_per_query_policy_values_all_models.csv", index=False)
    cost_all.to_csv(out_dir / "p0_deployment_cost_all_models.csv", index=False)
    strategy_cost_all.to_csv(out_dir / "p0_strategy_token_summary_all_models.csv", index=False)
    write_markdown_summary(out_dir, policy_all, sig_all, cost_all)
    print(f"[P0] Wrote rebuttal artifacts to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
