"""Run resubmission-oriented P1-P5 analyses from saved PromptForest results.

No LLM/API calls are made. The script rebuilds primary baselines on the same
split/filtering protocol, then adds:

1. Primary baseline table with Task-Best, kNN, direct RF, PromptForest, Oracle.
2. Selective routing and coverage-risk frontiers.
3. Oracle headroom and heterogeneity decomposition.
4. Regret/error diagnostics explaining where PromptForest differs from direct RF.

Outputs are written under data/resubmission_p1_p5/.
"""

from __future__ import annotations

import argparse
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics.pairwise import cosine_similarity

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from rebuttal_p0_experiments import (  # noqa: E402
    MODELS,
    feature_matrix,
    global_best_policy,
    load_model_data,
    oracle_scores,
    pivot_outcomes,
    policy_value,
    promptforest_policy,
    random_scores,
    realized_scores,
    task_best_policy,
    train_test_pivots,
)
from rebuttal_p1_experiments import (  # noqa: E402
    bm25_knn_policy,
    embedding_knn_policy,
    feature_groups,
    fit_direct_diff_rf_with_features,
)


def safe_mean(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0 or np.all(np.isnan(arr)):
        return float("nan")
    return float(np.nanmean(arr))


def outcome_lookup(md, test_uids: List[str]) -> Dict[tuple[str, int], float]:
    test_df = md.valid_df[md.valid_df["query_uid"].astype(str).isin(test_uids)].copy()
    return {
        (str(row.query_uid), int(row.assigned_strategy)): float(row.outcome_score_v2)
        for row in test_df[["query_uid", "assigned_strategy", "outcome_score_v2"]].itertuples(index=False)
    }


def policy_to_array(lookup: Dict[tuple[str, int], float], policy: Dict[str, int], test_uids: List[str]) -> np.ndarray:
    return np.asarray([lookup.get((str(uid), int(policy.get(uid, 0))), np.nan) for uid in test_uids], dtype=float)


def fit_direct_rf_with_uncertainty(md, train_pivot: pd.DataFrame, test_uids: List[str]) -> Tuple[Dict[str, int], pd.DataFrame]:
    """Fit direct-difference multi-output RF and estimate chosen-arm uncertainty.

    The per-query uncertainty is the tree-to-tree standard deviation for the
    chosen treatment-effect prediction. It is not a causal interval, but it is a
    reasonable supervised confidence proxy for selective-routing comparison.
    """
    train_uids = train_pivot.index.astype(str).tolist()
    X_train = feature_matrix(md.features, train_uids, md.feature_cols).to_numpy()
    X_test = feature_matrix(md.features, test_uids, md.feature_cols).to_numpy()
    treatment_ids = [int(c) for c in train_pivot.columns if int(c) != 0]
    y_diff = np.column_stack(
        [(train_pivot[tid].to_numpy(dtype=float) - train_pivot[0].to_numpy(dtype=float)) for tid in treatment_ids]
    )
    rf = RandomForestRegressor(
        n_estimators=500,
        max_depth=None,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=1,
    )
    rf.fit(X_train, y_diff)
    pred = rf.predict(X_test)
    tree_pred = np.stack([tree.predict(X_test) for tree in rf.estimators_], axis=0)
    tree_sd = tree_pred.std(axis=0, ddof=1)

    rows = []
    policy: Dict[str, int] = {}
    for i, uid in enumerate(test_uids):
        gains = pred[i]
        best_j = int(np.argmax(gains))
        best_gain = float(gains[best_j])
        if best_gain > 0:
            selected = int(treatment_ids[best_j])
            selected_sd = float(tree_sd[i, best_j])
            lcb = best_gain - 1.96 * selected_sd
        else:
            selected = 0
            selected_sd = 0.0
            lcb = 0.0
        policy[uid] = selected
        sorted_gains = np.sort(gains)
        margin = float(sorted_gains[-1] - sorted_gains[-2]) if len(sorted_gains) > 1 else float(best_gain)
        rows.append(
            {
                "query_uid": uid,
                "selected_strategy": selected,
                "predicted_gain": best_gain,
                "tree_sd": selected_sd,
                "confidence_lcb": lcb,
                "top2_margin": margin,
            }
        )
    return policy, pd.DataFrame(rows)


def embedding_knn_with_confidence(md, train_pivot: pd.DataFrame, test_uids: List[str], k: int = 15) -> Tuple[Dict[str, int], pd.DataFrame]:
    sem_cols = [c for c in md.feature_cols if c.startswith("sem_")]
    train_uids = train_pivot.index.astype(str).tolist()
    X_train = feature_matrix(md.features, train_uids, sem_cols).to_numpy()
    X_test = feature_matrix(md.features, test_uids, sem_cols).to_numpy()
    sims = cosine_similarity(X_test, X_train)
    base_train = train_pivot[0].to_numpy(dtype=float)
    treatment_ids = [int(c) for c in train_pivot.columns if int(c) != 0]
    gains = np.column_stack([train_pivot[tid].to_numpy(dtype=float) - base_train for tid in treatment_ids])
    neighbor_best = []
    for row in gains:
        j = int(np.argmax(row))
        neighbor_best.append(int(treatment_ids[j]) if float(row[j]) > 0 else 0)
    neighbor_best = np.asarray(neighbor_best, dtype=int)

    rows = []
    policy: Dict[str, int] = {}
    for i, uid in enumerate(test_uids):
        nn = np.argsort(-sims[i])[: min(k, len(train_uids))]
        avg_gain = gains[nn].mean(axis=0)
        best_j = int(np.argmax(avg_gain))
        best_gain = float(avg_gain[best_j])
        selected = int(treatment_ids[best_j]) if best_gain > 0 else 0
        agreement = float(np.mean(neighbor_best[nn] == selected)) if len(nn) else 0.0
        sorted_gains = np.sort(avg_gain)
        margin = float(sorted_gains[-1] - sorted_gains[-2]) if len(sorted_gains) > 1 else float(best_gain)
        policy[uid] = selected
        rows.append(
            {
                "query_uid": uid,
                "selected_strategy": selected,
                "predicted_gain": best_gain,
                "neighbor_agreement": agreement,
                "top2_margin": margin,
                "mean_similarity": float(np.mean(sims[i, nn])) if len(nn) else 0.0,
            }
        )
    return policy, pd.DataFrame(rows)


def promptforest_confidence(md, test_uids: List[str]) -> Tuple[Dict[str, int], pd.DataFrame]:
    assignment = md.assignment.set_index("query_uid").reindex(test_uids).copy()
    assignment["optimal_treatment_id"] = pd.to_numeric(assignment["optimal_treatment_id"], errors="coerce").fillna(0).astype(int)
    assignment["expected_gain"] = pd.to_numeric(assignment["expected_gain"], errors="coerce").fillna(0.0)
    assignment["ci_lower"] = pd.to_numeric(assignment["ci_lower"], errors="coerce").fillna(-np.inf)
    policy = {uid: int(row["optimal_treatment_id"]) for uid, row in assignment.iterrows()}
    conf = assignment.reset_index()[["query_uid", "optimal_treatment_id", "expected_gain", "ci_lower"]].rename(
        columns={
            "optimal_treatment_id": "selected_strategy",
            "expected_gain": "predicted_gain",
            "ci_lower": "confidence_lcb",
        }
    )
    return policy, conf


def fixed_policy(tid: int, test_uids: List[str]) -> Dict[str, int]:
    return {uid: int(tid) for uid in test_uids}


def evaluate_policy_rows(
    model: str,
    lookup: Dict[tuple[str, int], float],
    policies: Dict[str, Dict[str, int]],
    arrays: Dict[str, np.ndarray],
    test_uids: List[str],
) -> pd.DataFrame:
    rows = []
    zero = arrays["zero_shot_direct"]
    oracle = arrays["oracle"]
    for name, policy in policies.items():
        arr = policy_to_array(lookup, policy, test_uids)
        arrays[name] = arr
    for name, arr in arrays.items():
        rows.append(
            {
                "model": model,
                "method": name,
                "policy_value": policy_value(arr),
                "gain_vs_zero_shot": policy_value(arr) - policy_value(zero),
                "regret_vs_oracle": policy_value(oracle) - policy_value(arr),
                "n_queries": int(np.sum(~np.isnan(arr))),
            }
        )
    return pd.DataFrame(rows)


def selective_routing_table(
    model: str,
    lookup: Dict[tuple[str, int], float],
    test_uids: List[str],
    oracle_arr: np.ndarray,
    fallback_name: str,
    fallback_policy: Dict[str, int],
    router_name: str,
    router_policy: Dict[str, int],
    conf_df: pd.DataFrame,
    score_col: str,
    thresholds: List[float],
) -> pd.DataFrame:
    fallback_arr = policy_to_array(lookup, fallback_policy, test_uids)
    rows = []
    conf = conf_df.set_index("query_uid").reindex(test_uids)
    for threshold in thresholds:
        selected_policy = {}
        route_mask = []
        for uid in test_uids:
            row = conf.loc[uid]
            score = float(row[score_col]) if pd.notna(row[score_col]) else -np.inf
            selected = int(router_policy.get(uid, 0))
            do_route = selected != int(fallback_policy.get(uid, 0)) and score >= threshold
            selected_policy[uid] = selected if do_route else int(fallback_policy.get(uid, 0))
            route_mask.append(do_route)
        arr = policy_to_array(lookup, selected_policy, test_uids)
        route_mask_arr = np.asarray(route_mask, dtype=bool)
        routed_gain_vs_fallback = arr[route_mask_arr] - fallback_arr[route_mask_arr]
        rows.append(
            {
                "model": model,
                "router": router_name,
                "fallback": fallback_name,
                "confidence_score": score_col,
                "threshold": threshold,
                "coverage": float(np.mean(route_mask_arr)),
                "policy_value": policy_value(arr),
                "gain_vs_fallback": policy_value(arr) - policy_value(fallback_arr),
                "regret_vs_oracle": policy_value(oracle_arr) - policy_value(arr),
                "mean_routed_gain_vs_fallback": safe_mean(routed_gain_vs_fallback),
                "routed_harm_rate": float(np.mean(routed_gain_vs_fallback < 0)) if routed_gain_vs_fallback.size else 0.0,
                "n_routed": int(route_mask_arr.sum()),
                "n_queries": int(np.sum(~np.isnan(arr))),
            }
        )
    return pd.DataFrame(rows)


def quantile_thresholds(values: pd.Series, extra: List[float] | None = None) -> List[float]:
    vals = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if vals.empty:
        return [0.0]
    qs = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    out = [float(vals.quantile(q)) for q in qs]
    if extra:
        out.extend(extra)
    return sorted(set(round(x, 8) for x in out))


def headroom_decomposition(model: str, values: pd.DataFrame) -> pd.DataFrame:
    value_map = dict(zip(values["method"], values["policy_value"]))
    oracle = float(value_map["oracle"])
    rows = []
    anchors = ["zero_shot_direct", "global_best", "task_best"]
    methods = [
        "promptforest",
        "direct_diff_rf_multioutput",
        "direct_diff_rf_combined_all",
        "direct_diff_rf_handcrafted_only",
        "direct_diff_rf_semantic_only",
        "embedding_knn_k15",
        "bm25_knn_k15",
    ]
    for anchor in anchors:
        if anchor not in value_map:
            continue
        base = float(value_map[anchor])
        headroom = oracle - base
        for method in methods:
            if method not in value_map:
                continue
            recovered = float(value_map[method]) - base
            rows.append(
                {
                    "model": model,
                    "anchor": anchor,
                    "method": method,
                    "anchor_value": base,
                    "method_value": float(value_map[method]),
                    "oracle_value": oracle,
                    "headroom": headroom,
                    "recovered_headroom": recovered,
                    "recovered_headroom_pct": recovered / headroom if headroom > 1e-12 else np.nan,
                }
            )
    return pd.DataFrame(rows)


def heterogeneity_table(model: str, train_pivot: pd.DataFrame, test_pivot: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for split, pivot in [("train", train_pivot), ("test", test_pivot)]:
        complete = pivot.dropna(axis=0, how="any")
        if complete.empty:
            continue
        base = complete[0].to_numpy(dtype=float)
        strategy_cols = [int(c) for c in complete.columns]
        gains = complete.to_numpy(dtype=float) - base[:, None]
        best = np.nanmax(complete.to_numpy(dtype=float), axis=1)
        rows.append(
            {
                "model": model,
                "split": split,
                "n_queries_complete": len(complete),
                "mean_zero_shot": float(np.mean(base)),
                "mean_oracle": float(np.mean(best)),
                "oracle_headroom_vs_zero": float(np.mean(best - base)),
                "mean_within_query_strategy_sd": float(np.mean(np.std(complete.to_numpy(dtype=float), axis=1, ddof=0))),
                "share_any_strategy_beats_zero": float(np.mean(np.nanmax(gains[:, 1:], axis=1) > 0)),
                "share_zero_is_oracle": float(np.mean(np.argmax(complete.to_numpy(dtype=float), axis=1) == strategy_cols.index(0))),
            }
        )
    return pd.DataFrame(rows)


def regret_diagnostics(
    model: str,
    md,
    lookup: Dict[tuple[str, int], float],
    test_uids: List[str],
    policies: Dict[str, Dict[str, int]],
    arrays: Dict[str, np.ndarray],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    oracle = arrays["oracle"]
    fallback = arrays["task_best"]
    meta = md.features.set_index("query_uid").reindex(test_uids)[["dataset_name", "task_type"]]
    rows = []
    per_query_rows = []
    for name, policy in policies.items():
        arr = arrays[name] if name in arrays else policy_to_array(lookup, policy, test_uids)
        regret = oracle - arr
        harm_vs_task = arr - fallback
        selected = np.asarray([int(policy.get(uid, 0)) for uid in test_uids], dtype=int)
        rows.append(
            {
                "model": model,
                "method": name,
                "mean_regret_vs_oracle": safe_mean(regret),
                "median_regret_vs_oracle": float(np.nanmedian(regret)),
                "p90_regret_vs_oracle": float(np.nanpercentile(regret, 90)),
                "share_oracle_match": float(np.nanmean(np.isclose(regret, 0.0))),
                "share_harms_task_best": float(np.nanmean(harm_vs_task < 0)),
                "mean_delta_vs_task_best": safe_mean(harm_vs_task),
                "route_rate_nonzero": float(np.mean(selected != 0)),
            }
        )
        for uid, ds, task, val, reg, delta_task, sel in zip(
            test_uids,
            meta["dataset_name"].values,
            meta["task_type"].values,
            arr,
            regret,
            harm_vs_task,
            selected,
        ):
            per_query_rows.append(
                {
                    "model": model,
                    "method": name,
                    "query_uid": uid,
                    "dataset_name": ds,
                    "task_type": task,
                    "selected_strategy": int(sel),
                    "realized_value": val,
                    "regret_vs_oracle": reg,
                    "delta_vs_task_best": delta_task,
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(per_query_rows)


def dataset_regret_summary(per_query: pd.DataFrame) -> pd.DataFrame:
    return (
        per_query.groupby(["model", "method", "dataset_name"], dropna=False)
        .agg(
            n_queries=("query_uid", "nunique"),
            policy_value=("realized_value", "mean"),
            mean_regret_vs_oracle=("regret_vs_oracle", "mean"),
            mean_delta_vs_task_best=("delta_vs_task_best", "mean"),
        )
        .reset_index()
    )


def md_table(df: pd.DataFrame, max_rows: int | None = None) -> str:
    if max_rows is not None:
        df = df.head(max_rows)
    df = df.copy()
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
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


def write_report(out_dir: Path, tables: Dict[str, pd.DataFrame]) -> Path:
    model_short = {
        "gpt-5.5": "GPT",
        "deepseek-v4-pro": "DeepSeek",
        "claude-sonnet-4.6": "Claude",
    }
    method_short = {
        "zero_shot_direct": "Zero-shot",
        "random": "Random",
        "global_best": "Global",
        "task_best": "Task-best",
        "promptforest": "PromptForest",
        "direct_diff_rf_multioutput": "Direct RF",
        "direct_diff_rf_combined_all": "RF combined",
        "direct_diff_rf_handcrafted_only": "RF handcraft",
        "direct_diff_rf_semantic_only": "RF semantic",
        "embedding_knn_k15": "Embed-kNN",
        "bm25_knn_k15": "BM25-kNN",
        "oracle": "Oracle",
    }
    score_short = {
        "confidence_lcb": "LCB",
        "neighbor_agreement": "Agree",
        "top2_margin": "Margin",
    }

    primary = tables["primary_policy_values"].copy()
    primary["model"] = primary["model"].map(model_short).fillna(primary["model"])
    primary["method"] = primary["method"].map(method_short).fillna(primary["method"])
    primary_wide = primary.pivot(index="method", columns="model", values="policy_value").round(4).reset_index()
    selective_best = (
        tables["selective_routing"]
        .sort_values(["model", "fallback", "router", "policy_value"], ascending=[True, True, True, False])
        .groupby(["model", "fallback", "router"], as_index=False)
        .head(1)
        [["model", "router", "fallback", "confidence_score", "threshold", "coverage", "policy_value", "gain_vs_fallback", "routed_harm_rate"]]
        .round(4)
    )
    selective_best = selective_best.rename(
        columns={
            "model": "M",
            "router": "Rtr",
            "fallback": "Fb",
            "confidence_score": "Sc",
            "threshold": "Th",
            "coverage": "Cov.",
            "policy_value": "V",
            "gain_vs_fallback": "Gain",
            "routed_harm_rate": "Harm",
        }
    )
    selective_model_short = {
        "gpt-5.5": "GPT",
        "deepseek-v4-pro": "DS",
        "claude-sonnet-4.6": "Cl",
    }
    selective_method_short = {
        "promptforest": "PF",
        "direct_diff_rf_multioutput": "RF",
        "embedding_knn_k15": "kNN",
        "task_best": "Task",
    }
    selective_best["M"] = selective_best["M"].map(selective_model_short).fillna(selective_best["M"])
    selective_best["Rtr"] = selective_best["Rtr"].map(selective_method_short).fillna(selective_best["Rtr"])
    selective_best["Fb"] = selective_best["Fb"].map(selective_method_short).fillna(selective_best["Fb"])
    selective_best["Sc"] = selective_best["Sc"].map(score_short).fillna(selective_best["Sc"])

    heterogeneity = tables["heterogeneity"].copy().round(4)
    heterogeneity["model"] = heterogeneity["model"].map(model_short).fillna(heterogeneity["model"])
    heterogeneity = heterogeneity.rename(
        columns={
            "model": "Model",
            "split": "Split",
            "n_queries_complete": "N",
            "mean_zero_shot": "Zero",
            "mean_oracle": "Oracle",
            "oracle_headroom_vs_zero": "Headroom",
            "mean_within_query_strategy_sd": "Within-SD",
            "share_any_strategy_beats_zero": "Any>Zero",
            "share_zero_is_oracle": "Zero=Oracle",
        }
    )

    headroom = tables["headroom"].copy().round(4)
    headroom_method_short = {
        "zero_shot_direct": "ZS",
        "global_best": "Global",
        "task_best": "Task",
        "promptforest": "PF",
        "direct_diff_rf_multioutput": "RF",
        "direct_diff_rf_combined_all": "RF-all",
        "direct_diff_rf_handcrafted_only": "RF-H",
        "direct_diff_rf_semantic_only": "RF-S",
        "embedding_knn_k15": "kNN",
        "bm25_knn_k15": "BM25",
    }
    headroom["model"] = headroom["model"].map(selective_model_short).fillna(headroom["model"])
    headroom["method"] = headroom["method"].map(headroom_method_short).fillna(headroom["method"])
    headroom["anchor"] = headroom["anchor"].map(headroom_method_short).fillna(headroom["anchor"])
    headroom = headroom.rename(
        columns={
            "model": "M",
            "anchor": "Anchor",
            "method": "Method",
            "anchor_value": "Base",
            "method_value": "Val",
            "oracle_value": "Oracle",
            "headroom": "Head",
            "recovered_headroom": "Rec",
            "recovered_headroom_pct": "Rec%",
        }
    )

    regret = tables["regret_diagnostics"].copy().round(4)
    regret_method_short = {
        "promptforest": "PF",
        "direct_diff_rf_multioutput": "RF",
        "embedding_knn_k15": "kNN",
        "bm25_knn_k15": "BM25",
        "task_best": "Task",
        "global_best": "Global",
    }
    regret["model"] = regret["model"].map(selective_model_short).fillna(regret["model"])
    regret["method"] = regret["method"].map(regret_method_short).fillna(regret["method"])
    regret = regret.rename(
        columns={
            "model": "M",
            "method": "Method",
            "mean_regret_vs_oracle": "Mean",
            "median_regret_vs_oracle": "Med",
            "p90_regret_vs_oracle": "P90",
            "share_oracle_match": "Match",
            "share_harms_task_best": "Harm",
            "mean_delta_vs_task_best": "Delta",
            "route_rate_nonzero": "Route",
        }
    )
    lines = [
        "# PromptForest Resubmission P1-P5 Experiment Results",
        "",
        "Generated from saved full-factorial results. No new LLM/API calls were made.",
        "",
        "## P1 Primary Baselines",
        "",
        md_table(primary_wide),
        "",
        "## P2 Selective Routing: Best Threshold per Router/Fallback",
        "",
        md_table(selective_best),
        "",
        "## P3 Heterogeneity and Oracle Headroom",
        "",
        md_table(heterogeneity),
        "",
        "## P4 Headroom Recovery",
        "",
        md_table(headroom, max_rows=45),
        "",
        "## P5 Regret Diagnostics",
        "",
        md_table(regret),
        "",
        "## Reading Notes",
        "",
        "- Direct RF and kNN are promoted to primary comparisons, as requested by the reviewer.",
        "- PromptForest is evaluated both as a full-coverage router and as an uncertainty-aware selective router.",
        "- The selective-routing rows should be interpreted as offline diagnostics; thresholds are swept on the held-out set to map the coverage-risk frontier, not to claim a tuned deployment result.",
        "- If PromptForest does not dominate the selective frontier, the next manuscript should frame it as one diagnostic estimator rather than the routing method of choice.",
        "",
    ]
    report_path = out_dir / "resubmission_p1_p5_summary.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def run_model(md, k: int) -> Dict[str, pd.DataFrame]:
    train_pivot, test_pivot = train_test_pivots(md)
    test_uids = test_pivot.index.astype(str).tolist()
    lookup = outcome_lookup(md, test_uids)
    zero_policy = fixed_policy(0, test_uids)
    task_policy = task_best_policy(md, train_pivot, test_uids)
    global_policy = global_best_policy(train_pivot, test_uids, require_positive=False)
    pf_policy, pf_conf = promptforest_confidence(md, test_uids)
    direct_policy, direct_conf = fit_direct_rf_with_uncertainty(md, train_pivot, test_uids)
    emb_policy, emb_conf = embedding_knn_with_confidence(md, train_pivot, test_uids, k=k)
    bm25_policy = bm25_knn_policy(md, train_pivot, test_uids, k=k)
    feature_group_policies = {}
    for group_name, cols in feature_groups(md.feature_cols).items():
        if cols:
            feature_group_policies[f"direct_diff_rf_{group_name}"] = fit_direct_diff_rf_with_features(
                md, train_pivot, test_uids, cols
            )

    arrays = {
        "zero_shot_direct": policy_to_array(lookup, zero_policy, test_uids),
        "random": random_scores(test_pivot, test_uids),
        "oracle": oracle_scores(test_pivot, test_uids),
    }
    policies = {
        "global_best": global_policy,
        "task_best": task_policy,
        "promptforest": pf_policy,
        "direct_diff_rf_multioutput": direct_policy,
        f"embedding_knn_k{k}": emb_policy,
        f"bm25_knn_k{k}": bm25_policy,
    }
    policies.update(feature_group_policies)
    primary = evaluate_policy_rows(md.model, lookup, policies, arrays, test_uids)

    thresholds = []
    thresholds.append(("promptforest", pf_policy, pf_conf, "confidence_lcb", quantile_thresholds(pf_conf["confidence_lcb"], extra=[0.0])))
    thresholds.append(("direct_diff_rf_multioutput", direct_policy, direct_conf, "confidence_lcb", quantile_thresholds(direct_conf["confidence_lcb"], extra=[0.0])))
    thresholds.append((f"embedding_knn_k{k}", emb_policy, emb_conf, "neighbor_agreement", quantile_thresholds(emb_conf["neighbor_agreement"], extra=[0.5, 0.6, 0.7, 0.8])))
    thresholds.append((f"embedding_knn_k{k}", emb_policy, emb_conf, "top2_margin", quantile_thresholds(emb_conf["top2_margin"], extra=[0.0])))

    selective_parts = []
    fallback_specs = [("task_best", task_policy), ("direct_diff_rf_multioutput", direct_policy)]
    for router_name, router_policy, conf, score_col, thrs in thresholds:
        for fallback_name, fallback_policy in fallback_specs:
            if router_name == fallback_name:
                continue
            selective_parts.append(
                selective_routing_table(
                    md.model,
                    lookup,
                    test_uids,
                    arrays["oracle"],
                    fallback_name,
                    fallback_policy,
                    router_name,
                    router_policy,
                    conf,
                    score_col,
                    thrs,
                )
            )
    selective = pd.concat(selective_parts, ignore_index=True)
    hetero = heterogeneity_table(md.model, train_pivot, test_pivot)
    headroom = headroom_decomposition(md.model, primary)

    diag_policies = {
        "promptforest": pf_policy,
        "direct_diff_rf_multioutput": direct_policy,
        "embedding_knn_k15": emb_policy,
        "bm25_knn_k15": bm25_policy,
        "task_best": task_policy,
        "global_best": global_policy,
    }
    diag, per_query_diag = regret_diagnostics(md.model, md, lookup, test_uids, diag_policies, primary_to_arrays(primary, arrays, lookup, diag_policies, test_uids))
    per_query_values = pd.DataFrame({"query_uid": test_uids, "model": md.model})
    for name in ["zero_shot_direct", "oracle", "random"]:
        per_query_values[name] = arrays[name]
    for name, policy in policies.items():
        per_query_values[name] = policy_to_array(lookup, policy, test_uids)
    return {
        "primary_policy_values": primary,
        "selective_routing": selective,
        "heterogeneity": hetero,
        "headroom": headroom,
        "regret_diagnostics": diag,
        "per_query_regret": per_query_diag,
        "dataset_regret": dataset_regret_summary(per_query_diag),
        "per_query_policy_values": per_query_values,
        "promptforest_confidence": pf_conf.assign(model=md.model),
        "direct_rf_confidence": direct_conf.assign(model=md.model),
        "embedding_knn_confidence": emb_conf.assign(model=md.model),
    }


def primary_to_arrays(
    primary: pd.DataFrame,
    arrays: Dict[str, np.ndarray],
    lookup: Dict[tuple[str, int], float],
    policies: Dict[str, Dict[str, int]],
    test_uids: List[str],
) -> Dict[str, np.ndarray]:
    out = dict(arrays)
    for name, policy in policies.items():
        out[name] = policy_to_array(lookup, policy, test_uids)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--out", default="data/resubmission_p1_p5")
    parser.add_argument("--k", type=int, default=15)
    args = parser.parse_args()

    data_root = Path(args.data_root)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    collected: Dict[str, List[pd.DataFrame]] = {}
    for model in MODELS:
        print(f"[P1-P5] Processing {model}")
        md = load_model_data(data_root, model)
        model_tables = run_model(md, k=args.k)
        model_dir = out_dir / f"{model}_n100"
        model_dir.mkdir(parents=True, exist_ok=True)
        for name, df in model_tables.items():
            df.to_csv(model_dir / f"{name}_{model}.csv", index=False)
            collected.setdefault(name, []).append(df)

    all_tables = {name: pd.concat(parts, ignore_index=True) for name, parts in collected.items()}
    for name, df in all_tables.items():
        df.to_csv(out_dir / f"{name}_all_models.csv", index=False)

    report_path = write_report(out_dir, all_tables)
    print(f"[P1-P5] Wrote CSV artifacts to {out_dir.resolve()}")
    print(f"[P1-P5] Wrote summary report to {report_path.resolve()}")


if __name__ == "__main__":
    main()
