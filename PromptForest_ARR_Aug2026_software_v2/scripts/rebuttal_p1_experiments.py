"""Run P1 rebuttal analyses that require no new LLM/API calls.

This script adds:
  1. Feature-family ablation for direct outcome-difference routing:
     semantic-only vs handcrafted-only vs combined features.
  2. Similarity-based routing baselines:
     embedding kNN and lexical BM25 kNN.

It reuses the same filtering/split/evaluation helpers as the P0 rebuttal script.
Outputs are written under data/rebuttal_p1/.
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics.pairwise import cosine_similarity

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from rebuttal_p0_experiments import (  # noqa: E402
    MODELS,
    choose_by_predicted_gain,
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


def feature_groups(feature_cols: List[str]) -> Dict[str, List[str]]:
    semantic = [c for c in feature_cols if c.startswith("sem_")]
    handcrafted = [c for c in feature_cols if not c.startswith("sem_")]
    return {
        "semantic_only": semantic,
        "handcrafted_only": handcrafted,
        "combined_all": feature_cols,
    }


def fit_direct_diff_rf_with_features(md, train_pivot: pd.DataFrame, test_uids: List[str], cols: List[str]) -> Dict[str, int]:
    train_uids = train_pivot.index.astype(str).tolist()
    X_train = feature_matrix(md.features, train_uids, cols).to_numpy()
    X_test = feature_matrix(md.features, test_uids, cols).to_numpy()
    treatment_ids = [int(c) for c in train_pivot.columns if int(c) != 0]
    y_diff = np.column_stack(
        [(train_pivot[tid].to_numpy(dtype=float) - train_pivot[0].to_numpy(dtype=float)) for tid in treatment_ids]
    )
    model = RandomForestRegressor(
        n_estimators=500,
        max_depth=None,
        min_samples_leaf=3,
        random_state=2026,
        n_jobs=1,
    )
    model.fit(X_train, y_diff)
    pred = model.predict(X_test)
    return choose_by_predicted_gain(test_uids, treatment_ids, pred)


def embedding_knn_policy(md, train_pivot: pd.DataFrame, test_uids: List[str], k: int = 15) -> Dict[str, int]:
    sem_cols = [c for c in md.feature_cols if c.startswith("sem_")]
    train_uids = train_pivot.index.astype(str).tolist()
    X_train = feature_matrix(md.features, train_uids, sem_cols).to_numpy()
    X_test = feature_matrix(md.features, test_uids, sem_cols).to_numpy()
    sims = cosine_similarity(X_test, X_train)
    base_train = train_pivot[0].to_numpy(dtype=float)
    treatment_ids = [int(c) for c in train_pivot.columns if int(c) != 0]
    gains = np.column_stack([train_pivot[tid].to_numpy(dtype=float) - base_train for tid in treatment_ids])
    policy = {}
    for i, uid in enumerate(test_uids):
        nn = np.argsort(-sims[i])[: min(k, len(train_uids))]
        avg_gain = gains[nn].mean(axis=0)
        best = int(np.argmax(avg_gain))
        policy[uid] = int(treatment_ids[best]) if float(avg_gain[best]) > 0 else 0
    return policy


TOKEN_RE = re.compile(r"[A-Za-z0-9_]+", re.UNICODE)


def tokenize(text: object) -> List[str]:
    if pd.isna(text):
        return []
    return [t.lower() for t in TOKEN_RE.findall(str(text))]


def bm25_knn_policy(md, train_pivot: pd.DataFrame, test_uids: List[str], k: int = 15, k1: float = 1.5, b: float = 0.75) -> Dict[str, int]:
    train_uids = train_pivot.index.astype(str).tolist()
    meta = md.features.set_index("query_uid")
    train_docs = [tokenize(meta.loc[uid, "query_text"]) for uid in train_uids]
    test_docs = [tokenize(meta.loc[uid, "query_text"]) for uid in test_uids]
    n_docs = len(train_docs)
    doc_lens = np.array([len(d) for d in train_docs], dtype=float)
    avgdl = float(doc_lens.mean()) if n_docs else 1.0
    dfs = Counter()
    tfs = []
    for doc in train_docs:
        counts = Counter(doc)
        tfs.append(counts)
        dfs.update(counts.keys())

    idf = {term: math.log(1 + (n_docs - df + 0.5) / (df + 0.5)) for term, df in dfs.items()}
    base_train = train_pivot[0].to_numpy(dtype=float)
    treatment_ids = [int(c) for c in train_pivot.columns if int(c) != 0]
    gains = np.column_stack([train_pivot[tid].to_numpy(dtype=float) - base_train for tid in treatment_ids])

    policy = {}
    for uid, qdoc in zip(test_uids, test_docs):
        q_terms = set(qdoc)
        scores = np.zeros(n_docs, dtype=float)
        for j, counts in enumerate(tfs):
            dl = doc_lens[j] if doc_lens[j] > 0 else 1.0
            score = 0.0
            for term in q_terms:
                tf = counts.get(term, 0)
                if tf == 0:
                    continue
                denom = tf + k1 * (1 - b + b * dl / avgdl)
                score += idf.get(term, 0.0) * (tf * (k1 + 1)) / denom
            scores[j] = score
        nn = np.argsort(-scores)[: min(k, n_docs)]
        avg_gain = gains[nn].mean(axis=0)
        best = int(np.argmax(avg_gain))
        policy[uid] = int(treatment_ids[best]) if float(avg_gain[best]) > 0 else 0
    return policy


def evaluate_model(md, k: int) -> pd.DataFrame:
    train_pivot, test_pivot = train_test_pivots(md)
    test_uids = test_pivot.index.astype(str).tolist()
    test_df = md.valid_df[md.valid_df["query_uid"].astype(str).isin(test_uids)].copy()
    baseline = policy_value(realized_scores(test_df, {uid: 0 for uid in test_uids}, test_uids))

    rows = []

    policies = {
        "zero_shot_direct": {uid: 0 for uid in test_uids},
        "promptforest": promptforest_policy(md, test_uids),
        "task_best": task_best_policy(md, train_pivot, test_uids),
        "global_best": global_best_policy(train_pivot, test_uids, require_positive=False),
        f"embedding_knn_k{k}": embedding_knn_policy(md, train_pivot, test_uids, k=k),
        f"bm25_knn_k{k}": bm25_knn_policy(md, train_pivot, test_uids, k=k),
    }

    for group_name, cols in feature_groups(md.feature_cols).items():
        if cols:
            policies[f"direct_diff_rf_{group_name}"] = fit_direct_diff_rf_with_features(md, train_pivot, test_uids, cols)

    arrays = {}
    for name, policy in policies.items():
        arrays[name] = realized_scores(test_df, policy, test_uids)
    arrays["random"] = random_scores(test_pivot, test_uids)
    arrays["oracle"] = oracle_scores(test_pivot, test_uids)

    for name, arr in arrays.items():
        rows.append(
            {
                "model": md.model,
                "method": name,
                "policy_value": policy_value(arr),
                "gain_vs_zero_shot": policy_value(arr) - baseline,
                "n_queries": int(np.sum(~np.isnan(arr))),
            }
        )
    return pd.DataFrame(rows).sort_values("policy_value", ascending=False)


def write_summary(out_dir: Path, all_df: pd.DataFrame) -> None:
    def md_table(df: pd.DataFrame) -> str:
        cols = list(df.columns)
        lines = [
            "| " + " | ".join(cols) + " |",
            "| " + " | ".join(["---"] * len(cols)) + " |",
        ]
        for _, row in df.iterrows():
            vals = []
            for col in cols:
                val = row[col]
                vals.append(f"{val:.4f}" if isinstance(val, float) else str(val))
            lines.append("| " + " | ".join(vals) + " |")
        return "\n".join(lines)

    wide = all_df.pivot(index="method", columns="model", values="policy_value").round(4).reset_index()
    feature_rows = all_df[all_df["method"].str.contains("semantic|handcrafted|combined", regex=True)]
    routing_rows = all_df[all_df["method"].str.contains("knn|promptforest|task_best|zero_shot", regex=True)]
    lines = [
        "# P1 Rebuttal Results Summary",
        "",
        "Generated from saved full-factorial results; no new LLM/API calls.",
        "",
        "## All policy values",
        "",
        md_table(wide),
        "",
        "## Feature-family ablation",
        "",
        md_table(feature_rows[["model", "method", "policy_value", "gain_vs_zero_shot", "n_queries"]].round(4)),
        "",
        "## Similarity routing baselines",
        "",
        md_table(routing_rows[["model", "method", "policy_value", "gain_vs_zero_shot", "n_queries"]].round(4)),
        "",
    ]
    (out_dir / "p1_rebuttal_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--out", default="data/rebuttal_p1")
    parser.add_argument("--k", type=int, default=15)
    args = parser.parse_args()

    data_root = Path(args.data_root)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    for model in MODELS:
        print(f"[P1] Processing {model}")
        md = load_model_data(data_root, model)
        df = evaluate_model(md, k=args.k)
        mdir = out_dir / f"{model}_n100"
        mdir.mkdir(parents=True, exist_ok=True)
        df.to_csv(mdir / f"p1_feature_and_similarity_routing_{model}.csv", index=False)
        all_rows.append(df)

    all_df = pd.concat(all_rows, ignore_index=True)
    all_df.to_csv(out_dir / "p1_feature_and_similarity_routing_all_models.csv", index=False)
    write_summary(out_dir, all_df)
    print(f"[P1] Wrote rebuttal artifacts to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
