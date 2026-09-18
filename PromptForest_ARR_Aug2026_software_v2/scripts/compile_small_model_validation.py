"""Compile smaller/open-weight validation-slice summaries.

Inputs are the four compact validation-slice summary directories in the
companion data package:

  paper_results/rebuttal_small_model_inputs/

The script regenerates the Appendix-I summary CSV files without calling any
external APIs.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


MODEL_DIRS = [
    ("Qwen2.5-7B-Instruct", "rebuttal_p1_qwen"),
    ("Gemini-2.5-Flash-Lite", "rebuttal_p1_gemini"),
    ("GPT-4o-mini", "rebuttal_p1_gpt4omini"),
    ("Llama-3.2-3B-Instruct", "rebuttal_p1_llama32_3b"),
]


def read_first(path: Path, pattern: str) -> pd.DataFrame:
    matches = sorted(path.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No file matching {pattern} in {path}")
    return pd.read_csv(matches[0])


def markdown_table(df: pd.DataFrame) -> str:
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, default=Path("../PromptForest_ARR_Aug2026_data/paper_results/rebuttal_small_model_inputs"))
    parser.add_argument("--out", type=Path, default=Path("reproduced/rebuttal_small_models"))
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    fixed_rows = []
    routing_rows = []
    sensitivity_rows = []
    for model, dirname in MODEL_DIRS:
        directory = args.input_root / dirname
        fixed = read_first(directory, "*fixed_strategy_and_oracle.csv")
        routing = read_first(directory, "*routing_policy_values.csv")
        sensitivity = read_first(directory, "*strategy_sensitivity_summary.csv")
        fixed.insert(0, "model", model)
        routing.insert(0, "model", model)
        sensitivity.insert(0, "model", model)
        fixed_rows.append(fixed)
        routing_rows.append(routing)
        sensitivity_rows.append(sensitivity)

    fixed_all = pd.concat(fixed_rows, ignore_index=True)
    routing_all = pd.concat(routing_rows, ignore_index=True)
    sensitivity_all = pd.concat(sensitivity_rows, ignore_index=True)
    fixed_all.to_csv(args.out / "small_model_fixed_strategy_and_oracle_all.csv", index=False)
    routing_all.to_csv(args.out / "small_model_routing_policy_values_all.csv", index=False)
    sensitivity_all.to_csv(args.out / "small_model_strategy_sensitivity_summary_all.csv", index=False)

    key_rows = []
    for model in fixed_all["model"].unique():
        sub = fixed_all[fixed_all["model"] == model].copy()
        oracle = float(sub[sub["method"] == "Oracle"]["policy_value"].iloc[0])
        zero = float(sub[sub["method"].str.contains("zero_shot_direct")]["policy_value"].iloc[0])
        random = float(sub[sub["method"] == "Random"]["policy_value"].iloc[0])
        candidates = sub[~sub["method"].isin(["Oracle", "Random"])].copy()
        candidates["policy_value"] = pd.to_numeric(candidates["policy_value"], errors="coerce")
        best_fixed = candidates.sort_values("policy_value", ascending=False).iloc[0]
        route_sub = routing_all[routing_all["model"] == model].copy()
        route_sub["policy_value"] = pd.to_numeric(route_sub["policy_value"], errors="coerce")
        best_route = route_sub[route_sub["method"] != "oracle"].sort_values("policy_value", ascending=False).iloc[0]
        key_rows.append(
            {
                "model": model,
                "zero_shot": zero,
                "best_fixed_method": best_fixed["method"],
                "best_fixed": float(best_fixed["policy_value"]),
                "random": random,
                "oracle": oracle,
                "oracle_gain_vs_zero": oracle - zero,
                "best_routing_method": best_route["method"],
                "best_routing": float(best_route["policy_value"]),
                "routing_gain_vs_zero": float(best_route["policy_value"]) - zero,
            }
        )
    key = pd.DataFrame(key_rows)
    key.to_csv(args.out / "small_model_key_summary.csv", index=False)

    lines = [
        "# Smaller/Open-Weight Validation Slice Summary",
        "",
        "Generated from the compact validation-slice summaries in the companion data package.",
        "",
        markdown_table(key.round(4)),
        "",
    ]
    (args.out / "small_model_rebuttal_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
