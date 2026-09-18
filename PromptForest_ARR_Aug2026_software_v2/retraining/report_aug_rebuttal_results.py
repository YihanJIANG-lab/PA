"""Build tables and standalone figures from the additional offline experiments."""
from pathlib import Path
import importlib.metadata
import json
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from aug_rebuttal_experiments import ROOT, MODELS

LABEL = {"gpt-5.5":"GPT-5.5", "deepseek-v4-pro":"DeepSeek-v4-Pro", "claude-sonnet-4.6":"Claude-Sonnet-4.6"}


def md_table(frame):
    def fmt(value):
        return f"{value:.4f}" if isinstance(value, (float,np.floating)) else str(value)
    return "\n".join(["| " + " | ".join(map(str,frame.columns)) + " |", "| " + " | ".join(["---"]*len(frame.columns)) + " |"] + ["| " + " | ".join(fmt(v) for v in row) + " |" for row in frame.itertuples(index=False,name=None)])


def save(fig, directory, name):
    fig.savefig(directory / f"{name}.png", dpi=180)
    fig.savefig(directory / f"{name}.pdf")
    plt.close(fig)


def build(out):
    figures = out / "figures"
    figures.mkdir(exist_ok=True)
    summary = pd.read_csv(out / "policy_summary.csv")
    split = pd.read_csv(out / "policy_by_split.csv")
    gates = pd.read_csv(out / "validated_gates.csv")
    calibration = pd.read_csv(out / "calibration_by_split.csv")
    pairs = pd.read_csv(out / "paired_difference_summary.csv")
    methods = ["Zero","Task_mean","Dataset_mean","CF_grouped","CF_original_gate","Diff_RF","Diff_RF_flexible","Diff_RF_task_only","Diff_Ridge","Outcome_RF","Embedding_kNN"]
    policy = summary[(summary.stage == "outer") & summary.method.isin(methods)].copy()
    policy["value_mean_sd"] = policy.apply(lambda r: f"{r.value_mean:.4f} +/- {r.value_sd:.4f}",axis=1)
    table = policy.pivot(index="method",columns="model",values="value_mean_sd").reindex(methods).reset_index()
    table.to_csv(out / "table_policy_stability.csv",index=False)
    gate_summary = gates.groupby(["model","router"]).agg(value=("test_value","mean"),gain=("test_gain","mean"),gain_sd=("test_gain","std"),coverage=("test_coverage","mean"),harm_all=("harm_all","mean"),negative_magnitude=("mean_negative_magnitude","mean"),wins=("test_gain",lambda x:int((x>1e-12).sum())),never_route=("threshold",lambda x:int(np.isinf(x).sum()))).reset_index()
    gate_summary.to_csv(out / "table_validated_gates.csv",index=False)
    cal_summary = calibration[(calibration.stage=="outer") & (calibration.arm=="pooled")].groupby(["model","method"])[["rmse","mae","calibration_slope","calibration_intercept","spearman","within_task_arm_correlation"]].mean().reset_index()
    cal_summary.to_csv(out / "table_effect_calibration.csv",index=False)
    sensitivity = split[split.stage=="sensitivity"].copy()
    main42 = split[(split.seed==42)&(split.stage=="outer")&split.method.isin(["CF_grouped","CF_original_gate","Task_mean","Zero"])].copy()
    sensitivity = pd.concat([sensitivity, main42],ignore_index=True)
    sensitivity.to_csv(out / "table_protocol_sensitivity.csv",index=False)
    propensity_checks = []
    for model in MODELS:
        main_effect = pd.read_csv(out / "runs" / f"{model}_seed42/effect_predictions.csv")
        sensitivity_effect = pd.read_csv(out / "runs" / f"{model}_seed42_sensitivity/effect_predictions.csv")
        main_effect = main_effect[main_effect.method == "CF_grouped"].sort_values(["query_uid","arm"])
        known_effect = sensitivity_effect[sensitivity_effect.method == "CF_known_propensity_500"].sort_values(["query_uid","arm"])
        np.testing.assert_array_equal(main_effect.query_uid, known_effect.query_uid)
        np.testing.assert_array_equal(main_effect.arm, known_effect.arm)
        propensity_checks.append(dict(model=model, maximum_absolute_effect_difference=float(np.max(np.abs(main_effect.predicted_gain.to_numpy()-known_effect.predicted_gain.to_numpy())))))
    propensity_checks = pd.DataFrame(propensity_checks)
    propensity_checks.to_csv(out / "known_propensity_prediction_check.csv",index=False)
    plt.rcParams.update({"font.size":9,"axes.spines.top":False,"axes.spines.right":False})
    fig,axes = plt.subplots(1,3,figsize=(13,4.2),layout="constrained",sharey=True)
    shown = ["CF_grouped","CF_original_gate","Diff_RF","Diff_RF_flexible","Embedding_kNN"]
    labels = ["CF argmax","CF gated rule","RF leaf=10","RF flexible","Embedding kNN"]
    colors = ["#3b7a57","#aa3377","#4477aa","#ddaa33","#228899"]
    for ax,model in zip(axes,MODELS):
        sub = split[(split.stage=="outer")&(split.model==model)]
        for i,(method,color) in enumerate(zip(shown,colors)):
            values = sub[sub.method==method].sort_values("seed").gain_task.to_numpy()
            ax.scatter(np.arange(len(values))*.025+i-.11,values,color=color,s=22,alpha=.8)
            ax.plot([i-.2,i+.2],[values.mean()]*2,color="black",lw=2)
        ax.axhline(0,color="gray",ls="--",lw=1)
        ax.set_xticks(range(len(shown)),labels,rotation=35,ha="right")
        ax.set_title(LABEL[model])
    axes[0].set_ylabel("Held-out value minus Task-Best")
    fig.suptitle("Ten paired split comparisons (dots); means (black lines). No independence assumed.")
    save(fig,figures,"policy_split_stability")
    fig,axes = plt.subplots(1,3,figsize=(13,4),layout="constrained",sharey=True)
    routers = ["CF_grouped","Diff_RF","Embedding_kNN"]
    for ax,model in zip(axes,MODELS):
        sub = gates[gates.model==model]
        for i,router in enumerate(routers):
            values = sub[sub.router==router].sort_values("seed").test_gain.to_numpy()
            ax.scatter(np.arange(len(values))*.025+i-.11,values,s=25,color=colors[i])
            ax.plot([i-.2,i+.2],[values.mean()]*2,color="black",lw=2)
        ax.axhline(0,color="gray",ls="--",lw=1)
        ax.set_xticks(range(3),["CF score","RF score","kNN score"])
        ax.set_title(LABEL[model])
    axes[0].set_ylabel("Held-out gain over the selected fallback")
    fig.suptitle("Threshold and fallback selected on validation only; same fitted model at test time")
    save(fig,figures,"independent_gate_validation")
    bins = pd.read_csv(out / "calibration_bins.csv")
    fig,axes = plt.subplots(1,3,figsize=(13,4),layout="constrained",sharex=True,sharey=True)
    for ax,model in zip(axes,MODELS):
        sub = bins[(bins.model==model)&(bins.stage=="outer")&(bins.seed==42)]
        for method,color in zip(["CF_grouped","Diff_RF","Task_mean"],colors):
            part = sub[sub.method==method]
            ax.scatter(part.predicted_mean,part.observed_mean,s=20,color=color,alpha=.65,label=method)
        ax.axline((0,0),slope=1,color="gray",ls="--",lw=1)
        ax.set_title(LABEL[model]);ax.set_xlabel("Predicted bin mean effect")
    axes[0].set_ylabel("Observed bin mean paired score difference")
    axes[-1].legend(fontsize=8)
    fig.suptitle("Seed 42 effect calibration: bins formed separately for each strategy; noisy targets")
    save(fig,figures,"effect_calibration_seed42")
    env = {}
    for package in ["numpy","pandas","scipy","scikit-learn","econml","matplotlib","sentence-transformers","torch"]:
        try: env[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError: pass
    (out / "environment_versions.json").write_text(json.dumps(env,indent=2),encoding="utf-8")
    verified = out / "verification.csv"
    verified_count = len(pd.read_csv(verified)) if verified.exists() else 0
    lines = ["# ARR August Additional Experiment Results", "", f"Verified completed jobs: {verified_count}/33. All 30 primary jobs and three protocol-sensitivity jobs must pass verification.csv before using this report.",
             "", "## Protocol and provenance", "", "The original saved semantic features are reproduced by multilingual-e5-base, not the MiniLM model named in the paper. See inputs/embedding_metadata.json. New experiments refit PCA inside each training split and group nuisance cross-fitting by query. These are new controlled reruns, not exact reproductions of the original manuscript numbers.",
             "", "GPT has five incomplete queries, all in HotpotQA. They are excluded independently of policy choice; the two other models have complete outcomes. See cohort_audit.csv and each split_manifest.csv.",
             "", "## Policy stability", "", "Mean +/- SD across ten overlapping splits; descriptive, not ten independent experiments. CF_original_gate denotes the original gate decision rule applied to the NEW grouped-cross-fit model, not the stored original manuscript policy. Task_mean uses nine task categories. Dataset_mean uses 11 benchmark identities and is a benchmark-aware diagnostic comparator, not a deployment baseline when benchmark identity is unavailable. RF leaf/depth/tree settings are nominally matched; row-based CF leaves and query-based RF leaves do not imply equal capacity.","",md_table(table),
             "", "## Paired comparisons", "",md_table(pairs[pairs.method.isin(["CF_grouped","CF_original_gate","Diff_RF"]) & pairs.baseline.isin(["Task_mean","Diff_RF"])]),
             "", "## Effect prediction", "", "Metrics use noisy observed differences, not ground-truth CATE. Within-task/arm correlation removes group means descriptively and is not a prospective residualization procedure.","",md_table(cal_summary[cal_summary.method.isin(["CF_grouped","Diff_RF","Diff_RF_task_only","Task_mean"])]),
             "", "## Independent gate validation", "", "Comparisons are to each validation-selected fallback trained on the same inner training queries. Coverage is the fraction switched; ties are distinct from strictly harmful switches. No confidence-interval coverage guarantee is asserted.","",md_table(gate_summary),
             "", "## Protocol sensitivities", "",md_table(sensitivity[["model","method","n","value","gain_task"]]),
             "", "Seed 42 learned versus constant propensity prediction differences:", "", md_table(propensity_checks),
             "", "## Reviewer mapping", "", "- mBTK: policy stability, direct-difference baselines, task-only comparisons, and protocol sensitivities address estimator choice and incremental query-level value. Neither significance against Zero-Shot nor a non-significant difference proves superiority or equivalence to Task-Best.",
             "- zGRZ: repeated full retraining, held-out effect calibration, and independently selected gates address stability, calibration, and deployment reliability. Ten resplits do not increase the number of independent collected queries, and these tests do not establish robustness to future model-version updates.",
             "- fcLf: preserve the original Holm-corrected inference and distinguish observed numerical gains from statistically supported gains. The new descriptive split analysis does not replace or reverse the original hypothesis family.",
             "", "## Remaining limits", "", "Native forest intervals rely on row-level internal subsampling/honesty despite grouped nuisance cross-fitting. Effect calibration and validation-gate performance are empirical diagnostics, not formal interval guarantees. The reported latency observations exclude router overhead and are not a controlled deployment cost benchmark. Strategy template documentation and this retraining package are shipped in the same software package (prompts/ and retraining/)."]
    (out / "RESULTS.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(f"Wrote {out / 'RESULTS.md'} and three PNG/PDF figures")


if __name__ == "__main__":
    build(Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "rebuttal_outputs")
