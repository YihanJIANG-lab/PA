# PromptForest ARR August 2026 Software Package (v2)

Anonymous software package for the ARR August 2026 version of the PromptForest
paper. Version 2 adds the two deliverables requested in review: **an actual
retraining path for the core method** (causal forest + gating, from the saved
full outcome matrix) and **complete strategy template documentation**. No LLM
API calls are made by anything in this package.

## (a) Contents — three layers

```text
retraining/                        NEW: retrain the core method end-to-end
  aug_rebuttal_experiments.py        main retraining runner (10 splits x 3 models)
  prepare_aug_embeddings.py          rebuild raw 768-dim embeddings (optional)
  verify_aug_rebuttal_results.py     artifact verification (split isolation, fingerprints)
  report_aug_rebuttal_results.py     tables + figures + RESULTS.md
scripts/                           offline table recomputation (unchanged from v1)
  resubmission_p1_p5_experiments.py  main resubmission policy-learning tables
  rebuttal_p0_experiments.py         direct supervised baselines, corrected significance, cost
  rebuttal_p1_experiments.py         feature-family and similarity-routing baselines
  rebuttal_p2_experiments.py         partial-feedback and conservative-routing diagnostics
  rebuttal_cluster_robust_inference.py  query-cluster-robust inference
  compile_small_model_validation.py  smaller/open-weight validation-slice summaries
prompts/                           NEW: complete strategy template documentation
  strategy_templates.md              verbatim T0-T6 templates, examples, error lists
  prompt_pool_01_03.py               executable reference implementation of the pool
  t5_repeat_diversity_check.csv      audit: T5 repeat-call output diversity at temperature 0
requirements.txt
README.md
```

- **Layer 1 — retrain the main method** (`retraining/`): fits the orthogonalized
  causal forest (EconML `CausalForestDML`, 500 trees/arm, 6 treatment arms vs the
  T0 baseline) from the complete 7-strategy outcome matrix, with query-grouped
  nuisance cross-fitting and PCA refit inside each training split. Includes the
  ten-split stability analysis, fair direct-difference baselines (RandomForest /
  Ridge / kNN / task-only), effect calibration, independently validated gating
  (threshold and fallback selected on an inner validation split only), and
  protocol sensitivity arms (row-level folds, constant propensity, 2000 trees).
- **Layer 2 — recompute paper tables** (`scripts/`): the six v1 scripts, which
  read saved per-query outcomes/assignments and regenerate the resubmission and
  rebuttal tables (see "Reproduce Main Resubmission Tables" below).
- **Layer 3 — strategy templates** (`prompts/`): verbatim templates and example
  content for arms T0–T6.

## (b) Companion data package layout

Download and extract the companion data package next to this software package:

```text
PromptForest_ARR_Aug2026_software_v2/
PromptForest_ARR_Aug2026_data/
  source_data/data/
    processed/01_featured_queries.pkl      query metadata, handcrafted features, saved sem_* PCA-50
    processed/01_feature_columns.txt       feature column list
    results/<model>_n100/02_experiment_results_<model>_n100_scored_v2.csv
                                           full 7-strategy outcome matrix (model in
                                           {gpt-5.5, deepseek-v4-pro, claude-sonnet-4.6})
    results/<model>_n100/03_optimal_assignment_v3_<model>.csv
                                           stored original policy assignments
  paper_results/                           audited manuscript-number CSV assets
```

All retraining scripts take `--data-root` pointing at `source_data/data`; the
default is `../PromptForest_ARR_Aug2026_data/source_data/data` relative to this
package, i.e. no flags are needed with the layout above.

## (c) Running

Python 3.9+; install dependencies with `pip install -r requirements.txt`.

Retrain the core method (full run: 3 models x 10 seeds x 12 causal-forest arm
fits, plus inner gating fits; ~2–4 h with 4–6 workers):

```bash
cd retraining
python aug_rebuttal_experiments.py --out rebuttal_outputs --workers 4
# optional: protocol sensitivity arms (row folds, constant propensity, 2000 trees)
python aug_rebuttal_experiments.py --out rebuttal_outputs --workers 4 --sensitivity
python verify_aug_rebuttal_results.py --out rebuttal_outputs
python report_aug_rebuttal_results.py rebuttal_outputs
```

Smoke test (single model, single seed, 50 trees, a few minutes):

```bash
python aug_rebuttal_experiments.py --out smoke_outputs --models gpt-5.5 --seeds 42 --trees 50 --workers 1
python verify_aug_rebuttal_results.py --out smoke_outputs --partial
```

Semantic embeddings. The outcome/feature inputs above are sufficient to run
everything. For the semantic feature block, the runner needs raw 768-dim query
embeddings in `<out>/inputs/raw_embeddings.npy`:

- **Exact reconstruction (recommended):** `python prepare_aug_embeddings.py
  --out rebuttal_outputs` (requires `sentence-transformers` and downloads
  `intfloat/multilingual-e5-base`; encodes with the `query: ` prefix).
- **Offline fallback (automatic):** if the file is absent, the prepare step
  reuses the saved `sem_*` PCA-50 columns from `01_featured_queries.pkl` and
  says so loudly. This is the `saved_pca` variant: the old PCA was fit before
  the experiment splits, so the per-split "refit" then only rotates those 50
  dimensions. All other protocol corrections (grouped folds, validation-only
  gating) still apply. Use this only when the encoder cannot be downloaded, and
  report it as the saved_pca variant.

Each run writes `protocol.json` (pinned protocol + SHA-256 fingerprints of every
input file), per-query policy and effect predictions, split manifests, and a
`verification.csv` proving split isolation, common denominators, and correct
action-to-outcome lookup.

The v1 table scripts run as before, e.g.:

```bash
python scripts/resubmission_p1_p5_experiments.py --data-root ../PromptForest_ARR_Aug2026_data/source_data/data --out reproduced/resubmission_p1_p5
python scripts/rebuttal_p0_experiments.py --data-root ../PromptForest_ARR_Aug2026_data/source_data/data --out reproduced/rebuttal_p0
python scripts/rebuttal_p1_experiments.py --data-root ../PromptForest_ARR_Aug2026_data/source_data/data --out reproduced/rebuttal_p1
python scripts/rebuttal_p2_experiments.py --data-root ../PromptForest_ARR_Aug2026_data/source_data/data --out-dir reproduced/rebuttal_p2
python scripts/rebuttal_cluster_robust_inference.py --data-root ../PromptForest_ARR_Aug2026_data/source_data/data --out reproduced/rebuttal_cluster_robust
python scripts/compile_small_model_validation.py --input-root ../PromptForest_ARR_Aug2026_data/paper_results/rebuttal_small_model_inputs --out reproduced/rebuttal_small_models
```

Re-running the random-forest router scripts can produce small machine- or
scikit-learn-version-dependent differences (in our package check, maximum
policy-value drift below 0.004). Use `paper_results/` for exact manuscript
values and `reproduced/`/`rebuttal_outputs/` for independently regenerated ones.

## (d) Encoder correction

The manuscript text names **all-MiniLM-L6-v2** as the semantic encoder. That is
incorrect: the saved semantic features were actually produced by
**`intfloat/multilingual-e5-base`** with the `query: ` prefix (768-dimensional
pre-PCA vectors, PCA to 50 dimensions). Re-encoding all 1,100 queries with this
encoder and applying the saved PCA reproduces the stored 50 semantic features
with **R² = 1.0 and maximum absolute error 5.1e-7**. The manuscript and
documentation should be corrected accordingly; the retraining pipeline here uses
the verified original encoder.

## (e) Known protocol differences (short audit summary)

The retraining pipeline deliberately differs from the originally submitted
estimator in two ways, following a pre-experiment protocol audit:

1. **Query-grouped nuisance cross-fitting.** In pairwise long-form training, the
   control and treatment rows of one query previously could enter different
   nuisance folds; the new runner keeps both rows of a query in the same fold
   (a row-level-fold sensitivity arm is included). EconML's internal forest
   honesty/subsampling remains row-based, so native intervals are diagnostic
   scores, not certified query-cluster or post-selection coverage.
2. **PCA refit per training split.** The saved PCA was fit on 660 preprocessing
   queries that overlap the later routing test split (171 shared query IDs with
   the GPT main-test assignment). PCA uses no outcome labels, but the new runs
   refit PCA inside each training split; a saved-PCA variant quantifies the
   difference.

Cohort: GPT has 5 incomplete HotpotQA queries (7,695/7,700 valid outcomes);
DeepSeek and Claude are complete. All compared methods within a model/split
share the same held-out query set (complete-case, split first, then filter).

## (f) Anonymity

This package is anonymous: it contains no author names, institutions, local
absolute paths, API keys, notebooks, checkpoints, token spreadsheets, or
environment files. No network access is required except the one-time optional
encoder download in (c).
