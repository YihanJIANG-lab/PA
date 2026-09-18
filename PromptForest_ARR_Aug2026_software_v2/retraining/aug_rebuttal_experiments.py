"""Retrain the PromptForest main method offline from the saved full outcome matrix
(no LLM API calls): causal forest with query-grouped nuisance cross-fitting,
per-split PCA refit, direct-difference baselines, calibration, and independently
validated gating, across ten 75/25 train/test splits per model.

Inputs are read from --data-root (default: the companion data package extracted
next to this software package, ../PromptForest_ARR_Aug2026_data/source_data/data):

  processed/01_featured_queries.pkl          query metadata + handcrafted features
  processed/01_feature_columns.txt           feature column list
  results/<model>_n100/02_..._scored_v2.csv  full 7-strategy outcome matrix

Semantic embeddings: if <out>/inputs/raw_embeddings.npy does not exist, prepare
first runs a documented fallback that reuses the saved 50-dim PCA features
(saved_pca variant). For fresh 768-dim embeddings, run prepare_aug_embeddings.py
first (requires sentence-transformers and intfloat/multilingual-e5-base).

See protocol.json and README.md in the output directory before interpreting results.
"""
from __future__ import annotations

import os
for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from datetime import datetime
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
# Default: the companion data package extracted next to this software package.
# Override with --data-root.
DATA_ROOT = ROOT.parent / "PromptForest_ARR_Aug2026_data/source_data/data"
MODELS = ["gpt-5.5", "deepseek-v4-pro", "claude-sonnet-4.6"]
SEEDS = list(range(42, 52))
VERSION = 1


def dump_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=True), encoding="utf-8")
    temporary.replace(path)


def fingerprint(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def log(path, message):
    # Windows append offsets are not atomic across worker processes.
    lock_path = Path(path).with_suffix(".lock")
    with lock_path.open("a+b") as lock:
        if os.name == "nt":
            import msvcrt
            lock.seek(0, 2)
            if lock.tell() == 0:
                lock.write(b"0")
                lock.flush()
            while True:
                lock.seek(0)
                try:
                    msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
                    break
                except OSError:
                    time.sleep(.01)
        else:
            import fcntl
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            with Path(path).open("a", encoding="utf-8", buffering=1) as stream:
                stream.write(f"{datetime.now().isoformat(timespec='seconds')} {message}\n")
        finally:
            if os.name == "nt":
                lock.seek(0)
                msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def strict_valid(series):
    normalized = series.astype(str).str.lower()
    if not normalized.isin(["true", "false", "1", "0"]).all():
        raise ValueError("Unrecognized validity flag")
    return normalized.isin(["true", "1"])


def prepare(out, trees=500):
    out.mkdir(parents=True, exist_ok=True)
    cache = out / "inputs"
    cache.mkdir(exist_ok=True)
    feature_path = DATA_ROOT / "processed/01_featured_queries.pkl"
    features = pd.read_pickle(feature_path).sort_values("query_uid").reset_index(drop=True)
    assert features.query_uid.is_unique
    sources = {feature_path.relative_to(DATA_ROOT).as_posix(): fingerprint(feature_path)}
    audits = []
    for model in MODELS:
        path = DATA_ROOT / f"results/{model}_n100/02_experiment_results_{model}_n100_scored_v2.csv"
        sources[path.relative_to(DATA_ROOT).as_posix()] = fingerprint(path)
        frame = pd.read_csv(path)
        assert not frame.duplicated(["query_uid", "assigned_strategy"]).any()
        score = pd.to_numeric(frame.outcome_score_v2, errors="coerce")
        good = strict_valid(frame.is_valid_outcome) & np.isfinite(score)
        valid = frame.loc[good].copy()
        valid["outcome_score_v2"] = score.loc[good]
        pivot = valid.pivot(index="query_uid", columns="assigned_strategy", values="outcome_score_v2")
        pivot = pivot.reindex(index=features.query_uid, columns=range(7))
        complete = pivot.notna().all(axis=1).to_numpy()
        latency = frame.pivot(index="query_uid", columns="assigned_strategy", values="latency")
        latency = latency.reindex(index=features.query_uid, columns=range(7)).to_numpy(float)
        np.savez_compressed(cache / f"{model}.npz", outcomes=pivot.to_numpy(float), complete=complete, latency=latency)
        for ds, group in features.groupby("dataset_name"):
            inds = group.index.to_numpy()
            audits.append(dict(model=model, dataset=ds, total_queries=len(inds), complete_queries=int(complete[inds].sum()), excluded_queries=int((~complete[inds]).sum())))
    features.to_pickle(cache / "features.pkl")
    pd.DataFrame(audits).to_csv(out / "cohort_audit.csv", index=False)
    embedding_path = cache / "raw_embeddings.npy"
    text_hash = hashlib.sha256("\n".join(features.query_uid + "\t" + features.query_text).encode()).hexdigest()
    embed_meta = cache / "embedding_metadata.json"
    if not embedding_path.exists() or not embed_meta.exists():
        # Graceful fallback: reuse the saved 50-dim PCA features as the embedding
        # matrix. PCA(n_components=50, svd_solver="full") on 50-dim inputs is a
        # rotation, so the pipeline runs unchanged, but the semantic features then
        # correspond to the saved_pca variant (old PCA fit, refit per split only
        # as a rotation) rather than fresh_pca from raw 768-dim embeddings.
        # For the exact semantic reconstruction run prepare_aug_embeddings.py
        # (requires sentence-transformers and intfloat/multilingual-e5-base).
        print("WARNING: raw_embeddings.npy not found; falling back to the saved sem_* "
              "PCA-50 features (saved_pca variant). For fresh 768-dim embeddings run "
              "prepare_aug_embeddings.py --out " + str(cache), flush=True)
        saved = features[[f"sem_{k}" for k in range(50)]].to_numpy(float)
        np.save(embedding_path, saved)
        dump_json(embed_meta, dict(model="saved-pca-50 fallback (old PCA features)",
                                   prefix="", normalized=False, shape=list(saved.shape),
                                   saved_pca_fallback=True, text_hash=text_hash))
    embedding_info = json.loads(embed_meta.read_text())
    if embedding_info["text_hash"] != text_hash:
        raise RuntimeError("Embedding provenance mismatch: rerun prepare_aug_embeddings.py")
    sources["raw_embeddings.npy"] = fingerprint(embedding_path)
    protocol = dict(version=VERSION, seeds=SEEDS, models=MODELS, trees=trees, sensitivity_trees=2000,
                    outer_test_fraction=0.25, inner_validation_fraction_of_outer_train=0.20,
                    cv_folds=5, pca_dimensions=50, min_leaf=10, max_depth=20, embedding=embedding_info,
                    cohort="Complete seven-strategy outcomes per model; all policies share the same query denominator. Split all 1100 queries first, then restrict to eligibility.",
                    main_cf="Query-grouped nuisance cross-fitting; learned propensity, as in the original estimator. Orthogonalized CausalForestDML, not a claim of an additional identification requirement.",
                    caveat="Forest internal honesty/subsampling still operates on rows. Native pointwise intervals are diagnostic scores, not certified query-cluster or post-selection coverage.",
                    statistics="Across-split means/SD/win counts are descriptive; overlapping test splits are not independent replicates for significance tests.",
                    gating="Fit on inner train only; choose thresholds and fallback on validation only; do not refit before held-out scoring.",
                    thresholds=[-0.20, -0.10, -0.05, 0.0, 0.01, 0.02, 0.05, 0.10, 0.20, "never"],
                    sources=sources)
    protocol_path = out / "protocol.json"
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
        raise RuntimeError("Protocol/input mismatch: use a new output directory")
    dump_json(protocol_path, protocol)
    print("Prepared input manifest and fixed protocol", flush=True)


def grouped_folds(n, seed):
    # Pairwise rows are [all controls, all treated]; keep both rows in one fold.
    folds = []
    for train, test in KFold(5, shuffle=True, random_state=seed).split(np.arange(n)):
        folds.append((np.r_[train, train + n], np.r_[test, test + n]))
    return folds


def make_features(meta, embeddings, fit_idx, eval_idx, seed):
    columns = (DATA_ROOT / "processed/01_feature_columns.txt").read_text().splitlines()
    handcrafted = [c for c in columns if not c.startswith("sem_")]
    pca = PCA(n_components=50, svd_solver="full")
    sem_train = pca.fit_transform(embeddings[fit_idx])
    sem_eval = pca.transform(embeddings[eval_idx])
    hand = meta[handcrafted].apply(pd.to_numeric, errors="raise").fillna(0).to_numpy(float)
    xtrain = np.c_[sem_train, hand[fit_idx]]
    xeval = np.c_[sem_eval, hand[eval_idx]]
    scaler = StandardScaler().fit(xtrain)
    task_names = sorted(meta.task_type.unique())
    task = np.column_stack([(meta.task_type == name).to_numpy(float) for name in task_names])
    return scaler.transform(xtrain), scaler.transform(xeval), sem_train, sem_eval, task[fit_idx], task[eval_idx]


def fit_cf(x, y, xe, trees, seed, mode, logfile, label):
    from econml.dml import CausalForestDML
    n = len(x)
    xx = np.r_[x, x]
    t = np.r_[np.zeros(n, dtype=int), np.ones(n, dtype=int)]
    cv = grouped_folds(n, seed) if mode != "row_cv" else list(StratifiedKFold(5, shuffle=True, random_state=seed).split(xx, t))
    if mode != "row_cv":
        assert all(not set(a % n).intersection(b % n) for a, b in cv)
    effect, low, high = [], [], []
    for arm in range(1, 7):
        started = time.monotonic()
        log(logfile, f"START {label} arm={arm}/6 trees={trees} n_train={n}")
        propensity = DummyClassifier(strategy="prior") if mode == "known_propensity" else GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=seed)
        est = CausalForestDML(
            model_y=GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=seed),
            model_t=propensity, discrete_treatment=True, cv=cv,
            n_estimators=trees, min_samples_leaf=10, max_depth=20,
            honest=True, criterion="mse", random_state=seed, n_jobs=1,
        )
        est.fit(np.r_[y[:, 0], y[:, arm]], t, X=xx)
        point = est.effect(xe)
        lo, hi = est.effect_interval(xe, alpha=0.05)
        effect.append(point)
        low.append(lo)
        high.append(hi)
        log(logfile, f"DONE {label} arm={arm}/6 elapsed={time.monotonic()-started:.1f}s")
    return np.column_stack(effect), np.column_stack(low), np.column_stack(high)


def rf(seed, trees, flexible=False):
    return RandomForestRegressor(n_estimators=trees, min_samples_leaf=3 if flexible else 10,
                                 max_depth=None if flexible else 20, random_state=seed, n_jobs=1)


def choose(pred):
    return np.argmax(np.c_[np.zeros(len(pred)), pred], axis=1)


def scores(y, actions):
    return y[np.arange(len(y)), actions]


def task_gain_predictions(meta, train, test, gains, column="task_type"):
    table = pd.DataFrame(gains).assign(group=meta.iloc[train][column].to_numpy()).groupby("group").mean()
    return np.vstack([table.loc[group].to_numpy() if group in table.index else gains.mean(axis=0) for group in meta.iloc[test][column]])


def fit_suite(meta, embeddings, train, test, y, trees, seed, logfile, label):
    x, xe, sem, seme, task, taske = make_features(meta, embeddings, train, test, seed)
    gain = y[:, 1:] - y[:, [0]]
    pred, native_lo, native_hi = fit_cf(x, y, xe, trees, seed, "group_cv", logfile, label)
    predictions = {"CF_grouped": pred}
    forest = rf(seed, trees).fit(x, gain)
    predictions["Diff_RF"] = forest.predict(xe)
    predictions["Diff_RF_flexible"] = rf(seed, trees, True).fit(x, gain).predict(xe)
    predictions["Diff_Ridge"] = make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(x, gain).predict(xe)
    predictions["Diff_RF_task_only"] = rf(seed, trees).fit(task, gain).predict(taske)
    # Separate outcome regressors include T0, so all seven actions are available.
    outcome = np.column_stack([rf(seed, trees).fit(x, y[:, arm]).predict(xe) for arm in range(7)])
    predictions["Outcome_RF"] = outcome[:, 1:] - outcome[:, [0]]
    similarity = cosine_similarity(seme, sem)
    neighbors = np.argsort(-similarity, axis=1)[:, :15]
    predictions["Embedding_kNN"] = gain[neighbors].mean(axis=1)
    predictions["Task_mean"] = task_gain_predictions(meta, train, test, gain)
    predictions["Dataset_mean"] = task_gain_predictions(meta, train, test, gain, "dataset_name")
    predictions["Global_mean"] = np.tile(gain.mean(axis=0), (len(test), 1))
    # Tree dispersion is a heuristic score, not a confidence interval for the mean.
    tree_pred = np.stack([tree.predict(xe) for tree in forest.estimators_])
    heuristic_lo = predictions["Diff_RF"] - 1.96 * tree_pred.std(axis=0)
    knn_lo = predictions["Embedding_kNN"] - 1.96 * gain[neighbors].std(axis=1, ddof=1) / np.sqrt(15)
    return predictions, native_lo, native_hi, {"CF_grouped": native_lo, "Diff_RF": heuristic_lo, "Embedding_kNN": knn_lo}


def original_gate(pred, lower):
    best = np.argmax(pred, axis=1)
    return np.where(lower[np.arange(len(pred)), best] > 0, best + 1, 0)


def gate_actions(pred, lower, fallback, threshold):
    candidate = choose(pred)
    extended_pred = np.c_[np.zeros(len(pred)), pred]
    extended_lo = np.c_[np.zeros(len(pred)), lower]
    index = np.arange(len(pred))
    # A routing score relative to the fallback prediction, NOT a valid contrast CI.
    score = extended_lo[index, candidate] - extended_pred[index, fallback]
    route = (candidate != fallback) & (score > threshold)
    return np.where(route, candidate, fallback), route


def select_gate(predictions, lower, validation_y, nval):
    records, choices = [], {}
    for router, lo in lower.items():
        candidates = []
        for fallback_name in ("Task_mean", "Diff_RF"):
            fb = choose(predictions[fallback_name][:nval])
            for threshold in [-.20, -.10, -.05, 0., .01, .02, .05, .10, .20, np.inf]:
                act, mask = gate_actions(predictions[router][:nval], lo[:nval], fb, threshold)
                row = dict(router=router, fallback=fallback_name, threshold=threshold,
                           validation_value=float(scores(validation_y, act).mean()), validation_coverage=float(mask.mean()))
                candidates.append(row)
                records.append(row)
        # Prefer fewer switches, then Task-Best, on exact validation ties.
        chosen = max(candidates, key=lambda r: (r["validation_value"], -r["validation_coverage"], r["fallback"] == "Task_mean", r["threshold"]))
        choices[router] = chosen
    return choices, records


def prediction_frame(meta, idx, y, preds, stage):
    rows = []
    for method, matrix in preds.items():
        for arm in range(1, 7):
            rows.append(pd.DataFrame(dict(query_uid=meta.iloc[idx].query_uid.to_numpy(), dataset=meta.iloc[idx].dataset_name.to_numpy(),
                                          task=meta.iloc[idx].task_type.to_numpy(), method=method, arm=arm,
                                          predicted_gain=matrix[:, arm-1], observed_gain=y[:, arm]-y[:, 0], stage=stage)))
    return pd.concat(rows, ignore_index=True)


def evaluate_policies(meta, idx, y, latency, actions, task_actions, stage):
    frames = []
    task_value = scores(y, task_actions)
    for method, act in actions.items():
        value = scores(y, act)
        frames.append(pd.DataFrame(dict(query_uid=meta.iloc[idx].query_uid.to_numpy(), dataset=meta.iloc[idx].dataset_name.to_numpy(),
                                       task=meta.iloc[idx].task_type.to_numpy(), method=method, action=act, reward=value,
                                       zero_reward=y[:, 0], task_reward=task_value, oracle_reward=y.max(axis=1),
                                       latency_seconds=scores(latency, act), stage=stage)))
    return pd.concat(frames, ignore_index=True)


def run_job(out_string, model, seed, trees=500, sensitivity=False):
    out = Path(out_string)
    name = f"{model}_seed{seed}" + ("_sensitivity" if sensitivity else "")
    job = out / "runs" / name
    job.mkdir(parents=True, exist_ok=True)
    signature = dict(protocol=fingerprint(out / "protocol.json"), script=fingerprint(__file__), trees=trees, sensitivity=sensitivity)
    if (job / "complete.json").exists():
        if json.loads((job / "complete.json").read_text())["signature"] != signature:
            raise RuntimeError(f"Stale job checkpoint: {name}")
        return name
    logfile = out / "progress.log"
    log(logfile, f"JOB_START {name}")
    started = time.monotonic()
    meta = pd.read_pickle(out / "inputs/features.pkl")
    embeddings = np.load(out / "inputs/raw_embeddings.npy")
    data = np.load(out / f"inputs/{model}.npz")
    y, complete, latency = data["outcomes"], data["complete"], data["latency"]
    train_all, test_all = train_test_split(np.arange(len(meta)), test_size=.25, random_state=seed, stratify=meta.dataset_name)
    train, test = train_all[complete[train_all]], test_all[complete[test_all]]
    inner, val = train_test_split(train, test_size=.20, random_state=seed + 10000, stratify=meta.iloc[train].dataset_name)
    assert not set(train).intersection(test)
    assert not set(inner).intersection(val)
    roles = {i: "outer_test" for i in test}
    roles.update({i: "inner_train" for i in inner})
    roles.update({i: "validation" for i in val})
    pd.DataFrame([dict(query_uid=meta.iloc[i].query_uid, role=roles.get(i, "excluded"), eligible=bool(complete[i]), dataset=meta.iloc[i].dataset_name) for i in range(len(meta))]).to_csv(job / "split_manifest.csv", index=False)
    with threadpool_limits(limits=1):
        if sensitivity:
            x, xe, _, _, _, _ = make_features(meta, embeddings, train, test, seed)
            predictions, lower, upper = {}, {}, {}
            for mode, count in [("group_cv", 2000), ("row_cv", 500), ("known_propensity", 500)]:
                label = f"CF_{mode}_{count}"
                predictions[label], lower[label], upper[label] = fit_cf(x, y[train], xe, count, seed, mode, logfile, name + " " + label)
            task_pred = task_gain_predictions(meta, train, test, y[train, 1:]-y[train, [0]][:, None])
            actions = {"Task_mean": choose(task_pred), "Zero": np.zeros(len(test), int)}
            for method, matrix in predictions.items():
                actions[method + "_argmax"] = choose(matrix)
                actions[method + "_gate"] = original_gate(matrix, lower[method])
            policies = evaluate_policies(meta, test, y[test], latency[test], actions, actions["Task_mean"], "sensitivity")
            effects = prediction_frame(meta, test, y[test], predictions, "sensitivity")
        else:
            predictions, lower, upper, _ = fit_suite(meta, embeddings, train, test, y[train], trees, seed, logfile, name + " outer")
            actions = {method: choose(matrix) for method, matrix in predictions.items()}
            actions["CF_original_gate"] = original_gate(predictions["CF_grouped"], lower)
            actions["Zero"] = np.zeros(len(test), int)
            policies = evaluate_policies(meta, test, y[test], latency[test], actions, actions["Task_mean"], "outer")
            effects = prediction_frame(meta, test, y[test], predictions, "outer")
            pd.DataFrame(np.c_[lower, upper], columns=[f"lower_T{k}" for k in range(1,7)]+[f"upper_T{k}" for k in range(1,7)]).assign(query_uid=meta.iloc[test].query_uid.to_numpy()).to_csv(job / "native_intervals.csv", index=False)
            evaluation = np.r_[val, test]
            gp, gl, _, gate_lower = fit_suite(meta, embeddings, inner, evaluation, y[inner], trees, seed, logfile, name + " inner")
            choices, grid = select_gate(gp, gate_lower, y[val], len(val))
            pd.DataFrame(grid).to_csv(job / "validation_grid.csv", index=False)
            selected = []
            gate_actions_dict = {f"{method}_inner": choose(matrix[len(val):]) for method, matrix in gp.items()}
            gate_actions_dict["CF_original_gate_inner"] = original_gate(gp["CF_grouped"][len(val):], gl[len(val):])
            for router, setting in choices.items():
                fb = choose(gp[setting["fallback"]][len(val):])
                act, mask = gate_actions(gp[router][len(val):], gate_lower[router][len(val):], fb, setting["threshold"])
                gate_actions_dict[router + "_validated_gate"] = act
                diff = scores(y[test], act) - scores(y[test], fb)
                selected.append(dict(**setting, test_value=float(scores(y[test], act).mean()), test_fallback_value=float(scores(y[test], fb).mean()),
                                     test_gain=float(diff.mean()), test_coverage=float(mask.mean()), n_routed=int(mask.sum()),
                                     harm_all=float((diff < -1e-12).mean()), harm_routed=float((diff[mask] < -1e-12).mean()) if mask.any() else np.nan,
                                     zero_gain_routed=float((np.abs(diff[mask]) <= 1e-12).mean()) if mask.any() else np.nan,
                                     mean_negative_magnitude=float(np.maximum(-diff,0).mean())))
            pd.DataFrame(selected).to_csv(job / "selected_gates.csv", index=False)
            policies = pd.concat([policies, evaluate_policies(meta, test, y[test], latency[test], gate_actions_dict, choose(gp["Task_mean"][len(val):]), "independent_gate")], ignore_index=True)
        for frame in (policies, effects):
            frame["model"], frame["seed"] = model, seed
        policies.to_csv(job / "policy_predictions.csv", index=False)
        effects.to_csv(job / "effect_predictions.csv", index=False)
    dump_json(job / "complete.json", dict(signature=signature, model=model, seed=seed, trees=trees,
                                         n_train=len(train), n_test=len(test), elapsed_seconds=time.monotonic()-started))
    log(logfile, f"JOB_DONE {name} elapsed={time.monotonic()-started:.1f}s")
    return name


def aggregate(out):
    policy_frames, effect_frames, gate_frames = [], [], []
    for job in sorted((out / "runs").glob("*")):
        if not (job / "complete.json").exists():
            continue
        info = json.loads((job / "complete.json").read_text())
        policy_frames.append(pd.read_csv(job / "policy_predictions.csv"))
        effect_frames.append(pd.read_csv(job / "effect_predictions.csv"))
        if (job / "selected_gates.csv").exists():
            g = pd.read_csv(job / "selected_gates.csv")
            g["model"], g["seed"] = info["model"], info["seed"]
            gate_frames.append(g)
    if not policy_frames:
        return
    policies = pd.concat(policy_frames, ignore_index=True)
    policies["gain_task"] = policies.reward - policies.task_reward
    policies["gain_zero"] = policies.reward - policies.zero_reward
    policies["regret"] = policies.oracle_reward - policies.reward
    policies["harm_task"] = (policies.gain_task < -1e-12).astype(float)
    keys = ["model", "seed", "stage", "method"]
    by_seed = policies.groupby(keys).agg(value=("reward","mean"), gain_task=("gain_task","mean"), gain_zero=("gain_zero","mean"),
                                        regret=("regret","mean"), harm_task=("harm_task","mean"), n=("query_uid","size"), latency=("latency_seconds","mean")).reset_index()
    by_seed.to_csv(out / "policy_by_split.csv", index=False)
    summary = by_seed.groupby(["model", "stage", "method"]).agg(value_mean=("value","mean"), value_sd=("value","std"),
                    gain_task_mean=("gain_task","mean"), gain_task_sd=("gain_task","std"), gain_zero_mean=("gain_zero","mean"),
                    wins_task=("gain_task", lambda x: int((x > 1e-12).sum())), n_splits=("seed","size"), harm_task=("harm_task","mean")).reset_index()
    summary.to_csv(out / "policy_summary.csv", index=False)
    policies.groupby(keys + ["dataset"]).agg(value=("reward","mean"), gain_task=("gain_task","mean"), n=("query_uid","size")).reset_index().to_csv(out / "policy_by_dataset.csv", index=False)
    # Each effect metric uses held-out observations from a single split.
    effects = pd.concat(effect_frames, ignore_index=True)
    metrics, bins = [], []
    for key, group in effects.groupby(keys):
        for arm in ["pooled"] + list(range(1,7)):
            g = group if arm == "pooled" else group[group.arm == arm]
            p, observed = g.predicted_gain.to_numpy(), g.observed_gain.to_numpy()
            variance = np.var(p)
            slope = float(np.mean((p-p.mean())*(observed-observed.mean())) / variance) if variance > 1e-14 else np.nan
            demean = g[["predicted_gain","observed_gain"]] - g.groupby(["task","arm"])[["predicted_gain","observed_gain"]].transform("mean")
            within_corr = float(demean.corr().iloc[0,1])
            metrics.append(dict(zip(keys,key), arm=arm, n=len(g), rmse=float(np.sqrt(np.mean((p-observed)**2))), mae=float(np.mean(np.abs(p-observed))),
                                calibration_slope=slope, calibration_intercept=float(observed.mean()-slope*p.mean()),
                                spearman=float(spearmanr(p,observed).statistic) if np.std(p)>1e-12 and np.std(observed)>1e-12 else np.nan,
                                within_task_arm_correlation=within_corr))
            if arm != "pooled":
                labels = pd.qcut(g.predicted_gain, q=5, duplicates="drop")
                for bin_id, (_, b) in enumerate(g.groupby(labels, observed=True)):
                    bins.append(dict(zip(keys,key), arm=arm, bin=bin_id, n=len(b), predicted_mean=float(b.predicted_gain.mean()), observed_mean=float(b.observed_gain.mean())))
    pd.DataFrame(metrics).to_csv(out / "calibration_by_split.csv", index=False)
    pd.DataFrame(bins).to_csv(out / "calibration_bins.csv", index=False)
    if gate_frames:
        pd.concat(gate_frames, ignore_index=True).to_csv(out / "validated_gates.csv", index=False)
    paired = []
    for (model, seed), group in by_seed[by_seed.stage == "outer"].groupby(["model","seed"]):
        values = group.set_index("method").value
        for method in ["CF_grouped", "CF_original_gate", "Diff_RF", "Diff_RF_flexible", "Embedding_kNN"]:
            for baseline in ["Task_mean", "Dataset_mean", "Diff_RF", "Zero"]:
                paired.append(dict(model=model, seed=seed, method=method, baseline=baseline, difference=float(values[method]-values[baseline])))
    if paired:
        paired = pd.DataFrame(paired)
        paired.to_csv(out / "paired_differences_by_split.csv", index=False)
        paired.groupby(["model","method","baseline"]).difference.agg(["mean","std","min","max",lambda x: int((x>1e-12).sum())]).rename(columns={"<lambda_0>":"wins"}).to_csv(out / "paired_difference_summary.csv")
    print(f"Aggregated {len(policy_frames)} completed jobs", flush=True)


def main():
    global DATA_ROOT
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT,
                        help="Root of the extracted companion data package's source_data/data directory")
    parser.add_argument("--out", type=Path, default=ROOT / "rebuttal_outputs")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--models", nargs="+", default=MODELS)
    parser.add_argument("--trees", type=int, default=500)
    parser.add_argument("--sensitivity", action="store_true")
    args = parser.parse_args()
    DATA_ROOT = args.data_root.resolve()
    out = args.out.resolve()
    if args.aggregate_only:
        aggregate(out)
        return
    # EconML CausalForestDML requires n_estimators divisible by subforest_size (4).
    if args.trees % 4:
        rounded = (args.trees // 4 + 1) * 4
        print(f"NOTE: --trees {args.trees} rounded up to {rounded} (must be divisible by 4)", flush=True)
        args.trees = rounded
    prepare(out, trees=args.trees)
    if args.prepare_only:
        return
    jobs = [(str(out), model, seed, args.trees, False) for seed in args.seeds for model in args.models]
    if args.sensitivity:
        jobs += [(str(out), model, 42, 2000, True) for model in args.models]
    from tqdm import tqdm
    total_arm_fits = sum(18 if job[4] else 12 for job in jobs)
    started = time.monotonic()
    failed = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        pending = {pool.submit(run_job, *job): job for job in jobs}
        done_count = 0
        progress = tqdm(total=len(jobs), desc="ARR August experiments", mininterval=1)
        while pending:
            completed, _ = wait(pending, timeout=15, return_when=FIRST_COMPLETED)
            for future in completed:
                job = pending.pop(future)
                try:
                    future.result()
                except Exception as error:
                    import traceback
                    failure = dict(job=list(job), error=str(error), traceback=traceback.format_exc())
                    failed.append(failure)
                    log(out / "progress.log", "JOB_FAILED " + json.dumps(failure))
                    print(failure["traceback"], flush=True)
                done_count += 1
                progress.update(1)
            status = dict(completed=done_count-len(failed), finished=done_count, failed=len(failed), total=len(jobs),
                          elapsed_seconds=time.monotonic()-started, updated=datetime.now().isoformat(timespec="seconds"))
            progress_text = (out / "progress.log").read_text(encoding="utf-8") if (out / "progress.log").exists() else ""
            status["strategy_fits_done"] = sum(" DONE " in line and "arm=" in line for line in progress_text.splitlines())
            status["strategy_fits_planned"] = total_arm_fits
            dump_json(out / "status.json", status)
            log(out / "progress.log", "TOTAL " + json.dumps(status))
        progress.close()
    dump_json(out / "failures.json", failed)
    aggregate(out)
    if failed:
        raise SystemExit("Some jobs failed; inspect failures.json")


if __name__ == "__main__":
    main()
