"""Independently verify completed ARR August experiment artifacts."""
from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd

from aug_rebuttal_experiments import ROOT, DATA_ROOT, MODELS, SEEDS, fingerprint
import aug_rebuttal_experiments


def verify(out, partial=False):
    data_root = aug_rebuttal_experiments.DATA_ROOT
    protocol = json.loads((out / "protocol.json").read_text())
    for relative, expected in protocol["sources"].items():
        path = out / "inputs" / relative if relative == "raw_embeddings.npy" else data_root / relative
        assert fingerprint(path) == expected, f"Source changed: {relative}"
    meta = pd.read_pickle(out / "inputs/features.pkl")
    lookup = {uid: i for i, uid in enumerate(meta.query_uid)}
    checked = []
    for model in MODELS:
        source = np.load(out / f"inputs/{model}.npz")
        for seed, suffix in [(s, "") for s in SEEDS] + [(42, "_sensitivity")]:
            job = out / "runs" / f"{model}_seed{seed}{suffix}"
            if partial and not (job / "complete.json").exists():
                continue
            info = json.loads((job / "complete.json").read_text())
            assert info["signature"]["protocol"] == fingerprint(out / "protocol.json")
            assert info["signature"]["script"] == fingerprint(Path(aug_rebuttal_experiments.__file__))
            split = pd.read_csv(job / "split_manifest.csv")
            assert split.query_uid.is_unique and len(split) == 1100
            test = set(split.loc[split.role == "outer_test", "query_uid"])
            train = set(split.loc[split.role.isin(["inner_train", "validation"]), "query_uid"])
            assert not train & test
            assert len(test) == info["n_test"]
            assert len(train) == info["n_train"]
            policy = pd.read_csv(job / "policy_predictions.csv")
            assert not policy.duplicated(["stage", "method", "query_uid"]).any()
            for _, block in policy.groupby(["stage", "method"]):
                assert set(block.query_uid) == test
            ids = policy.query_uid.map(lookup).to_numpy()
            assert source["complete"][ids].all()
            np.testing.assert_allclose(policy.reward, source["outcomes"][ids, policy.action.to_numpy(int)], atol=1e-12)
            np.testing.assert_allclose(policy.oracle_reward, source["outcomes"][ids].max(axis=1), atol=1e-12)
            effects = pd.read_csv(job / "effect_predictions.csv")
            assert not effects.duplicated(["stage", "method", "arm", "query_uid"]).any()
            assert np.isfinite(effects[["predicted_gain", "observed_gain"]].to_numpy()).all()
            ei = effects.query_uid.map(lookup).to_numpy()
            actual = source["outcomes"][ei, effects.arm.to_numpy(int)] - source["outcomes"][ei, 0]
            np.testing.assert_allclose(effects.observed_gain, actual, atol=1e-12)
            if not suffix:
                grid = pd.read_csv(job / "validation_grid.csv")
                gates = pd.read_csv(job / "selected_gates.csv")
                assert len(grid) == 60 and len(gates) == 3
                for row in gates.itertuples():
                    candidates = grid[grid.router == row.router].to_dict("records")
                    best = max(candidates, key=lambda r: (r["validation_value"], -r["validation_coverage"], r["fallback"] == "Task_mean", r["threshold"]))
                    assert row.threshold == best["threshold"] and row.fallback == best["fallback"]
                    p = policy[(policy.stage == "independent_gate") & (policy.method == row.router + "_validated_gate")]
                    np.testing.assert_allclose(p.reward.mean(), row.test_value, atol=1e-12)
                interval = pd.read_csv(job / "native_intervals.csv")
                for arm in range(1, 7):
                    assert (interval[f"lower_T{arm}"] <= interval[f"upper_T{arm}"]).all()
            checked.append(dict(job=job.name, n_test=len(test), policies=len(policy), effects=len(effects), verified=True))
    if not partial:
        assert json.loads((out / "failures.json").read_text()) == []
        assert len(checked) == 33
    pd.DataFrame(checked).to_csv(out / ("verification_partial.csv" if partial else "verification.csv"), index=False)
    print(f"PASS: {len(checked)} jobs; split isolation, common denominators, action lookup, effect targets, validation-only gate selection, and provenance.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT,
                        help="Must match the --data-root used for the experiment run")
    parser.add_argument("--out", type=Path, default=ROOT / "rebuttal_outputs")
    parser.add_argument("--partial", action="store_true")
    args = parser.parse_args()
    aug_rebuttal_experiments.DATA_ROOT = args.data_root.resolve()
    verify(args.out, args.partial)
