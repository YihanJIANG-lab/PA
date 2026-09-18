"""Rebuild the raw 768-dim semantic embeddings for the retraining pipeline.

Encodes the query texts from <data-root>/processed/01_featured_queries.pkl with
intfloat/multilingual-e5-base (sentence-transformers, "query: " prefix, no
normalization) and writes raw_embeddings.npy + embedding_metadata.json into
<out>/inputs, where <out> is the same output directory passed to
aug_rebuttal_experiments.py.

Encoder correction: the manuscript text names all-MiniLM-L6-v2, but the saved
semantic features were actually produced by intfloat/multilingual-e5-base with
the "query: " prefix (768-dim pre-PCA vectors, PCA to 50 dims). Re-encoding all
1,100 queries with this encoder and applying the saved PCA reproduces the stored
50 semantic features with R^2 = 1.0 and maximum absolute error 5.1e-7.

If the model cannot be downloaded, aug_rebuttal_experiments.py falls back to the
saved sem_* PCA-50 columns automatically (the saved_pca variant); this script is
then not required.

Usage:
    python prepare_aug_embeddings.py --out rebuttal_outputs \
        [--data-root ../PromptForest_ARR_Aug2026_data/source_data/data]
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = PACKAGE_ROOT.parent / "PromptForest_ARR_Aug2026_data/source_data/data"
MODEL_NAME = "intfloat/multilingual-e5-base"
PREFIX = "query: "


def dump_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=True), encoding="utf-8")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--out", type=Path, required=True,
                        help="Same --out directory as aug_rebuttal_experiments.py")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    from sentence_transformers import SentenceTransformer
    import torch

    torch.set_num_threads(4)
    features = pd.read_pickle(args.data_root / "processed/01_featured_queries.pkl")
    features = features.sort_values("query_uid").reset_index(drop=True)
    assert features.query_uid.is_unique

    model = SentenceTransformer(MODEL_NAME, device="cuda" if torch.cuda.is_available() else "cpu")
    vectors = model.encode([PREFIX + t for t in features.query_text], batch_size=args.batch_size,
                           show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=False)
    assert vectors.shape == (len(features), 768)

    cache = args.out / "inputs"
    cache.mkdir(parents=True, exist_ok=True)
    np.save(cache / "raw_embeddings.npy", vectors)
    metadata = dict(model=MODEL_NAME, prefix=PREFIX, normalized=False, shape=list(vectors.shape),
                    text_hash=hashlib.sha256("\n".join(features.query_uid + "\t" + features.query_text).encode()).hexdigest(),
                    note="Fresh re-encoding with the verified original encoder; replaces the saved-pca fallback.")
    dump_json(cache / "embedding_metadata.json", metadata)
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
