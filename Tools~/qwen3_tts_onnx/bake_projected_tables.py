#!/usr/bin/env python3
"""Pre-apply the code-predictor projection to the codec embedding tables.

At runtime the engine needs `table[token] @ W.T + b` for one talker row and
fourteen code-predictor rows on every output frame. Those are pure functions of
weights fixed at export time, but computing them on demand was the single
largest cost in synthesis — half the wall clock before the matvec was
parallelised, and a few hundred milliseconds after. Baking them turns the whole
thing into a memory read.

Runs against an already-exported folder, so an existing install can get the
tables without reloading the 1.7B checkpoint. `export_embeddings.py` calls the
same function, so a fresh export produces them too.

Adds ~138 MB per checkpoint next to the ~8 GB of graphs. The unprojected
tables stay: the engine still needs them to build the talker's next input, and
an install without the baked files falls back to projecting on demand.

    python bake_projected_tables.py ~/Downloads/Qwen3-TTS-ONNX/Qwen3-1.7B-Base
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

np.seterr(all="ignore")

TALKER_SRC = "talker_codec_embedding.npy"
TALKER_DST = "talker_codec_embedding_proj.npy"


def bake(embed_dir: str, force: bool = False) -> bool:
    """Write the projected tables. False when the folder has no projection."""
    wp = os.path.join(embed_dir, "cp_projection_weight.npy")
    bp = os.path.join(embed_dir, "cp_projection_bias.npy")
    if not (os.path.isfile(wp) and os.path.isfile(bp)):
        print("  no cp_projection_*.npy — this checkpoint projects with Identity, nothing to bake")
        return False

    w = np.load(wp)
    b = np.load(bp)

    def project(table):
        if table.shape[1] != w.shape[1]:
            raise SystemExit(
                f"  table has {table.shape[1]} columns, projection expects {w.shape[1]}")
        # Accumulated in float64: computed once, then trusted for the life of
        # the export.
        return (table.astype(np.float64) @ w.T.astype(np.float64)
                + b.astype(np.float64)).astype(np.float32)

    total = 0
    written = 0

    dst = os.path.join(embed_dir, TALKER_DST)
    if force or not os.path.isfile(dst):
        arr = project(np.load(os.path.join(embed_dir, TALKER_SRC)))
        np.save(dst, arr)
        total += arr.nbytes
        written += 1
        print(f"  {TALKER_DST}: {arr.shape}")

    group = 0
    while True:
        src = os.path.join(embed_dir, f"cp_codec_embedding_{group}.npy")
        if not os.path.isfile(src):
            break
        out = os.path.join(embed_dir, f"cp_codec_embedding_{group}_proj.npy")
        if force or not os.path.isfile(out):
            arr = project(np.load(src))
            np.save(out, arr)
            total += arr.nbytes
            written += 1
        group += 1

    if written == 0:
        print("  already baked (pass --force to redo)")
    else:
        print(f"  {group} code-predictor groups, {total / 1e6:.0f} MB written")
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("checkpoint_dir", nargs="+",
                    help="Export folder, or its embeddings/ subfolder")
    ap.add_argument("--force", action="store_true", help="Rewrite existing tables")
    args = ap.parse_args()

    for raw in args.checkpoint_dir:
        d = os.path.expanduser(raw)
        embed = d if os.path.basename(d) == "embeddings" else os.path.join(d, "embeddings")
        if not os.path.isdir(embed):
            print(f"{raw}: no embeddings/ folder", file=sys.stderr)
            return 1
        print(f"{os.path.basename(os.path.dirname(embed + os.sep))}:")
        bake(embed, args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
