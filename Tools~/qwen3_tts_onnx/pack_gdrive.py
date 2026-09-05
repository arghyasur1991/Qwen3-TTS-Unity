#!/usr/bin/env python3
"""Pack the Qwen3-TTS ONNX files the Unity package actually opens into a zip.

The export folders accumulate superseded graphs (talker_prefill / talker_decode,
a second vocoder, Unity .meta, leftover tokenizer JSON). This copies only the
runtime set into ``QwenTTS/<checkpoint>/`` and stores it (no deflate — the
``.onnx.data`` files are already packed).

    python3 pack_gdrive.py \\
        --src ~/Downloads/Qwen3-TTS-ONNX \\
        --out ~/Downloads/QwenTTS.zip

Default ``--src`` is ``~/Downloads/Qwen3-TTS-ONNX``. Default ``--out`` is
``~/Downloads/QwenTTS.zip``.
"""
from __future__ import annotations

import argparse
import os
import sys
import zipfile
from datetime import datetime, timezone

VOICE = "Qwen3-1.7B-VoiceDesign"
BASE = "Qwen3-1.7B-Base"
CP_GROUPS = 15

README = """Qwen3-TTS ONNX (Unity)

Place this QwenTTS folder where the package can see it, then point
QwenTtsSettings.ModelRoot at it. The engine looks for the two checkpoint
subfolders by these exact names.

    QwenTTS/
      Qwen3-1.7B-VoiceDesign/     design a speaker from a description
      Qwen3-1.7B-Base/            clone from a reference recording

Editor convenience: Assets/StreamingAssets/QwenTTS/ (the package default).
A shipped player should keep the weights outside StreamingAssets so they
are not copied into the build — persistentDataPath, DLC, or a download.

Each checkpoint is ~8 GB fp32. Optional int8 graphs (talker_int8,
code_predictor_int8) are included; set QwenTtsSettings.Precision =
QwenPrecision.Int8 to use them (~1.4× faster, talker ~2.35 GB resident
instead of ~5.67 GB). Missing int8 files fall back to fp32.

Do not also copy talker_prefill / talker_decode: the unified talker.onnx
covers both phases.

Export these yourself with Tools~/qwen3_tts_onnx/ in the Qwen3-TTS-Unity
repository if you would rather not use this zip.
"""


def rels(checkpoint: str) -> list[str]:
    files = [
        "talker.onnx", "talker.onnx.data",
        "talker_int8.onnx", "talker_int8.onnx.data",
        "code_predictor.onnx", "code_predictor.onnx.data",
        "code_predictor_int8.onnx", "code_predictor_int8.onnx.data",
        "vocoder.onnx", "vocoder.onnx.data",
        "embeddings/config.json",
        "embeddings/talker_codec_embedding.npy",
        "embeddings/talker_codec_embedding_proj.npy",
        "embeddings/text_embedding.npy",
        "embeddings/text_projection_fc1_weight.npy",
        "embeddings/text_projection_fc1_bias.npy",
        "embeddings/text_projection_fc2_weight.npy",
        "embeddings/text_projection_fc2_bias.npy",
        "embeddings/cp_projection_weight.npy",
        "embeddings/cp_projection_bias.npy",
        "tokenizer/vocab.json",
        "tokenizer/merges.txt",
    ]
    for i in range(CP_GROUPS):
        files.append("embeddings/cp_codec_embedding_%d.npy" % i)
        files.append("embeddings/cp_codec_embedding_%d_proj.npy" % i)
    if checkpoint == BASE:
        files += [
            "speaker_encoder.onnx", "speaker_encoder.onnx.data",
            "tokenizer_encoder.onnx", "tokenizer_encoder.onnx.data",
        ]
    return files


def collect(src_root: str) -> list[tuple[str, str, int]]:
    """(abspath, zip arcname, size)."""
    rows = []
    missing = []
    for folder in (VOICE, BASE):
        root = os.path.join(src_root, folder)
        for rel in rels(folder):
            path = os.path.join(root, rel)
            arc = os.path.join("QwenTTS", folder, rel).replace("\\", "/")
            if not os.path.isfile(path):
                missing.append(arc)
                continue
            rows.append((path, arc, os.path.getsize(path)))
    if missing:
        print("missing %d files:" % len(missing), file=sys.stderr)
        for m in missing:
            print("  ", m, file=sys.stderr)
        sys.exit(2)
    return rows


def write_zip(out_path: str, rows: list[tuple[str, str, int]]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    if os.path.isfile(out_path):
        os.remove(out_path)
    total = sum(s for _, _, s in rows)
    written = 0
    n = len(rows)
    print("packing %d files, %.2f GB -> %s" % (n, total / 1e9, out_path), flush=True)
    with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as zf:
        zf.writestr("QwenTTS/README.txt", README)
        manifest = ["# QwenTTS ONNX manifest", "# packed %s UTC" % datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"), ""]
        for i, (path, arc, size) in enumerate(rows, 1):
            zf.write(path, arc)
            written += size
            manifest.append("%12d  %s" % (size, arc))
            if i == 1 or i == n or i % 8 == 0:
                print("  %3d/%d  %.2f / %.2f GB  %s" % (
                    i, n, written / 1e9, total / 1e9, os.path.basename(path)), flush=True)
        manifest.append("")
        manifest.append("total_bytes %d" % total)
        zf.writestr("QwenTTS/MANIFEST.txt", "\n".join(manifest) + "\n")
    print("wrote %s (%.2f GB)" % (out_path, os.path.getsize(out_path) / 1e9), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src", default=os.path.expanduser("~/Downloads/Qwen3-TTS-ONNX"),
                   help="folder holding Qwen3-1.7B-VoiceDesign and Qwen3-1.7B-Base")
    p.add_argument("--out", default=os.path.expanduser("~/Downloads/QwenTTS.zip"))
    args = p.parse_args()
    if not os.path.isdir(args.src):
        print("src not found:", args.src, file=sys.stderr)
        sys.exit(2)
    rows = collect(args.src)
    write_zip(args.out, rows)


if __name__ == "__main__":
    main()
