#!/usr/bin/env python3
"""Pack the Qwen3-TTS ONNX files the Unity package actually opens into zips.

A single archive of both checkpoints is ~20 GiB and is a bad Drive upload.
This writes four store-only zips (``.onnx.data`` does not deflate) that
extract into the same ``QwenTTS/<checkpoint>/`` layout:

    QwenTTS-VoiceDesign.zip       fp32 VoiceDesign (~8.3 GB)
    QwenTTS-Base.zip              fp32 Base (~8.7 GB)
    QwenTTS-VoiceDesign-int8.zip  int8 overlay for VoiceDesign (~2.5 GB)
    QwenTTS-Base-int8.zip         int8 overlay for Base (~2.5 GB)

The fp32 zip is the required set. The int8 zip is optional: extract it
on top of the matching fp32 folder and set ``QwenTtsSettings.Precision =
QwenPrecision.Int8``. Missing int8 files fall back to fp32.

    python3 pack_gdrive.py
    python3 pack_gdrive.py --src ~/Downloads/Qwen3-TTS-ONNX --out-dir ~/Downloads
    python3 pack_gdrive.py --only VoiceDesign

Default ``--src`` is ``~/Downloads/Qwen3-TTS-ONNX``. Default ``--out-dir``
is ``~/Downloads``.
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

# (zip filename, checkpoint folder, kind)
ARCHIVES = (
    ("QwenTTS-VoiceDesign.zip", VOICE, "fp32"),
    ("QwenTTS-Base.zip", BASE, "fp32"),
    ("QwenTTS-VoiceDesign-int8.zip", VOICE, "int8"),
    ("QwenTTS-Base-int8.zip", BASE, "int8"),
)

ONLY_ALIASES = {
    "VoiceDesign": "QwenTTS-VoiceDesign.zip",
    "Base": "QwenTTS-Base.zip",
    "VoiceDesign-int8": "QwenTTS-VoiceDesign-int8.zip",
    "Base-int8": "QwenTTS-Base-int8.zip",
}

README_FP32 = """Qwen3-TTS ONNX (Unity) — fp32 checkpoint

Extract this zip so the checkpoint folder sits under QwenTTS/:

    QwenTTS/
      Qwen3-1.7B-VoiceDesign/     this zip, or the Base zip
      Qwen3-1.7B-Base/

Point QwenTtsSettings.ModelRoot at that QwenTTS folder. You only need the
checkpoint you will load: VoiceDesign to invent a speaker, Base to clone
one. Design-then-clone wants both, extracted into the same parent.

Optional int8: extract the matching *-int8.zip on top of this folder, then
set QwenTtsSettings.Precision = QwenPrecision.Int8. Without those files
the engine stays on fp32.

Do not also copy talker_prefill / talker_decode: the unified talker.onnx
covers both phases.

Export these yourself with Tools~/qwen3_tts_onnx/ in the Qwen3-TTS-Unity
repository if you would rather not use this zip.
"""

README_INT8 = """Qwen3-TTS ONNX (Unity) — int8 overlay

Extract this zip into the same parent as the matching fp32 zip so the
int8 graphs land next to talker.onnx inside the checkpoint folder.

    QwenTTS/<checkpoint>/talker_int8.onnx[.data]
    QwenTTS/<checkpoint>/code_predictor_int8.onnx[.data]

Then set QwenTtsSettings.Precision = QwenPrecision.Int8. This overlay is
not a complete checkpoint — install the fp32 zip first. Missing int8
files fall back to fp32.

Int8 is ~1.4× faster; the talker is ~2.35 GB resident instead of ~5.67 GB.
"""


def fp32_rels(checkpoint: str) -> list[str]:
    files = [
        "talker.onnx", "talker.onnx.data",
        "code_predictor.onnx", "code_predictor.onnx.data",
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


def int8_rels() -> list[str]:
    return [
        "talker_int8.onnx", "talker_int8.onnx.data",
        "code_predictor_int8.onnx", "code_predictor_int8.onnx.data",
    ]


def rels(checkpoint: str, kind: str) -> list[str]:
    return int8_rels() if kind == "int8" else fp32_rels(checkpoint)


def collect_one(src_root: str, checkpoint: str, kind: str) -> list[tuple[str, str, int]]:
    """(abspath, zip arcname, size)."""
    rows = []
    missing = []
    root = os.path.join(src_root, checkpoint)
    for rel in rels(checkpoint, kind):
        path = os.path.join(root, rel)
        arc = os.path.join("QwenTTS", checkpoint, rel).replace("\\", "/")
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


def write_zip(out_path: str, kind: str, rows: list[tuple[str, str, int]]) -> None:
    if os.path.isfile(out_path):
        os.remove(out_path)
    total = sum(s for _, _, s in rows)
    written = 0
    n = len(rows)
    print("packing %d files, %.2f GB -> %s" % (n, total / 1e9, out_path), flush=True)
    readme = README_INT8 if kind == "int8" else README_FP32
    with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as zf:
        zf.writestr("QwenTTS/README.txt", readme)
        manifest = [
            "# QwenTTS ONNX manifest",
            "# packed %s UTC" % datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"),
            "# kind %s" % kind,
            "",
        ]
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


def selected_archives(only: str | None) -> tuple[tuple[str, str, str], ...]:
    if not only:
        return ARCHIVES
    name = ONLY_ALIASES.get(only, only)
    picked = tuple(a for a in ARCHIVES if a[0] == name)
    if not picked:
        print("unknown --only %r (want %s)" % (only, ", ".join(ONLY_ALIASES)), file=sys.stderr)
        sys.exit(2)
    return picked


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src", default=os.path.expanduser("~/Downloads/Qwen3-TTS-ONNX"),
                   help="folder holding Qwen3-1.7B-VoiceDesign and Qwen3-1.7B-Base")
    p.add_argument("--out-dir", default=os.path.expanduser("~/Downloads"),
                   help="directory to write the four zips into")
    p.add_argument("--only", default=None,
                   help="one zip: VoiceDesign, Base, VoiceDesign-int8, Base-int8")
    p.add_argument("--dry-run", action="store_true",
                   help="list files and sizes, do not write zips")
    args = p.parse_args()
    if not os.path.isdir(args.src):
        print("src not found:", args.src, file=sys.stderr)
        sys.exit(2)
    os.makedirs(args.out_dir, exist_ok=True)
    archives = selected_archives(args.only)
    grand = 0
    for zip_name, checkpoint, kind in archives:
        rows = collect_one(args.src, checkpoint, kind)
        total = sum(s for _, _, s in rows)
        grand += total
        print("%s  %d files  %.2f GB" % (zip_name, len(rows), total / 1e9), flush=True)
        if args.dry_run:
            continue
        write_zip(os.path.join(args.out_dir, zip_name), kind, rows)
    print("total %.2f GB across %d zip(s)" % (grand / 1e9, len(archives)), flush=True)


if __name__ == "__main__":
    main()
