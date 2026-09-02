#!/usr/bin/env python3
"""Export Qwen3-TTS 1.7B PyTorch to the ElBruno ONNX layout Spark C# loads.

Loads the HuggingFace checkpoint once, then writes:

  talker.onnx[.data]          (both phases; zero-length past = prefill)
  code_predictor.onnx[.data]     # 1024-dim inputs; projection is npy
  vocoder.onnx[.data]
  embeddings/*.npy + nested config.json + empty speaker_ids.json
  tokenizer/vocab.json + merges.txt
  speaker_encoder.onnx[.data]    # Base only
  tokenizer_encoder.onnx[.data]  # Base ICL (12 Hz codec encode)

Output is NOT committed. Defaults are VoiceDesign. Base:

  HF_HUB_DISABLE_XET=1 conda run -n sparktts python tools/qwen3_tts_onnx/export_all.py \\
    --model-id Qwen/Qwen3-TTS-12Hz-1.7B-Base \\
    --output-dir ~/Downloads/Qwen3-TTS-ONNX/Qwen3-1.7B-Base

Move any zukky single-file talker_*.onnx out of the dest first. Those
~5 GB protobufs cannot be converted in place (onnx.load hits the 2 GB
limit). Do not use HuggingFace xet for the safetensors download.

Reference: ElBruno CustomVoice I/O dump, local Spark-TTS export_sparktts_onnx.py
(external-data + opset), wavekat TalkerPrefill/Decode wrappers + mask_patch.
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

THIS = os.path.dirname(os.path.abspath(__file__))
if THIS not in sys.path:
    sys.path.insert(0, THIS)

from export_code_predictor import export_code_predictor
from export_embeddings import export_embeddings
from export_speaker_encoder import export_speaker_encoder
from export_talker import export_talker
from export_tokenizer_encoder import export_from_tts_model
from export_vocoder import export_vocoder
from mask_patch import patch_causal_mask


DEFAULT_MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
DEFAULT_OUT = os.path.expanduser("~/Downloads/Qwen3-TTS-ONNX/Qwen3-1.7B-VoiceDesign")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", default=DEFAULT_OUT)
    parser.add_argument(
        "--skip",
        nargs="*",
        default=[],
        choices=["embeddings", "speaker_encoder", "tokenizer_encoder", "talker", "code_predictor", "vocoder"],
        help="Skip named stages (resume after a partial run)",
    )
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    skip = set(args.skip or [])
    for name in ("talker.onnx", "talker_prefill.onnx", "talker_decode.onnx"):
        path = os.path.join(args.output_dir, name)
        if os.path.isfile(path) and os.path.getsize(path) > 100_000_000:
            raise SystemExit(
                f"{path} is {os.path.getsize(path) / 1e9:.1f} GB (single-file protobuf). "
                "Move the zukky folder aside, then export into an empty dest. "
                "onnx.load cannot convert those files in place."
            )

    print(f"Loading {args.model_id} (fp32 eager) ...")
    patch_causal_mask()
    from qwen_tts.core.models.modeling_qwen3_tts import Qwen3TTSForConditionalGeneration
    model = Qwen3TTSForConditionalGeneration.from_pretrained(
        args.model_id, dtype=torch.float32, attn_implementation="eager"
    )
    model.eval()
    print(f" Loaded. Exporting to {args.output_dir}")

    if "embeddings" not in skip:
        print("\n===== embeddings =====")
        export_embeddings(model, args.output_dir, args.model_id)
    if "speaker_encoder" not in skip:
        print("\n===== speaker_encoder =====")
        export_speaker_encoder(model, args.output_dir)
    if "tokenizer_encoder" not in skip:
        print("\n===== tokenizer_encoder =====")
        export_from_tts_model(model, args.output_dir)
    if "talker" not in skip:
        print("\n===== talker =====")
        export_talker(model, args.output_dir)
    if "code_predictor" not in skip:
        print("\n===== code_predictor =====")
        export_code_predictor(model, args.output_dir)
    if "vocoder" not in skip:
        print("\n===== vocoder =====")
        dynamic = export_vocoder(model, args.output_dir)
        if not dynamic:
            shared = os.path.expanduser("~/Downloads/Qwen3-TTS-ONNX/Qwen3-1.7B/vocoder.onnx")
            if os.path.isfile(shared):
                import shutil
                jit_path = os.path.join(args.output_dir, "vocoder.jit_T100.onnx")
                src = os.path.join(args.output_dir, "vocoder.onnx")
                if os.path.isfile(src) and not os.path.isfile(jit_path):
                    shutil.move(src, jit_path)
                    data = src + ".data"
                    if os.path.isfile(data):
                        shutil.move(data, jit_path + ".data")
                shutil.copy2(shared, src)
                if os.path.isfile(shared + ".data"):
                    shutil.copy2(shared + ".data", src + ".data")
                print(f" Copied shared 12Hz vocoder from {shared} (variable T)")

    print("\nDone. Layout:")
    for name in (
        "talker.onnx",
        "talker.onnx.data",
        "code_predictor.onnx",
        "code_predictor.onnx.data",
        "vocoder.onnx",
        "vocoder.onnx.data",
        "embeddings/config.json",
        "tokenizer/vocab.json",
        "speaker_encoder.onnx",
        "tokenizer_encoder.onnx",
    ):
        path = os.path.join(args.output_dir, name)
        ok = os.path.exists(path)
        print(f"  {'OK' if ok else 'MISSING'} {path}")


if __name__ == "__main__":
    main()
