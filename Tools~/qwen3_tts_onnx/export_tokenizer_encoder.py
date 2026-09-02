#!/usr/bin/env python3
"""Export the 12 Hz speech-tokenizer encoder for official Base ICL clone.

I/O for Spark C# QwenTokenizerEncoderModel:
  wav (1, GRAPH_SAMPLES) float32 at 24 kHz, zero-padded
  → audio_codes (1, GRAPH_FRAMES, 16) int64

The Mimi encoder trace freezes T (Python bools in pad/reshape). Dynamo
fails on data-dependent pad. Do not retry dynamo. C# pads/crops to
GRAPH_SAMPLES then keeps the first (original_samples // 1920) frames.
Zero-padding the tail does not change prefix codes (checked vs
tokenizer.encode).

Official Qwen3TTSTokenizer.encode:
  EncodecFeatureExtractor is identity on an unpadded 24 kHz wav.
  encoder.encode outputs (1, 32, T_enc); we keep 16 quantizers and
  transpose to time-major.

Load the tokenizer only — do not load the 1.7B talker.

GRAPH_SAMPLES / SamplesPerFrame must match Spark
QwenTokenizerEncoderModel.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch
import torch.nn as nn

from _onnx_util import consolidate


DEFAULT_TOKENIZER = os.path.expanduser(
    "~/.cache/huggingface/hub/models--Qwen--Qwen3-TTS-12Hz-1.7B-Base/"
    "snapshots/fd4b254389122332181a7c3db7f27e918eec64e3/speech_tokenizer"
)
DEFAULT_OUT = os.path.expanduser("~/Downloads/Qwen3-TTS-ONNX/Qwen3-1.7B-Base")
# 20 s at 24 kHz. Must match QwenTokenizerEncoderModel.GraphSamples.
SAMPLE_RATE = 24000
SAMPLES_PER_FRAME = 1920
GRAPH_SECONDS = 20
GRAPH_SAMPLES = GRAPH_SECONDS * SAMPLE_RATE  # 480000


class TokenizerEncoderWrapper(nn.Module):
    def __init__(self, tokenizer_model):
        super().__init__()
        self.encoder = tokenizer_model.encoder
        self.q = int(tokenizer_model.encoder_valid_num_quantizers)

    def forward(self, wav):
        # wav: (1, T) → encoder wants (1, 1, T)
        encoded = self.encoder.encode(input_values=wav.unsqueeze(1), return_dict=True)
        codes = encoded.audio_codes[:, : self.q]
        return codes.transpose(1, 2).contiguous()


def export_tokenizer_encoder(tokenizer_model, output_dir: str) -> bool:
    os.makedirs(output_dir, exist_ok=True)
    wrapper = TokenizerEncoderWrapper(tokenizer_model)
    wrapper.eval()
    dummy = torch.randn(1, GRAPH_SAMPLES)
    print(f" Tracing at T={GRAPH_SAMPLES} samples ({GRAPH_SECONDS}s, {GRAPH_SAMPLES // SAMPLES_PER_FRAME} frames)")
    onnx_path = os.path.join(output_dir, "tokenizer_encoder.onnx")
    pre_export = set(os.listdir(output_dir))
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (dummy,),
            onnx_path,
            opset_version=17,
            dynamo=False,
            input_names=["wav"],
            output_names=["audio_codes"],
        )
    consolidate(onnx_path, pre_export)
    print(f" Saved: {onnx_path}")

    import onnxruntime as ort

    with torch.no_grad():
        pt = wrapper(dummy).numpy()
    sess = ort.InferenceSession(onnx_path)
    ort_out = sess.run(None, {"wav": dummy.numpy()})[0]
    max_err = float(np.max(np.abs(pt.astype(np.int64) - ort_out.astype(np.int64))))
    print(f" Tokenizer encoder validation: max_err={max_err:.6e}, pt={pt.shape} ort={ort_out.shape}")
    if max_err > 0:
        print(f" WARNING: discrete codes differ (max_err={max_err})")
    return True


def load_tokenizer(tokenizer_dir: str):
    from qwen_tts.inference.qwen3_tts_tokenizer import Qwen3TTSTokenizer

    print(f" Loading 12 Hz tokenizer from {tokenizer_dir}")
    tok = Qwen3TTSTokenizer.from_pretrained(
        tokenizer_dir, dtype=torch.float32, attn_implementation="eager"
    )
    print(
        f" valid_quantizers={tok.model.encoder_valid_num_quantizers} "
        f"downsample={tok.model.encode_downsample_rate} "
        f"sr={tok.feature_extractor.sampling_rate}"
    )
    return tok


def export_from_tts_model(model, output_dir: str) -> bool:
    tok = getattr(model, "speech_tokenizer", None)
    if tok is None:
        print(" speech_tokenizer is None — skip tokenizer encoder")
        return False
    inner = tok.model if hasattr(tok, "model") else tok
    return export_tokenizer_encoder(inner, output_dir)


def main():
    parser = argparse.ArgumentParser(description="Export Qwen3-TTS 12 Hz tokenizer encoder")
    parser.add_argument("--tokenizer-dir", default=DEFAULT_TOKENIZER)
    parser.add_argument("--output-dir", default=DEFAULT_OUT)
    args = parser.parse_args()
    tok = load_tokenizer(args.tokenizer_dir)
    tok.model.eval()
    export_tokenizer_encoder(tok.model, args.output_dir)


if __name__ == "__main__":
    main()
