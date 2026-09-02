#!/usr/bin/env python3
"""Export Base speaker_encoder.onnx.

I/O matches Spark C# QwenSpeakerEncoderModel:
  mels (1, T, 128) float32 → speaker_embedding (1, enc_dim) float32

VoiceDesign has no encoder (tts_model_type != base). Skip there.
"""

from __future__ import annotations

import argparse
import os

import torch
import torch.nn as nn

from _onnx_util import consolidate


class SpeakerEncoderWrapper(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, mels):
        return self.encoder(mels)


def export_speaker_encoder(model, output_dir: str) -> bool:
    encoder = getattr(model, "speaker_encoder", None)
    if encoder is None:
        print(" speaker_encoder is None (not a Base checkpoint) — skip")
        return False

    os.makedirs(output_dir, exist_ok=True)
    enc_dim = int(model.config.speaker_encoder_config.enc_dim)
    mel_dim = int(getattr(model.config.speaker_encoder_config, "mel_dim", 128))
    print(f" Speaker encoder: enc_dim={enc_dim}, mel_dim={mel_dim}")

    wrapper = SpeakerEncoderWrapper(encoder)
    wrapper.eval()
    T = 100
    dummy = torch.randn(1, T, mel_dim)
    onnx_path = os.path.join(output_dir, "speaker_encoder.onnx")
    pre_export = set(os.listdir(output_dir))
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (dummy,),
            onnx_path,
            opset_version=17,
            dynamo=False,
            input_names=["mels"],
            output_names=["speaker_embedding"],
            dynamic_axes={
                "mels": {1: "time"},
            },
        )
    consolidate(onnx_path, pre_export)
    print(f" Saved: {onnx_path}")

    import numpy as np
    import onnxruntime as ort

    with torch.no_grad():
        pt = wrapper(dummy).numpy()
    sess = ort.InferenceSession(onnx_path)
    ort_out = sess.run(None, {"mels": dummy.numpy()})[0]
    max_err = float(np.max(np.abs(pt - ort_out)))
    print(f" Speaker encoder validation: max_err={max_err:.6e}, shape={ort_out.shape}")
    if max_err > 1e-3:
        print(f" WARNING: max error {max_err:.6e} exceeds 1e-3 threshold")
    return True


def main():
    from qwen_tts.core.models.modeling_qwen3_tts import Qwen3TTSForConditionalGeneration
    parser = argparse.ArgumentParser(description="Export Qwen3-TTS Base speaker encoder")
    parser.add_argument("--model-id", default="Qwen/Qwen3-TTS-12Hz-1.7B-Base")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    model = Qwen3TTSForConditionalGeneration.from_pretrained(
        args.model_id, dtype=torch.float32, attn_implementation="eager"
    )
    model.eval()
    export_speaker_encoder(model, args.output_dir)


if __name__ == "__main__":
    main()
