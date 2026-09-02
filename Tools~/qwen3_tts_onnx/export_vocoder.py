#!/usr/bin/env python3
"""Export VoiceDesign speech-tokenizer decoder as vocoder.onnx.

I/O matches ElBruno / Spark QwenVocoderModel.CustomVoice:
  codes (1, 16, T) int64 → waveform (1, 1, N) float32
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from qwen_tts.core.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2 import EuclideanCodebook
from _onnx_util import consolidate


def precompute_codebook_embeddings(decoder):
    count = 0
    for _, module in decoder.named_modules():
        if isinstance(module, EuclideanCodebook):
            with torch.no_grad():
                embedding = module.embedding_sum / module.cluster_usage.clamp(min=module.epsilon)[:, None]
            module.register_buffer("_precomputed_embedding", embedding)

            def make_fast_decode(mod):
                def fast_decode(codes):
                    return F.embedding(codes, mod._precomputed_embedding)
                return fast_decode

            module.decode = make_fast_decode(module)
            count += 1
    print(f" Precomputed {count} EuclideanCodebook embeddings")


class VocoderWrapper(nn.Module):
    def __init__(self, decoder):
        super().__init__()
        self.decoder = decoder

    def forward(self, codes):
        return self.decoder(codes)


def export_vocoder(model, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    decoder = model.speech_tokenizer.model.decoder
    num_quantizers = decoder.config.num_quantizers
    upsample = decoder.total_upsample
    print(f" Vocoder: num_quantizers={num_quantizers}, upsample={upsample}")
    precompute_codebook_embeddings(decoder)
    decoder.pre_transformer.config.use_cache = False
    wrapper = VocoderWrapper(decoder)
    wrapper.eval()
    T = 100
    dummy_codes = torch.randint(0, 2048, (1, num_quantizers, T), dtype=torch.int64)
    onnx_path = os.path.join(output_dir, "vocoder.onnx")
    print(f"\nExporting vocoder.onnx (dynamic T, traced with T={T}) ...")
    pre_export = set(os.listdir(output_dir)) if os.path.exists(output_dir) else set()
    used_dynamo = False
    with torch.no_grad():
        try:
            num_frames = torch.export.Dim("num_frames", min=2, max=4096)
            torch.onnx.export(
                wrapper,
                (dummy_codes,),
                onnx_path,
                dynamo=True,
                input_names=["codes"],
                output_names=["waveform"],
                dynamic_shapes={"codes": {2: num_frames}},
            )
            used_dynamo = True
        except Exception as e:
            print(f" Dynamo export failed ({e}); falling back to JIT opset 17")
            torch.onnx.export(
                wrapper,
                (dummy_codes,),
                onnx_path,
                opset_version=17,
                dynamo=False,
                input_names=["codes"],
                output_names=["waveform"],
                dynamic_axes={"codes": {2: "num_timesteps"}, "waveform": {2: "num_samples"}},
            )
    print(f" Saved: {onnx_path} ({'dynamo' if used_dynamo else 'jit'})")
    try:
        consolidate(onnx_path, pre_export)
    except Exception as e:
        print(f" Note: consolidation skipped ({e})")
    _validate(wrapper, dummy_codes, onnx_path)
    dynamic_ok = used_dynamo
    if not used_dynamo:
        for test_T in (50, 200, 299):
            test_codes = torch.randint(0, 2048, (1, num_quantizers, test_T), dtype=torch.int64)
            if not _validate(wrapper, test_codes, onnx_path, label=f"T={test_T}", require_match=False):
                dynamic_ok = False
                break
        else:
            dynamic_ok = True
    if not dynamic_ok:
        print(" JIT vocoder is T-static; use a shared 12Hz tokenizer vocoder for variable length.")
    print("\nVocoder export complete.")
    return dynamic_ok


def _validate(wrapper, codes, onnx_path, label=None, require_match=True):
    import onnxruntime as ort
    with torch.no_grad():
        pt_out = wrapper(codes)
    sess = ort.InferenceSession(onnx_path)
    ort_wav = sess.run(None, {"codes": codes.numpy()})[0]
    pt_wav = pt_out.numpy()
    tag = f" ({label})" if label else ""
    if pt_wav.shape != ort_wav.shape:
        print(
            f" Vocoder validation{tag}: shape mismatch pt={pt_wav.shape} ort={ort_wav.shape} "
            "(JIT baked T)"
        )
        return False
    max_err = np.max(np.abs(pt_wav - ort_wav))
    print(
        f" Vocoder validation{tag}: max_err={max_err:.6e}, "
        f"shape={ort_wav.shape}, range=[{ort_wav.min():.3f}, {ort_wav.max():.3f}]"
    )
    if max_err > 1e-3:
        print(f" WARNING: max error {max_err:.6e} exceeds 1e-3 threshold")
        return False
    return True


def main():
    from qwen_tts.core.models.modeling_qwen3_tts import Qwen3TTSForConditionalGeneration
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    model = Qwen3TTSForConditionalGeneration.from_pretrained(
        args.model_id, dtype=torch.float32, attn_implementation="eager"
    )
    model.eval()
    export_vocoder(model, args.output_dir)


if __name__ == "__main__":
    main()
