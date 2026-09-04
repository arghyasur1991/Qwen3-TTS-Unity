#!/usr/bin/env python3
"""Export VoiceDesign code predictor.

Layout the C# expects (the projection is applied separately, not baked in here):
  inputs_embeds: (B, S, 1024)  — projection is a separate npy, applied in C#
  generation_steps, past_keys, past_values
  → logits (B, S, 2048), present_keys, present_values
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.cache_utils import DynamicCache

from mask_patch import patch_causal_mask
from _onnx_util import consolidate


class CodePredictorWrapper(nn.Module):
    """CP graph without small_to_mtp_projection — C# HasCpProjection path."""

    def __init__(self, code_predictor, num_layers):
        super().__init__()
        self.cp_model = code_predictor.model
        self.num_layers = num_layers
        self.register_buffer(
            "lm_head_weights",
            torch.stack([h.weight for h in code_predictor.lm_head]),
        )

    def forward(self, inputs_embeds, generation_steps, past_keys, past_values):
        cache = DynamicCache()
        for i in range(self.num_layers):
            cache.update(past_keys[i], past_values[i], i)
        past_seq = past_keys.shape[3]
        cache_position = torch.arange(
            past_seq, past_seq + inputs_embeds.shape[1], device=inputs_embeds.device
        )
        outputs = self.cp_model(
            inputs_embeds=inputs_embeds,
            use_cache=True,
            past_key_values=cache,
            cache_position=cache_position,
        )
        hidden_states = outputs.last_hidden_state
        weight = self.lm_head_weights[generation_steps[0]]
        logits = F.linear(hidden_states, weight)
        past_kv = outputs.past_key_values
        keys_list, values_list = [], []
        for i in range(self.num_layers):
            k, v = past_kv[i]
            keys_list.append(k)
            values_list.append(v)
        return logits, torch.stack(keys_list), torch.stack(values_list)


def export_code_predictor(model, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    patch_causal_mask()

    cp = model.talker.code_predictor
    cp_cfg = model.config.talker_config.code_predictor_config
    num_layers = cp_cfg.num_hidden_layers
    cp_hidden = cp_cfg.hidden_size
    num_kv_heads = cp_cfg.num_key_value_heads
    head_dim = cp_cfg.head_dim
    print(
        f" Code Predictor: cp_hidden={cp_hidden}, layers={num_layers}, "
        f"kv_heads={num_kv_heads} (projection stays in npy, not this graph)"
    )

    wrapper = CodePredictorWrapper(cp, num_layers)
    wrapper.eval()
    S = 2
    dummy_embeds = torch.randn(1, S, cp_hidden)
    dummy_gen_steps = torch.tensor([0], dtype=torch.int64)
    dummy_past_keys = torch.zeros(num_layers, 1, num_kv_heads, 0, head_dim)
    dummy_past_values = torch.zeros(num_layers, 1, num_kv_heads, 0, head_dim)
    dynamic_axes = {
        "inputs_embeds": {1: "seq_len"},
        "past_keys": {3: "past_seq"},
        "past_values": {3: "past_seq"},
        "logits": {1: "seq_len"},
        "present_keys": {3: "total_seq"},
        "present_values": {3: "total_seq"},
    }
    onnx_path = os.path.join(output_dir, "code_predictor.onnx")
    print("\nExporting code_predictor.onnx ...")
    pre_export = set(os.listdir(output_dir)) if os.path.exists(output_dir) else set()
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (dummy_embeds, dummy_gen_steps, dummy_past_keys, dummy_past_values),
            onnx_path,
            opset_version=17,
            dynamo=False,
            input_names=["inputs_embeds", "generation_steps", "past_keys", "past_values"],
            output_names=["logits", "present_keys", "present_values"],
            dynamic_axes=dynamic_axes,
        )
    consolidate(onnx_path, pre_export)
    print(f" Saved: {onnx_path}")
    _validate(wrapper, dummy_embeds, dummy_gen_steps, dummy_past_keys, dummy_past_values, onnx_path)
    print(" Validating with decode step (S=1, past_seq=2) ...")
    _validate(
        wrapper,
        torch.randn(1, 1, cp_hidden),
        torch.tensor([1], dtype=torch.int64),
        torch.randn(num_layers, 1, num_kv_heads, 2, head_dim),
        torch.randn(num_layers, 1, num_kv_heads, 2, head_dim),
        onnx_path,
    )
    print("\nCode predictor export complete.")


def _validate(wrapper, embeds, gen_steps, past_keys, past_values, onnx_path):
    import onnxruntime as ort
    with torch.no_grad():
        pt_out = wrapper(embeds, gen_steps, past_keys, past_values)
    sess = ort.InferenceSession(onnx_path)
    ort_out = sess.run(None, {
        "inputs_embeds": embeds.numpy(),
        "generation_steps": gen_steps.numpy(),
        "past_keys": past_keys.numpy(),
        "past_values": past_values.numpy(),
    })
    max_err = np.max(np.abs(pt_out[0].numpy() - ort_out[0]))
    print(f" CP validation: logits max_err={max_err:.6e}, shape={ort_out[0].shape}")
    if max_err > 1e-3:
        print(f" WARNING: max error {max_err:.6e} exceeds 1e-3 threshold")


def main():
    from qwen_tts.core.models.modeling_qwen3_tts import Qwen3TTSForConditionalGeneration
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    patch_causal_mask()
    model = Qwen3TTSForConditionalGeneration.from_pretrained(
        args.model_id, dtype=torch.float32, attn_implementation="eager"
    )
    model.eval()
    export_code_predictor(model, args.output_dir)


if __name__ == "__main__":
    main()
