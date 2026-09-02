#!/usr/bin/env python3
"""Export VoiceDesign talker as talker_prefill.onnx + talker_decode.onnx.

I/O matches ElBruno CustomVoice (and Spark C# LanguageModel):
  prefill: inputs_embeds / attention_mask / position_ids
           → logits, hidden_states, present_key_i, present_value_i
  decode:  + past_keys / past_values (stacked layers)
           → logits, hidden_states, present_keys, present_values
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch
import torch.nn as nn
from transformers.cache_utils import DynamicCache

from mask_patch import patch_causal_mask
from _onnx_util import consolidate


class TalkerPrefillWrapper(nn.Module):
    def __init__(self, talker_model, codec_head, num_layers):
        super().__init__()
        self.talker_model = talker_model
        self.codec_head = codec_head
        self.num_layers = num_layers

    def forward(self, inputs_embeds, attention_mask, position_ids):
        cache_position = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device)
        outputs = self.talker_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=True,
            cache_position=cache_position,
        )
        hidden_states = outputs.last_hidden_state
        logits = self.codec_head(hidden_states)
        past_kv = outputs.past_key_values
        result = [logits, hidden_states]
        for i in range(self.num_layers):
            k, v = past_kv[i]
            result.append(k)
            result.append(v)
        return tuple(result)


class TalkerDecodeWrapper(nn.Module):
    def __init__(self, talker_model, codec_head, num_layers):
        super().__init__()
        self.talker_model = talker_model
        self.codec_head = codec_head
        self.num_layers = num_layers

    def forward(self, inputs_embeds, attention_mask, position_ids, past_keys, past_values):
        cache = DynamicCache()
        for i in range(self.num_layers):
            cache.update(past_keys[i], past_values[i], i)
        past_seq = past_keys.shape[3]
        cache_position = torch.arange(
            past_seq, past_seq + inputs_embeds.shape[1], device=inputs_embeds.device
        )
        outputs = self.talker_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=cache,
            use_cache=True,
            cache_position=cache_position,
        )
        hidden_states = outputs.last_hidden_state
        logits = self.codec_head(hidden_states)
        past_kv = outputs.past_key_values
        keys_list = []
        values_list = []
        for i in range(self.num_layers):
            k, v = past_kv[i]
            keys_list.append(k)
            values_list.append(v)
        return logits, hidden_states, torch.stack(keys_list), torch.stack(values_list)


def export_talker(model, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    patch_causal_mask()

    talker = model.talker
    talker_model = talker.model
    codec_head = talker.codec_head
    cfg = model.config.talker_config
    num_layers = cfg.num_hidden_layers
    hidden_size = cfg.hidden_size
    num_kv_heads = cfg.num_key_value_heads
    head_dim = cfg.head_dim
    print(f" Talker: hidden={hidden_size}, layers={num_layers}, kv_heads={num_kv_heads}, head_dim={head_dim}")

    T = 10
    dummy_embeds = torch.randn(1, T, hidden_size)
    dummy_mask = torch.ones(1, T, dtype=torch.int64)
    dummy_pos = torch.arange(T).unsqueeze(0).unsqueeze(0).expand(3, 1, T)

    print("\nExporting talker_prefill.onnx ...")
    prefill_wrapper = TalkerPrefillWrapper(talker_model, codec_head, num_layers)
    prefill_wrapper.eval()
    prefill_output_names = ["logits", "hidden_states"]
    for i in range(num_layers):
        prefill_output_names.append(f"present_key_{i}")
        prefill_output_names.append(f"present_value_{i}")
    prefill_dynamic = {
        "inputs_embeds": {1: "seq_len"},
        "attention_mask": {1: "seq_len"},
        "position_ids": {2: "seq_len"},
        "logits": {1: "seq_len"},
        "hidden_states": {1: "seq_len"},
    }
    for i in range(num_layers):
        prefill_dynamic[f"present_key_{i}"] = {2: "seq_len"}
        prefill_dynamic[f"present_value_{i}"] = {2: "seq_len"}

    prefill_path = os.path.join(output_dir, "talker_prefill.onnx")
    pre_export = set(os.listdir(output_dir)) if os.path.exists(output_dir) else set()
    with torch.no_grad():
        torch.onnx.export(
            prefill_wrapper,
            (dummy_embeds, dummy_mask, dummy_pos),
            prefill_path,
            opset_version=17,
            dynamo=False,
            input_names=["inputs_embeds", "attention_mask", "position_ids"],
            output_names=prefill_output_names,
            dynamic_axes=prefill_dynamic,
        )
    consolidate(prefill_path, pre_export)
    print(f" Saved: {prefill_path}")
    _validate_prefill(prefill_wrapper, dummy_embeds, dummy_mask, dummy_pos, prefill_path)

    print("\nExporting talker_decode.onnx ...")
    decode_wrapper = TalkerDecodeWrapper(talker_model, codec_head, num_layers)
    decode_wrapper.eval()
    dummy_decode_embeds = torch.randn(1, 1, hidden_size)
    dummy_decode_mask = torch.ones(1, T + 1, dtype=torch.int64)
    dummy_decode_pos = torch.tensor([[[T]]]).expand(3, 1, 1)
    dummy_past_keys = torch.randn(num_layers, 1, num_kv_heads, T, head_dim)
    dummy_past_values = torch.randn(num_layers, 1, num_kv_heads, T, head_dim)
    decode_dynamic = {
        "attention_mask": {1: "total_seq"},
        "past_keys": {3: "past_seq"},
        "past_values": {3: "past_seq"},
        "present_keys": {3: "total_seq"},
        "present_values": {3: "total_seq"},
    }
    decode_path = os.path.join(output_dir, "talker_decode.onnx")
    pre_export = set(os.listdir(output_dir))
    with torch.no_grad():
        torch.onnx.export(
            decode_wrapper,
            (dummy_decode_embeds, dummy_decode_mask, dummy_decode_pos, dummy_past_keys, dummy_past_values),
            decode_path,
            opset_version=17,
            dynamo=False,
            input_names=["inputs_embeds", "attention_mask", "position_ids", "past_keys", "past_values"],
            output_names=["logits", "hidden_states", "present_keys", "present_values"],
            dynamic_axes=decode_dynamic,
        )
    consolidate(decode_path, pre_export)
    print(f" Saved: {decode_path}")
    _validate_decode(
        decode_wrapper, dummy_decode_embeds, dummy_decode_mask, dummy_decode_pos,
        dummy_past_keys, dummy_past_values, decode_path,
    )
    print("\nTalker export complete.")


def _validate_prefill(wrapper, embeds, mask, pos, onnx_path):
    import onnxruntime as ort
    with torch.no_grad():
        pt_out = wrapper(embeds, mask, pos)
    sess = ort.InferenceSession(onnx_path)
    ort_out = sess.run(None, {
        "inputs_embeds": embeds.numpy(),
        "attention_mask": mask.numpy(),
        "position_ids": pos.numpy(),
    })
    max_err = np.max(np.abs(pt_out[0].numpy() - ort_out[0]))
    print(f" Prefill validation: logits max_err={max_err:.6e}, shape={ort_out[0].shape}")
    if max_err > 1e-3:
        print(f" WARNING: max error {max_err:.6e} exceeds 1e-3 threshold")


def _validate_decode(wrapper, embeds, mask, pos, past_keys, past_values, onnx_path):
    import onnxruntime as ort
    with torch.no_grad():
        pt_out = wrapper(embeds, mask, pos, past_keys, past_values)
    sess = ort.InferenceSession(onnx_path)
    ort_out = sess.run(None, {
        "inputs_embeds": embeds.numpy(),
        "attention_mask": mask.numpy(),
        "position_ids": pos.numpy(),
        "past_keys": past_keys.numpy(),
        "past_values": past_values.numpy(),
    })
    max_err = np.max(np.abs(pt_out[0].numpy() - ort_out[0]))
    print(f" Decode validation: logits max_err={max_err:.6e}, shape={ort_out[0].shape}")
    if max_err > 1e-3:
        print(f" WARNING: max error {max_err:.6e} exceeds 1e-3 threshold")


def main():
    from qwen_tts.core.models.modeling_qwen3_tts import Qwen3TTSForConditionalGeneration
    parser = argparse.ArgumentParser(description="Export Qwen3-TTS VoiceDesign talker")
    parser.add_argument("--model-id", default="Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    patch_causal_mask()
    model = Qwen3TTSForConditionalGeneration.from_pretrained(
        args.model_id, dtype=torch.float32, attn_implementation="eager"
    )
    model.eval()
    export_talker(model, args.output_dir)


if __name__ == "__main__":
    main()
