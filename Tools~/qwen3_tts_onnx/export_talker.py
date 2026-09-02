#!/usr/bin/env python3
"""Export the talker as one graph that serves both prefill and decode.

    talker.onnx[.data]
      in:  inputs_embeds [1,S,H]  attention_mask [1,P+S]  position_ids [3,1,S]
           past_keys / past_values [L,1,KV,P,D]
      out: logits, hidden_states, present_keys, present_values

A prefill is this graph with `P = 0`. That works because nothing here is
written against a fixed length: `past_seq` comes off `past_keys.shape[3]`,
and the patched causal mask takes `kv_len` from
`past_key_values.get_seq_length() + query_len`.

This used to be exported twice, as `talker_prefill.onnx` and
`talker_decode.onnx` — the same 1.7B weights under two signatures, both of
which had to be resident to say one sentence (8.2 GB of an 11.35 GB set).
The only thing that made two graphs necessary was export metadata: the
decode export never listed `inputs_embeds` in `dynamic_axes`, so tracing
pinned its sequence axis to the dummy's length of 1. Naming the axis is the
whole fix. Verified bit-exact (`0.000e+00` on logits, hidden states and all
28 layers of KV, in both modes, on both checkpoints) against the pair it
replaced; see `docs/Qwen3-TTS-Unity.md` §4.

Usually driven by `export_all.py`. Standalone:

    conda activate sparktts
    python export_talker.py --output-dir ~/Downloads/Qwen3-TTS-ONNX/Qwen3-1.7B-Base \
        --model-id Qwen/Qwen3-TTS-12Hz-1.7B-Base
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch
import torch.nn as nn
from transformers.cache_utils import DynamicCache

from _onnx_util import consolidate
from mask_patch import patch_causal_mask

# Traced at a past and a sequence that are both longer than the degenerate
# case on purpose. Tracing at P=0 or S=1 invites the exporter to fold that
# shape into a constant, and the graph then silently only works in the phase
# it was traced for.
TRACE_PAST = 4
TRACE_SEQ = 3


class TalkerWrapper(nn.Module):
    """Length-agnostic in both the cache and the query."""

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
        keys, values = [], []
        for i in range(self.num_layers):
            k, v = past_kv[i]
            keys.append(k)
            values.append(v)
        return logits, hidden_states, torch.stack(keys), torch.stack(values)


def export_talker(model, output_dir: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    patch_causal_mask()

    talker = model.talker
    cfg = model.config.talker_config
    dims = (cfg.num_hidden_layers, cfg.hidden_size,
            cfg.num_key_value_heads, cfg.head_dim)
    num_layers, hidden_size, num_kv_heads, head_dim = dims
    print(f" Talker: hidden={hidden_size}, layers={num_layers}, "
          f"kv_heads={num_kv_heads}, head_dim={head_dim}")

    wrapper = TalkerWrapper(talker.model, talker.codec_head, num_layers)
    wrapper.eval()

    args = _inputs(TRACE_SEQ, TRACE_PAST, dims, seed=0)
    path = os.path.join(output_dir, "talker.onnx")
    pre_export = set(os.listdir(output_dir))

    print(f"\nExporting talker.onnx (traced past={TRACE_PAST}, seq={TRACE_SEQ}) ...")
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            tuple(torch.from_numpy(a) for a in args),
            path,
            opset_version=17,
            dynamo=False,
            input_names=list(INPUT_NAMES),
            output_names=list(OUTPUT_NAMES),
            dynamic_axes={
                "inputs_embeds": {1: "seq_len"},
                "attention_mask": {1: "total_seq"},
                "position_ids": {2: "seq_len"},
                "past_keys": {3: "past_seq"},
                "past_values": {3: "past_seq"},
                "logits": {1: "seq_len"},
                "hidden_states": {1: "seq_len"},
                "present_keys": {3: "total_seq"},
                "present_values": {3: "total_seq"},
            },
        )
    consolidate(path, pre_export)
    total = os.path.getsize(path) + os.path.getsize(path + ".data")
    print(f" Saved: {path} ({total / 1e9:.2f} GB)")

    _validate(wrapper, path, dims)
    return path


INPUT_NAMES = ("inputs_embeds", "attention_mask", "position_ids",
               "past_keys", "past_values")
OUTPUT_NAMES = ("logits", "hidden_states", "present_keys", "present_values")


def _inputs(seq, past, dims, seed):
    """Numpy inputs for a (past, seq) pair. Same values for torch and ORT."""
    num_layers, hidden_size, num_kv_heads, head_dim = dims
    rng = np.random.default_rng(seed)
    kv_shape = (num_layers, 1, num_kv_heads, past, head_dim)
    return (
        rng.standard_normal((1, seq, hidden_size)).astype(np.float32),
        np.ones((1, past + seq), dtype=np.int64),
        np.tile(np.arange(past, past + seq, dtype=np.int64), (3, 1, 1)),
        rng.standard_normal(kv_shape).astype(np.float32),
        rng.standard_normal(kv_shape).astype(np.float32),
    )


def _validate(wrapper, onnx_path, dims) -> bool:
    """
    Check the graph in both phases against the torch module it came from.

    Both, because they exercise different shapes through the same weights and
    an export can specialise one of them away. A prefill-only check would have
    passed on the old decode graph.
    """
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.log_severity_level = 3
    sess = ort.InferenceSession(onnx_path, so, providers=["CPUExecutionProvider"])

    cases = [
        ("prefill (past=0, seq=7)", 7, 0),
        ("decode  (past=11, seq=1)", 1, 11),
        # Neither of the two graphs this replaces could do this shape at all:
        # prefill had no cache input and decode was pinned to one token. It is
        # what chunked prefill would need, so it is worth keeping honest.
        ("chunk   (past=11, seq=3)", 3, 11),
    ]

    ok = True
    for label, seq, past in cases:
        args = _inputs(seq, past, dims, seed=abs(hash(label)) % 2**31)
        with torch.no_grad():
            expected = wrapper(*[torch.from_numpy(a) for a in args])
        actual = sess.run(list(OUTPUT_NAMES), dict(zip(INPUT_NAMES, args)))

        worst, where = 0.0, ""
        for name, want, got in zip(OUTPUT_NAMES, expected, actual):
            d = float(np.abs(want.numpy() - got).max())
            if d > worst:
                worst, where = d, name
        verdict = "ok" if worst <= 1e-3 else "FAIL"
        print(f" {label}  max abs diff {worst:.3e} ({where})  {verdict}")
        ok &= worst <= 1e-3

    if not ok:
        raise SystemExit("talker.onnx does not match the torch module — do not ship it")
    return ok


def main():
    from qwen_tts.core.models.modeling_qwen3_tts import Qwen3TTSForConditionalGeneration

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default="Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    patch_causal_mask()
    print(f"Loading {args.model_id} (fp32 eager) ...")
    model = Qwen3TTSForConditionalGeneration.from_pretrained(
        args.model_id, dtype=torch.float32, attn_implementation="eager")
    model.eval()
    export_talker(model, os.path.expanduser(args.output_dir))


if __name__ == "__main__":
    main()
