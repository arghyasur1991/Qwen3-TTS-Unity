#!/usr/bin/env python3
"""Export the talker once instead of twice, and prove it matches both halves.

`talker_prefill.onnx` and `talker_decode.onnx` are the same 1.7B weights
exported under two different signatures, and both have to be resident for a
single utterance — 8.2 GB of the 11.4 GB an utterance needs. They differ only
in whether a KV cache is passed in:

  prefill: no past          cache_position = arange(0, seq)
  decode:  past of length N cache_position = arange(N, N + seq)

The decode wrapper already derives `past_seq` from the tensor shape and the
patched causal mask derives `kv_len` from `past.get_seq_length() + query_len`,
so decode with a zero-length past *is* prefill. The only reason it cannot do
that today is that its exported `dynamic_axes` leave `inputs_embeds` fixed at
sequence length 1.

This exports one `talker.onnx` with the sequence axis dynamic on both sides,
then runs it in prefill mode against `talker_prefill.onnx` and in decode mode
against `talker_decode.onnx` and reports the difference. Nothing is adopted on
the strength of "it ran".

    conda activate sparktts
    python export_talker_unified.py \
        --model-id Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign \
        --output-dir ~/Downloads/Qwen3-TTS-ONNX/Qwen3-1.7B-VoiceDesign
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _onnx_util import consolidate
from export_talker import TalkerDecodeWrapper
from mask_patch import patch_causal_mask

GRAPH_NAME = "talker"


def export_unified(model, output_dir: str, trace_past: int = 4, trace_seq: int = 3) -> str:
    """
    One graph for both phases.

    Traced with past > 0 and seq > 1 on purpose: tracing at either extreme
    (past=0, or seq=1) invites the exporter to specialise a shape away, and
    then the graph silently only works in the phase it was traced for.
    """
    patch_causal_mask()

    talker = model.talker
    cfg = model.config.talker_config
    num_layers = cfg.num_hidden_layers
    hidden_size = cfg.hidden_size
    num_kv_heads = cfg.num_key_value_heads
    head_dim = cfg.head_dim

    wrapper = TalkerDecodeWrapper(talker.model, talker.codec_head, num_layers)
    wrapper.eval()

    embeds = torch.randn(1, trace_seq, hidden_size)
    mask = torch.ones(1, trace_past + trace_seq, dtype=torch.int64)
    pos = torch.arange(trace_past, trace_past + trace_seq).unsqueeze(0).unsqueeze(0).expand(3, 1, trace_seq)
    past_keys = torch.randn(num_layers, 1, num_kv_heads, trace_past, head_dim)
    past_values = torch.randn(num_layers, 1, num_kv_heads, trace_past, head_dim)

    dynamic = {
        "inputs_embeds": {1: "seq_len"},
        "attention_mask": {1: "total_seq"},
        "position_ids": {2: "seq_len"},
        "past_keys": {3: "past_seq"},
        "past_values": {3: "past_seq"},
        "logits": {1: "seq_len"},
        "hidden_states": {1: "seq_len"},
        "present_keys": {3: "total_seq"},
        "present_values": {3: "total_seq"},
    }

    path = os.path.join(output_dir, GRAPH_NAME + ".onnx")
    pre_export = set(os.listdir(output_dir))
    print(f"Exporting {path} (traced past={trace_past}, seq={trace_seq}) ...")
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (embeds, mask, pos, past_keys, past_values),
            path,
            opset_version=17,
            dynamo=False,
            input_names=["inputs_embeds", "attention_mask", "position_ids",
                         "past_keys", "past_values"],
            output_names=["logits", "hidden_states", "present_keys", "present_values"],
            dynamic_axes=dynamic,
        )
    consolidate(path, pre_export)
    size = os.path.getsize(path) + os.path.getsize(path + ".data")
    print(f" Saved: {path} ({size / 1e9:.2f} GB)")
    return path


def _session(path):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.log_severity_level = 3
    return ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])


def validate(output_dir: str, cfg, seq: int = 7, decode_past: int = 11) -> bool:
    """Unified graph vs the two graphs it is meant to replace."""
    num_layers = cfg.num_hidden_layers
    hidden = cfg.hidden_size
    kv_heads = cfg.num_key_value_heads
    head_dim = cfg.head_dim

    unified_path = os.path.join(output_dir, GRAPH_NAME + ".onnx")
    prefill_path = os.path.join(output_dir, "talker_prefill.onnx")
    decode_path = os.path.join(output_dir, "talker_decode.onnx")

    rng = np.random.default_rng(11)
    ok = True

    print("\n--- prefill mode (past_seq = 0) ---")
    embeds = rng.standard_normal((1, seq, hidden)).astype(np.float32)
    mask = np.ones((1, seq), dtype=np.int64)
    pos = np.tile(np.arange(seq, dtype=np.int64), (3, 1, 1))
    empty_k = np.zeros((num_layers, 1, kv_heads, 0, head_dim), dtype=np.float32)

    uni = _session(unified_path)
    u_out = uni.run(None, {
        "inputs_embeds": embeds, "attention_mask": mask, "position_ids": pos,
        "past_keys": empty_k, "past_values": empty_k,
    })
    u_names = [o.name for o in uni.get_outputs()]
    u = dict(zip(u_names, u_out))

    if os.path.exists(prefill_path):
        pre = _session(prefill_path)
        p_out = pre.run(None, {
            "inputs_embeds": embeds, "attention_mask": mask, "position_ids": pos})
        p = dict(zip([o.name for o in pre.get_outputs()], p_out))
        for key in ("logits", "hidden_states"):
            d = np.abs(u[key] - p[key]).max()
            print(f"  {key:<14} max abs diff {d:.3e}")
            ok &= d < 1e-3
        # prefill emits per-layer KV; the unified graph stacks them.
        worst = 0.0
        for i in range(num_layers):
            worst = max(worst, np.abs(u["present_keys"][i] - p[f"present_key_{i}"]).max())
            worst = max(worst, np.abs(u["present_values"][i] - p[f"present_value_{i}"]).max())
        print(f"  {'KV (stacked)':<14} max abs diff {worst:.3e}")
        ok &= worst < 1e-3
        del pre
    else:
        print("  talker_prefill.onnx not present; skipped")

    print("\n--- decode mode (past_seq > 0, seq = 1) ---")
    step_embeds = rng.standard_normal((1, 1, hidden)).astype(np.float32)
    step_mask = np.ones((1, decode_past + 1), dtype=np.int64)
    step_pos = np.full((3, 1, 1), decode_past, dtype=np.int64)
    past_k = rng.standard_normal((num_layers, 1, kv_heads, decode_past, head_dim)).astype(np.float32)
    past_v = rng.standard_normal((num_layers, 1, kv_heads, decode_past, head_dim)).astype(np.float32)

    feeds = {
        "inputs_embeds": step_embeds, "attention_mask": step_mask,
        "position_ids": step_pos, "past_keys": past_k, "past_values": past_v,
    }
    u2 = dict(zip(u_names, uni.run(None, feeds)))

    if os.path.exists(decode_path):
        dec = _session(decode_path)
        d2 = dict(zip([o.name for o in dec.get_outputs()], dec.run(None, feeds)))
        for key in ("logits", "hidden_states", "present_keys", "present_values"):
            d = np.abs(u2[key] - d2[key]).max()
            print(f"  {key:<14} max abs diff {d:.3e}")
            ok &= d < 1e-3
        del dec
    else:
        print("  talker_decode.onnx not present; skipped")

    print("\n--- multi-token continuation (past > 0, seq > 1) ---")
    # Not something the current pair can do at all: prefill has no past and
    # decode is pinned to one token. Worth exercising because it is what would
    # make batched or chunked prefill possible later.
    cont_seq = 3
    cont = uni.run(None, {
        "inputs_embeds": rng.standard_normal((1, cont_seq, hidden)).astype(np.float32),
        "attention_mask": np.ones((1, decode_past + cont_seq), dtype=np.int64),
        "position_ids": np.tile(
            np.arange(decode_past, decode_past + cont_seq, dtype=np.int64), (3, 1, 1)),
        "past_keys": past_k, "past_values": past_v,
    })
    shapes = {n: np.asarray(v).shape for n, v in zip(u_names, cont)}
    print(f"  ran, logits {shapes['logits']} present_keys {shapes['present_keys']}")
    ok &= shapes["logits"][1] == cont_seq
    ok &= shapes["present_keys"][3] == decode_past + cont_seq

    return bool(ok)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-id", default="Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--validate-only", action="store_true",
                    help="Skip the export and just compare what is on disk.")
    args = ap.parse_args()

    output_dir = os.path.expanduser(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    from qwen_tts.core.models.configuration_qwen3_tts import Qwen3TTSConfig

    if args.validate_only:
        from transformers import AutoConfig
        AutoConfig.register("qwen3_tts", Qwen3TTSConfig, exist_ok=True)
        cfg = AutoConfig.from_pretrained(args.model_id).talker_config
    else:
        print(f"Loading {args.model_id} (fp32 eager) ...")
        from qwen_tts.core.models.modeling_qwen3_tts import Qwen3TTSForConditionalGeneration
        model = Qwen3TTSForConditionalGeneration.from_pretrained(
            args.model_id, dtype=torch.float32, attn_implementation="eager")
        model.eval()
        cfg = model.config.talker_config
        export_unified(model, output_dir)
        del model

    passed = validate(output_dir, cfg)
    print("\n" + ("PASS — one graph reproduces both" if passed else "FAIL — do not adopt"))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
