#!/usr/bin/env python3
"""Dump VoiceDesign embedding tables + ElBruno-shaped nested config.json.

Spark C# EmbeddingStore reads nested talker / code_predictor / tts / language_ids,
plus cp_projection_*.npy (small_to_mtp) and an empty speaker_ids.json.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch


def export_embeddings(model, output_dir: str, model_id: str) -> None:
    talker = model.talker
    config = model.config
    talker_config = config.talker_config
    cp_config = talker_config.code_predictor_config

    embed_dir = os.path.join(output_dir, "embeddings")
    tok_dir = os.path.join(output_dir, "tokenizer")
    os.makedirs(embed_dir, exist_ok=True)
    os.makedirs(tok_dir, exist_ok=True)

    w = talker.model.text_embedding.weight.detach().cpu().numpy()
    np.save(os.path.join(embed_dir, "text_embedding.npy"), w)
    print(f" text_embedding: {w.shape}")

    named = dict(talker.text_projection.named_parameters())
    for name, param_name in [
        ("text_projection_fc1_weight", "linear_fc1.weight"),
        ("text_projection_fc1_bias", "linear_fc1.bias"),
        ("text_projection_fc2_weight", "linear_fc2.weight"),
        ("text_projection_fc2_bias", "linear_fc2.bias"),
    ]:
        arr = named[param_name].detach().cpu().numpy()
        np.save(os.path.join(embed_dir, f"{name}.npy"), arr)
        print(f" {name}: {arr.shape}")

    w = talker.model.codec_embedding.weight.detach().cpu().numpy()
    np.save(os.path.join(embed_dir, "talker_codec_embedding.npy"), w)
    print(f" talker_codec_embedding: {w.shape}")
    talker_hidden = int(talker_config.hidden_size)
    cp_embeddings = talker.code_predictor.model.codec_embedding
    num_cp_groups = cp_config.num_code_groups - 1
    for i in range(num_cp_groups):
        arr = cp_embeddings[i].weight.detach().cpu().numpy()
        # ElBruno / Spark C# project 2048→1024. A 1024-col table would
        # OOB in ProjectRow (weight cols == talker hidden). Pad if needed.
        if arr.shape[1] < talker_hidden:
            padded = np.zeros((arr.shape[0], talker_hidden), dtype=arr.dtype)
            padded[:, : arr.shape[1]] = arr
            arr = padded
        np.save(os.path.join(embed_dir, f"cp_codec_embedding_{i}.npy"), arr)
        print(f" cp_codec_embedding_{i}: {arr.shape}")

    proj = talker.code_predictor.small_to_mtp_projection
    if isinstance(proj, torch.nn.Linear):
        np.save(os.path.join(embed_dir, "cp_projection_weight.npy"), proj.weight.detach().cpu().numpy())
        np.save(os.path.join(embed_dir, "cp_projection_bias.npy"), proj.bias.detach().cpu().numpy())
        print(f" cp_projection_weight: {tuple(proj.weight.shape)}")
        print(f" cp_projection_bias: {tuple(proj.bias.shape)}")
    else:
        print(" small_to_mtp_projection: Identity (no cp_projection npy)")

    head_w = talker.codec_head.weight.detach().cpu().numpy()
    np.save(os.path.join(embed_dir, "codec_head_weight.npy"), head_w)
    print(f" codec_head_weight: {head_w.shape}")

    language_ids = {}
    raw_lang = talker_config.codec_language_id or {}
    for k, v in raw_lang.items():
        language_ids[str(k).lower()] = int(v)

    nested = {
        "talker": {
            "hidden_size": talker_config.hidden_size,
            "text_hidden_size": talker_config.text_hidden_size,
            "vocab_size": talker_config.vocab_size,
            "num_hidden_layers": talker_config.num_hidden_layers,
            "num_attention_heads": talker_config.num_attention_heads,
            "num_key_value_heads": talker_config.num_key_value_heads,
            "head_dim": talker_config.head_dim,
            "num_code_groups": talker_config.num_code_groups,
            "codec_eos_token_id": talker_config.codec_eos_token_id,
            "codec_think_id": talker_config.codec_think_id,
            "codec_nothink_id": talker_config.codec_nothink_id,
            "codec_think_bos_id": talker_config.codec_think_bos_id,
            "codec_think_eos_id": talker_config.codec_think_eos_id,
            "codec_pad_id": talker_config.codec_pad_id,
            "codec_bos_id": talker_config.codec_bos_id,
            "rope_theta": getattr(talker_config, "rope_theta", 1000000),
        },
        "code_predictor": {
            "hidden_size": cp_config.hidden_size,
            "vocab_size": cp_config.vocab_size,
            "num_hidden_layers": cp_config.num_hidden_layers,
            "num_attention_heads": cp_config.num_attention_heads,
            "num_key_value_heads": cp_config.num_key_value_heads,
            "head_dim": cp_config.head_dim,
            "rope_theta": getattr(cp_config, "rope_theta", 1000000),
        },
        "tts": {
            "tts_bos_token_id": config.tts_bos_token_id,
            "tts_eos_token_id": config.tts_eos_token_id,
            "tts_pad_token_id": config.tts_pad_token_id,
            "im_start_token_id": config.im_start_token_id,
            "im_end_token_id": config.im_end_token_id,
        },
        "language_ids": language_ids,
        "speaker_dialect": {},
        "tts_model_type": getattr(config, "tts_model_type", "voice_design"),
        "model_id": model_id,
    }
    with open(os.path.join(embed_dir, "config.json"), "w") as f:
        json.dump(nested, f, indent=2, ensure_ascii=False)
    print(" embeddings/config.json written (nested ElBruno shape)")

    with open(os.path.join(embed_dir, "speaker_ids.json"), "w") as f:
        json.dump({}, f)
    print(" embeddings/speaker_ids.json written (empty — VoiceDesign has no presets)")

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.save_pretrained(tok_dir)
    print(f" tokenizer saved to {tok_dir}")


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
    export_embeddings(model, args.output_dir, args.model_id)


if __name__ == "__main__":
    main()
