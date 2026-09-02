#!/usr/bin/env python3
"""ONNX-only VoiceDesign generate. Prefill matches Spark C# BuildVoiceDesignPrefill.

  python generate_onnx.py --text "The scanner sees your ceiling as their sky." \
      --instruct "Male, thirties, warm conversational friend." \
      --model-dir ~/Downloads/Qwen3-TTS-ONNX/Qwen3-1.7B-VoiceDesign \
      -o /tmp/vd_friend.wav
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import onnxruntime as ort
import soundfile as sf
from transformers import AutoTokenizer


def text_project_numpy(token_ids, text_emb, fc1_w, fc1_b, fc2_w, fc2_b):
    embeds = text_emb[token_ids]
    hidden = embeds @ fc1_w.T + fc1_b
    activated = hidden * (1.0 / (1.0 + np.exp(-hidden)))
    return activated @ fc2_w.T + fc2_b


def load_embeddings(onnx_dir):
    edir = os.path.join(onnx_dir, "embeddings")
    d = {}
    for name in (
        "text_embedding",
        "text_projection_fc1_weight", "text_projection_fc1_bias",
        "text_projection_fc2_weight", "text_projection_fc2_bias",
        "talker_codec_embedding",
    ):
        d[name] = np.load(os.path.join(edir, f"{name}.npy"))
    d["cp_codec_embeddings"] = []
    i = 0
    while True:
        path = os.path.join(edir, f"cp_codec_embedding_{i}.npy")
        if not os.path.exists(path):
            break
        d["cp_codec_embeddings"].append(np.load(path))
        i += 1
    proj_w = os.path.join(edir, "cp_projection_weight.npy")
    proj_b = os.path.join(edir, "cp_projection_bias.npy")
    if os.path.exists(proj_w) and os.path.exists(proj_b):
        d["cp_projection_weight"] = np.load(proj_w)
        d["cp_projection_bias"] = np.load(proj_b)
    return d


def load_config(onnx_dir):
    path = os.path.join(onnx_dir, "embeddings", "config.json")
    with open(path) as f:
        return json.load(f)


def sample_top_k(logits, top_k, temperature):
    if temperature != 1.0:
        logits = logits / temperature
    if top_k > 0 and top_k < len(logits):
        top_k_idx = np.argpartition(logits, -top_k)[-top_k:]
        mask = np.full_like(logits, -np.inf)
        mask[top_k_idx] = logits[top_k_idx]
        logits = mask
    logits = logits - np.max(logits)
    probs = np.exp(logits)
    probs = probs / probs.sum()
    return int(np.random.choice(len(probs), p=probs))


def generate_onnx(model_dir, text, instruct, language, output_path,
                  max_new_tokens, temperature, top_k, repetition_penalty, seed):
    if seed is not None:
        np.random.seed(seed)

    cfg = load_config(model_dir)
    talker = cfg["talker"]
    cp = cfg["code_predictor"]
    tts = cfg["tts"]
    emb = load_embeddings(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(os.path.join(model_dir, "tokenizer"))

    print(f"Loading ONNX from {model_dir} ...")
    prefill_sess = ort.InferenceSession(os.path.join(model_dir, "talker_prefill.onnx"))
    decode_sess = ort.InferenceSession(os.path.join(model_dir, "talker_decode.onnx"))
    cp_sess = ort.InferenceSession(os.path.join(model_dir, "code_predictor.onnx"))
    vocoder_sess = ort.InferenceSession(os.path.join(model_dir, "vocoder.onnx"))

    text_emb = emb["text_embedding"]
    fc1_w, fc1_b = emb["text_projection_fc1_weight"], emb["text_projection_fc1_bias"]
    fc2_w, fc2_b = emb["text_projection_fc2_weight"], emb["text_projection_fc2_bias"]
    codec_emb = emb["talker_codec_embedding"]
    cp_codec_embs = emb["cp_codec_embeddings"]
    has_proj = "cp_projection_weight" in emb
    proj_w = emb.get("cp_projection_weight")
    proj_b = emb.get("cp_projection_bias")

    def text_proj(token_ids):
        return text_project_numpy(np.array(token_ids), text_emb, fc1_w, fc1_b, fc2_w, fc2_b)

    def project(x):
        return x @ proj_w.T + proj_b

    num_layers = talker["num_hidden_layers"]
    hidden_size = talker["hidden_size"]
    num_code_groups = talker["num_code_groups"]
    cp_num_layers = cp["num_hidden_layers"]
    cp_num_kv_heads = cp["num_key_value_heads"]
    cp_head_dim = cp["head_dim"]
    cp_hidden = cp["hidden_size"]
    vocab_size = talker["vocab_size"]
    codec_eos = talker["codec_eos_token_id"]

    chat_text = f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"
    input_ids = tokenizer.encode(chat_text, add_special_tokens=False)
    instruct_tokens = None
    if instruct:
        instruct_text = f"<|im_start|>user\n{instruct}<|im_end|>\n"
        instruct_tokens = tokenizer.encode(instruct_text, add_special_tokens=False)

    print(f" Text: '{text}'")
    if instruct:
        print(f" Instruct: '{instruct}'")
    print(f" Language: {language}")

    lang_map = {k.lower(): v for k, v in cfg["language_ids"].items()}
    language_id = lang_map.get(language.lower())
    if language_id is not None:
        codec_prefix_ids = [
            talker["codec_think_id"], talker["codec_think_bos_id"],
            language_id, talker["codec_think_eos_id"],
        ]
    else:
        codec_prefix_ids = [
            talker["codec_nothink_id"], talker["codec_think_bos_id"], talker["codec_think_eos_id"],
        ]

    tts_pad_embed = text_proj([tts["tts_pad_token_id"]])[0]
    tts_bos_embed = text_proj([tts["tts_bos_token_id"]])[0]
    tts_eos_embed = text_proj([tts["tts_eos_token_id"]])[0]
    codec_pad_embed = codec_emb[talker["codec_pad_id"]]
    codec_bos_embed = codec_emb[talker["codec_bos_id"]]

    embeds_list = []
    if instruct_tokens is not None:
        embeds_list.append(text_proj(instruct_tokens))
    embeds_list.append(text_proj(input_ids[:3]))
    for cid in codec_prefix_ids:
        embeds_list.append((tts_pad_embed + codec_emb[cid]).reshape(1, -1))
    embeds_list.append((tts_bos_embed + codec_pad_embed).reshape(1, -1))
    text_tokens = input_ids[3:-5]
    for tid in text_tokens:
        embeds_list.append((text_proj([tid])[0] + codec_pad_embed).reshape(1, -1))
    embeds_list.append((tts_eos_embed + codec_pad_embed).reshape(1, -1))
    embeds_list.append((tts_pad_embed + codec_bos_embed).reshape(1, -1))

    prefill_embeds = np.concatenate(embeds_list, axis=0)[np.newaxis, :, :].astype(np.float32)
    T = prefill_embeds.shape[1]
    attention_mask = np.ones((1, T), dtype=np.int64)
    position_ids = np.arange(T).reshape(1, 1, T).repeat(3, axis=0)

    print(f" Prefill: {T} tokens")
    prefill_out = prefill_sess.run(None, {
        "inputs_embeds": prefill_embeds,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
    })
    logits = prefill_out[0]
    hidden_states = prefill_out[1]
    kv_outputs = prefill_out[2:]
    past_keys = np.stack([kv_outputs[i * 2] for i in range(num_layers)])
    past_values = np.stack([kv_outputs[i * 2 + 1] for i in range(num_layers)])
    trailing_hidden = tts_pad_embed.reshape(1, -1)

    suppress_mask = np.zeros(vocab_size, dtype=bool)
    suppress_mask[vocab_size - 1024:vocab_size] = True
    suppress_mask[codec_eos] = False

    all_codes = []
    current_pos = T
    generated_tokens = []

    for step in range(max_new_tokens):
        last_logits = logits[0, -1, :].copy()
        last_logits[suppress_mask] = -np.inf
        if step < 2:
            last_logits[codec_eos] = -np.inf
        if repetition_penalty != 1.0 and generated_tokens:
            seen = np.array(generated_tokens)
            scores = last_logits[seen]
            scores = np.where(scores > 0, scores / repetition_penalty, scores * repetition_penalty)
            last_logits[seen] = scores
        group0_token = sample_top_k(last_logits, top_k, temperature)
        if group0_token == codec_eos:
            break
        generated_tokens.append(group0_token)

        frame_codes = [group0_token]
        talker_hidden = hidden_states[0, -1:, :]
        group0_embed = codec_emb[group0_token].reshape(1, -1)
        if has_proj:
            talker_hidden = project(talker_hidden)
            group0_embed = project(group0_embed)
        cp_input = np.concatenate([talker_hidden, group0_embed], axis=0)
        cp_input = cp_input[np.newaxis, :, :].astype(np.float32)
        cp_past_keys = np.zeros((cp_num_layers, 1, cp_num_kv_heads, 0, cp_head_dim), dtype=np.float32)
        cp_past_values = np.zeros((cp_num_layers, 1, cp_num_kv_heads, 0, cp_head_dim), dtype=np.float32)

        for g in range(num_code_groups - 1):
            cp_out = cp_sess.run(None, {
                "inputs_embeds": cp_input,
                "generation_steps": np.array([g], dtype=np.int64),
                "past_keys": cp_past_keys,
                "past_values": cp_past_values,
            })
            token = sample_top_k(cp_out[0][0, -1, :], top_k, temperature)
            frame_codes.append(token)
            cp_past_keys, cp_past_values = cp_out[1], cp_out[2]
            next_cp = cp_codec_embs[g][token].reshape(1, -1)
            if has_proj:
                next_cp = project(next_cp)
            cp_input = next_cp.reshape(1, 1, -1).astype(np.float32)

        all_codes.append(frame_codes)
        next_embed = codec_emb[group0_token].copy()
        for g in range(num_code_groups - 1):
            next_embed = next_embed + cp_codec_embs[g][frame_codes[g + 1]]
        next_embed = next_embed + trailing_hidden[0]
        next_embed = next_embed.reshape(1, 1, -1).astype(np.float32)

        decode_mask = np.ones((1, current_pos + 1), dtype=np.int64)
        decode_pos = np.array([[[current_pos]]]).repeat(3, axis=0)
        decode_out = decode_sess.run(None, {
            "inputs_embeds": next_embed,
            "attention_mask": decode_mask,
            "position_ids": decode_pos,
            "past_keys": past_keys,
            "past_values": past_values,
        })
        logits, hidden_states = decode_out[0], decode_out[1]
        past_keys, past_values = decode_out[2], decode_out[3]
        current_pos += 1
        if (step + 1) % 50 == 0:
            print(f"  ... {step + 1} frames")

    num_frames = len(all_codes)
    print(f" Generated {num_frames} frames")
    if num_frames == 0:
        print(" ERROR: no frames generated")
        return
    codes_arr = np.array(all_codes, dtype=np.int64)
    codes_input = codes_arr.T[np.newaxis, :, :]
    wav = vocoder_sess.run(None, {"codes": codes_input})[0].flatten()
    sf.write(output_path, wav, 24000)
    print(f" Saved: {output_path} ({len(wav) / 24000:.2f}s)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", required=True)
    parser.add_argument("--instruct", default=None)
    parser.add_argument("--lang", default="english")
    parser.add_argument("--model-dir", default=os.path.expanduser(
        "~/Downloads/Qwen3-TTS-ONNX/Qwen3-1.7B-VoiceDesign"))
    parser.add_argument("-o", "--output", default="output.wav")
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--repetition-penalty", type=float, default=1.05)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    generate_onnx(
        args.model_dir, args.text, args.instruct, args.lang, args.output,
        args.max_tokens, args.temperature, args.top_k, args.repetition_penalty, args.seed,
    )


if __name__ == "__main__":
    main()
