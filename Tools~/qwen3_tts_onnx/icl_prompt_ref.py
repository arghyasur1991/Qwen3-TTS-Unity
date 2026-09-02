#!/usr/bin/env python3
"""Reference ICL prefill for the Base model, straight off the exported tables.

Rebuilds `Qwen3TTSForConditionalGeneration.generate` + `generate_icl_prompt`
using the same npy tables the Unity engine reads, so the C# prompt can be
compared number for number. Prompt geometry cannot be judged by listening —
a wrong-but-plausible prompt just makes the clone drift.

    conda activate sparktts
    python tools/qwen3_tts_onnx/icl_prompt_ref.py \
        --ref-wav PocketHamletUnity/Assets/Game/Audio/Dialogue/friend_design_ref.wav \
        --ref-text "The scanner sees your ceiling as their sky." \
        --text "I am good. What about you?"

Prints `prefill T / sum / absSum` and `trailing T / sum`, which is exactly what
`LanguageModel.LogPrefillFingerprint` writes to the Unity console.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import onnxruntime as ort
import soundfile as sf
from transformers import AutoTokenizer

DEFAULT_BASE = os.path.expanduser("~/Downloads/Qwen3-TTS-ONNX/Qwen3-1.7B-Base")
GRAPH_SAMPLES = 480000  # must track QwenTokenizerEncoderModel.GraphSamples
SAMPLES_PER_FRAME = 1920


def load_tables(model_dir):
    edir = os.path.join(model_dir, "embeddings")
    t = {}
    for name in (
        "text_embedding",
        "text_projection_fc1_weight", "text_projection_fc1_bias",
        "text_projection_fc2_weight", "text_projection_fc2_bias",
        "talker_codec_embedding",
    ):
        t[name] = np.load(os.path.join(edir, f"{name}.npy"))
    t["cp"] = [np.load(os.path.join(edir, f"cp_codec_embedding_{i}.npy")) for i in range(15)]
    with open(os.path.join(edir, "config.json")) as f:
        t["config"] = json.load(f)
    return t


def text_proj(tables, ids):
    e = tables["text_embedding"][np.asarray(ids)]
    h = e @ tables["text_projection_fc1_weight"].T + tables["text_projection_fc1_bias"]
    h = h * (1.0 / (1.0 + np.exp(-h)))  # SiLU
    return h @ tables["text_projection_fc2_weight"].T + tables["text_projection_fc2_bias"]


def resample(x, src_rate, dst_rate, lobes=16):
    """Port of SparkTTS AudioResample so both sides see identical samples.

    librosa.resample would be close but not equal, and the reference codes are
    sensitive enough that the prompt checksums would drift.
    """
    if src_rate == dst_rate:
        return x
    step = src_rate / dst_rate
    out_len = max(1, int(round(len(x) / step)))
    cutoff = 1.0 / step if step > 1.0 else 1.0
    half = lobes / cutoff
    taps = int(np.ceil(half))

    i = np.arange(out_len)
    centers = i * step
    first = np.floor(centers).astype(np.int64) - taps + 1
    offsets = np.arange(2 * taps)
    idx = first[:, None] + offsets[None, :]
    xdist = centers[:, None] - idx

    inside = (xdist > -half) & (xdist < half)
    t = xdist / half
    a = np.pi * (t + 1.0)
    window = 0.42 - 0.5 * np.cos(a) + 0.08 * np.cos(2.0 * a)
    px = np.pi * cutoff * xdist
    sinc = np.where(np.abs(cutoff * xdist) < 1e-9, 1.0, np.sin(px) / np.where(px == 0, 1, px))
    h = np.where(inside, cutoff * sinc * window, 0.0)

    valid = (idx >= 0) & (idx < len(x)) & inside
    samples = np.where(valid, x[np.clip(idx, 0, len(x) - 1)], 0.0)
    num = (h * samples).sum(axis=1)
    norm = h.sum(axis=1)
    return np.where(np.abs(norm) > 1e-9, num / np.where(norm == 0, 1, norm), 0.0).astype(np.float32)


def load_mono_24k(wav_path):
    audio, sr = sf.read(wav_path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=-1)
    return resample(audio.astype(np.float32), sr, 24000), sr


def encode_reference(model_dir, wav_path):
    """12 Hz codes via the same traced graph the C# uses (fixed 20 s window)."""
    audio, _ = load_mono_24k(wav_path)
    keep = min(len(audio), GRAPH_SAMPLES) // SAMPLES_PER_FRAME
    padded = np.zeros(GRAPH_SAMPLES, dtype=np.float32)
    padded[: min(len(audio), GRAPH_SAMPLES)] = audio[:GRAPH_SAMPLES]

    sess = ort.InferenceSession(os.path.join(model_dir, "tokenizer_encoder.onnx"))
    name = sess.get_inputs()[0].name
    codes = sess.run(None, {name: padded[None, :]})[0]
    codes = np.asarray(codes).reshape(-1, 16)[:keep]
    return codes, len(audio) / 24000.0


def build_prefill(tables, tokenizer, ref_text, text, ref_code, language, spk_embedding,
                  non_streaming_mode=False):
    cfg = tables["config"]
    talker, tts = cfg["talker"], cfg["tts"]
    codec_emb = tables["talker_codec_embedding"]

    input_ids = tokenizer.encode(
        f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n",
        add_special_tokens=False)
    ref_ids = tokenizer.encode(
        f"<|im_start|>assistant\n{ref_text}<|im_end|>\n", add_special_tokens=False)

    text_id = input_ids[3:-5]
    ref_id = ref_ids[3:-2]

    bos, eos, pad = text_proj(tables, [tts["tts_bos_token_id"],
                                       tts["tts_eos_token_id"],
                                       tts["tts_pad_token_id"]])

    lang_map = {k.lower(): v for k, v in cfg["language_ids"].items()}
    lang_id = lang_map.get(language.lower())
    if lang_id is None:
        prefix = [talker["codec_nothink_id"], talker["codec_think_bos_id"],
                  talker["codec_think_eos_id"]]
    else:
        prefix = [talker["codec_think_id"], talker["codec_think_bos_id"],
                  lang_id, talker["codec_think_eos_id"]]

    # codec_input_embedding = [prefix..., speaker, codec_pad, codec_bos]
    codec_input = [codec_emb[i] for i in prefix]
    codec_input.append(spk_embedding)
    codec_input.append(codec_emb[talker["codec_pad_id"]])
    codec_input.append(codec_emb[talker["codec_bos_id"]])
    codec_input = np.stack(codec_input)

    role = text_proj(tables, input_ids[:3])
    # tts_pad * (L-2) + tts_bos, added to codec_input[:-1]
    text_side = np.concatenate(
        [np.tile(pad, (codec_input.shape[0] - 2, 1)), bos[None, :]], axis=0)
    talker_input = text_side + codec_input[:-1]

    embeds = [role, talker_input]

    # --- generate_icl_prompt ---
    icl_text = text_proj(tables, list(ref_id) + list(text_id))
    icl_text = np.concatenate([icl_text, eos[None, :]], axis=0)

    frames = []
    for t in range(ref_code.shape[0]):
        acc = codec_emb[ref_code[t, 0]].copy()
        for g in range(1, 16):
            acc = acc + tables["cp"][g - 1][ref_code[t, g]]
        frames.append(acc)
    codec_side = np.concatenate(
        [codec_emb[talker["codec_bos_id"]][None, :], np.stack(frames)], axis=0)

    t1, t2 = icl_text.shape[0], codec_side.shape[0]
    if non_streaming_mode:
        icl = np.concatenate(
            [icl_text + codec_emb[talker["codec_pad_id"]], codec_side + pad], axis=0)
        trailing = pad[None, :]
    elif t1 > t2:
        icl = icl_text[:t2] + codec_side
        trailing = icl_text[t2:]
    else:
        padded_text = np.concatenate([icl_text, np.tile(pad, (t2 - t1, 1))], axis=0)
        icl = padded_text + codec_side
        trailing = pad[None, :]

    embeds.append(icl)
    prefill = np.concatenate(embeds, axis=0)
    return prefill, trailing, t1, t2


def main():
    np.seterr(all="ignore")  # Accelerate mis-reports matmul flags on macOS
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=DEFAULT_BASE)
    ap.add_argument("--ref-wav", required=True)
    ap.add_argument("--ref-text", required=True)
    ap.add_argument("--text", required=True)
    ap.add_argument("--lang", default="english")
    ap.add_argument("--non-streaming", action="store_true",
                    help="Qwen's generate_voice_clone leaves this off.")
    args = ap.parse_args()

    tables = load_tables(args.model_dir)
    tokenizer = AutoTokenizer.from_pretrained(os.path.join(args.model_dir, "tokenizer"))

    ref_code, seconds = encode_reference(args.model_dir, args.ref_wav)
    print(f"reference {seconds:.2f}s -> ref_code T={ref_code.shape[0]} "
          f"Q={ref_code.shape[1]} codeSum={int(ref_code.sum())}")

    # Speaker embedding via the exported encoder, matching C# ClipToMono24k.
    audio, _ = load_mono_24k(args.ref_wav)
    from librosa.filters import mel as librosa_mel_fn
    n_fft, hop = 1024, 256
    padding = (n_fft - hop) // 2
    y = np.pad(audio, (padding, padding), mode="reflect")
    frames = 1 + (len(y) - n_fft) // hop
    win = np.hanning(n_fft + 1)[:n_fft]
    spec = np.empty((n_fft // 2 + 1, frames), dtype=np.float64)
    for i in range(frames):
        seg = y[i * hop: i * hop + n_fft] * win
        spec[:, i] = np.sqrt(np.abs(np.fft.rfft(seg, n_fft)) ** 2 + 1e-9)
    mel_basis = librosa_mel_fn(sr=24000, n_fft=n_fft, n_mels=128, fmin=0, fmax=12000)
    mels = np.log(np.clip(mel_basis @ spec, 1e-5, None)).T[None, :, :].astype(np.float32)
    se = ort.InferenceSession(os.path.join(args.model_dir, "speaker_encoder.onnx"))
    spk = np.asarray(se.run(None, {se.get_inputs()[0].name: mels})[0]).reshape(-1)
    print(f"speaker embedding dim={spk.shape[0]} xvecSum={spk.sum():.4f}")

    prefill, trailing, t1, t2 = build_prefill(
        tables, tokenizer, args.ref_text, args.text, ref_code, args.lang, spk,
        non_streaming_mode=args.non_streaming)

    print(f"icl text T1={t1} codec T2={t2} "
          f"({'non-streaming concat' if args.non_streaming else 'aligned sum'})")
    print(f"prefill T={prefill.shape[0]} sum={prefill.sum():.3f} "
          f"absSum={np.abs(prefill).sum():.3f} "
          f"trailing T={trailing.shape[0]} sum={trailing.sum():.3f}")


if __name__ == "__main__":
    main()
