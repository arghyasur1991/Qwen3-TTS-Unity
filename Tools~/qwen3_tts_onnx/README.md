# Qwen3-TTS → ONNX exporter

Turns a HuggingFace Qwen3-TTS 12 Hz checkpoint into the folder layout the
`Qwen3-TTS-Unity` package reads. Nothing here runs inside Unity.

## Environment

Needs the `qwen_tts` reference package, which is not on PyPI — these scripts
import its model classes directly to trace them.

```bash
conda activate sparktts          # torch, transformers>=4.57, onnx,
                                 # onnxruntime, numpy, soundfile, qwen_tts
```

Set `HF_HUB_DISABLE_XET=1` when downloading weights. Xet hung on
VoiceDesign's 3.6 GB `model.safetensors` and left a corrupt header; plain
`huggingface-cli download` works.

## Exporting a checkpoint

```bash
python export_all.py \
    --model-id Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign \
    --output-dir ~/Downloads/Qwen3-TTS-ONNX/Qwen3-1.7B-VoiceDesign
```

Do the same with `Qwen/Qwen3-TTS-12Hz-1.7B-Base` for the cloning checkpoint.
Point the Cinematic Studio window at the parent folder. Roughly 8 GB and a
few minutes per checkpoint; peak RSS while tracing is around 10 GB, so do not
run two at once.

`--skip <stage>` resumes a partial run. Stages: `embeddings`,
`speaker_encoder`, `tokenizer_encoder`, `talker`, `code_predictor`,
`vocoder`.

## Layout produced

```
talker.onnx[.data]            both phases; a zero-length past is a prefill
code_predictor.onnx[.data]
vocoder.onnx[.data]
speaker_encoder.onnx          Base only (x-vector for cloning)
tokenizer_encoder.onnx        Base only (reference audio → codes)
embeddings/*.npy              tables the C# reads directly, + config.json
tokenizer/                    HF tokenizer files
```

## Files

| | |
|---|---|
| `export_all.py` | Orchestrator. Loads the checkpoint once, runs every stage. |
| `export_talker.py` | The 1.7B talker, one graph for prefill and decode. Self-checks all three shapes against torch and refuses to ship a mismatch. |
| `export_code_predictor.py` | Per-frame residual code predictor (15 groups). |
| `export_vocoder.py` | 12 Hz codec decoder, variable frame count. |
| `export_speaker_encoder.py` | Mel → x-vector. |
| `export_tokenizer_encoder.py` | Reference waveform → codec codes. |
| `export_embeddings.py` | Dumps embedding / projection tables as `.npy`. |
| `mask_patch.py` | Replaces `transformers`' `create_causal_mask`, which uses `torch.vmap` and cannot be traced. Shape-generic, which is what lets one talker graph serve both phases. |
| `_onnx_util.py` | Collapses an export to one `.onnx` + one `.onnx.data`. |

## Reference implementations (debugging, not export)

Independent Python walks of the same graphs, for when C# output is suspect and
you need something with no Unity in it to diff against. Prompt geometry in
particular cannot be judged by listening — a wrong-but-plausible prompt just
makes a clone drift.

```bash
# VoiceDesign: instruct + text → wav
python generate_onnx.py --text "..." --instruct "Male, thirties, warm." \
    --model-dir ~/Downloads/Qwen3-TTS-ONNX/Qwen3-1.7B-VoiceDesign -o /tmp/out.wav

# Base clone: prints the prefill fingerprint LanguageModel logs to the console
python icl_prompt_ref.py --ref-wav <ref>.wav --ref-text "..." --text "..."
```

## Gotchas

- **Do not re-export into a folder you have not cleaned** if you are on an
  older `_onnx_util.py`. `onnx.save_model` appends to an existing
  `.onnx.data` rather than truncating, which silently doubled the talker to
  10.5 GB. Fixed here by removing the destination first, but a stale copy of
  this script elsewhere still has the bug.
- **A talker exported before the graphs were unified** leaves
  `talker_prefill.onnx` and `talker_decode.onnx`. The C# still reads that
  pair, but it costs twice the memory; delete both after re-exporting.
- **Warnings from numpy matmul** (`divide by zero`, `invalid value`) on Apple
  Accelerate are spurious — verified NaN-free in and out. They are silenced in
  the reference scripts.
