#!/usr/bin/env bash
# Link Qwen3-TTS 1.7B ONNX for Spark (VoiceDesign style + Base clone).
# Both layouts come from tools/qwen3_tts_onnx/export_all.py into
# ~/Downloads/Qwen3-TTS-ONNX/. This script does not copy ONNX into git.
# MysteryAI SparkTTS/Qwen3-1.7B points at VoiceDesign when that export
# exists, else leftover CustomVoice. Base is our ElBruno export, not zukky.
set -euo pipefail

DEST="${QWEN3_TTS_DEST:-$HOME/Downloads/Qwen3-TTS-ONNX}"
MYSTERY_SPARK="${MYSTERY_SPARK:-$HOME/Personal/Projects/ML/MysteryAI/Assets/StreamingAssets/SparkTTS}"

VD_DIR="$DEST/Qwen3-1.7B-VoiceDesign"
CV_DIR="$DEST/Qwen3-1.7B"
BASE_DIR="$DEST/Qwen3-1.7B-Base"

base_elbruno() {
  [[ -f "$1/talker_prefill.onnx.data" && -f "$1/talker_decode.onnx.data" \
     && -f "$1/embeddings/config.json" && -f "$1/speaker_encoder.onnx" \
     && -f "$1/tokenizer/vocab.json" && -f "$1/tokenizer_encoder.onnx" ]]
}

if [[ -f "$VD_DIR/talker_prefill.onnx.data" && -f "$VD_DIR/talker_decode.onnx.data" ]]; then
  STYLE_DIR="$VD_DIR"
  echo "[qwen3-tts] VoiceDesign export present → $STYLE_DIR"
elif [[ -f "$CV_DIR/talker_prefill.onnx.data" && -f "$CV_DIR/talker_decode.onnx.data" ]]; then
  STYLE_DIR="$CV_DIR"
  echo "[qwen3-tts] VoiceDesign missing; using CustomVoice at $STYLE_DIR"
else
  echo "[qwen3-tts] No style ONNX. Export VoiceDesign:" >&2
  echo "  HF_HUB_DISABLE_XET=1 conda run -n sparktts python tools/qwen3_tts_onnx/export_all.py" >&2
  exit 1
fi

if base_elbruno "$BASE_DIR"; then
  echo "[qwen3-tts] Base ElBruno export present → $BASE_DIR"
else
  echo "[qwen3-tts] Base ElBruno layout missing at $BASE_DIR" >&2
  echo "  Move zukky single-file talkers aside, then:" >&2
  echo "  HF_HUB_DISABLE_XET=1 conda run -n sparktts python tools/qwen3_tts_onnx/export_all.py \\" >&2
  echo "    --model-id Qwen/Qwen3-TTS-12Hz-1.7B-Base \\" >&2
  echo "    --output-dir $BASE_DIR" >&2
  echo "[qwen3-tts] Linking style only (clone Preview will stay gated)" >&2
fi

if [[ -d "$MYSTERY_SPARK" ]]; then
  ln -sfn "$STYLE_DIR" "$MYSTERY_SPARK/Qwen3-1.7B"
  echo "[qwen3-tts] Linked $STYLE_DIR → $MYSTERY_SPARK/Qwen3-1.7B"
  if base_elbruno "$BASE_DIR"; then
    ln -sfn "$BASE_DIR" "$MYSTERY_SPARK/Qwen3-1.7B-Base"
    echo "[qwen3-tts] Linked $BASE_DIR → $MYSTERY_SPARK/Qwen3-1.7B-Base"
  fi
else
  echo "[qwen3-tts] MysteryAI SparkTTS missing at $MYSTERY_SPARK — skip symlink" >&2
fi

echo "Qwen3-TTS layout:"
ls -lh "$STYLE_DIR/talker_prefill.onnx" "$STYLE_DIR/talker_prefill.onnx.data"
if base_elbruno "$BASE_DIR"; then
  ls -lh "$BASE_DIR/talker_prefill.onnx" "$BASE_DIR/talker_prefill.onnx.data" \
    "$BASE_DIR/speaker_encoder.onnx" "$BASE_DIR/tokenizer_encoder.onnx"
fi
