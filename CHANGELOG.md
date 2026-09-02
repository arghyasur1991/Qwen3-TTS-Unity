# Changelog

## 0.1.0

First release as a standalone package. The ONNX inference path started life in
Spark-TTS-Unity's `qwen3-tts` branch; history is preserved here, and the
Spark-TTS engine that shared that repo is not carried over.

### Engine

- Qwen3-TTS 12 Hz **VoiceDesign** and **Base** checkpoints, addressed
  independently.
- Base cloning follows Qwen's reference implementation: the in-context prompt
  sums the text and codec streams position-aligned (the layout
  `generate_voice_clone` actually defaults to), the reference codes are decoded
  together with the generated ones and the reference portion trimmed off, and
  the speaker encoder's mel filterbank uses librosa's Slaney defaults.
- Band-limited resampling. Linear interpolation folded 8-12 kHz back into the
  speech band when downsampling and left a stair-stepped spectrum going up,
  both of which move a reference away from the take being cloned.
- WAV reading honours the header. The previous code assumed 16-bit mono at
  16 kHz, so a 24 kHz reference would have been read 1.5x slow.

### API

- `QwenTts` facade with explicit residency: `WarmUpAsync`, `Evict`, `EvictAll`,
  `GetStatus`. Each checkpoint is ~13 GB resident and they are normally needed
  in different phases, so holding both is opt-in rather than automatic.
- Configurable `ModelRoot`. Weights no longer have to live in StreamingAssets.
- Per-utterance `SpeechOptions`: language (10 plus auto), sampling and
  sub-talker sampling, output rate, frame cap.
- `CancellationToken` and `IProgress<SpeechProgress>` on generation.
- Clone prompts persist. A saved cloned voice stores its x-vector and reference
  codes, so reloading no longer re-runs the speaker encoder and the 12 Hz
  tokenizer behind a ~370 MB session.
- `VoiceDesignSpec.Instruct` is the voice. The previous API took
  gender/pitch/speed dropdowns and synthesised an English sentence from them
  inside the library, which was host policy in engine code and English-only.
- Voice manifests no longer name files that may not exist.

### Performance

- Projected codec rows are filled on first use. Projecting every row of all 16
  tables up front was ~71 GFLOP to serve the ~16 rows a frame reads, and
  dominated a ~14 s load; checkpoints now come up in well under a second.
- Talker decode and the 15-step code-predictor loop reuse KV buffers over the
  `OrtValue` API. Copying every output every step was gigabytes of large-object
  churn per line.
- The two checkpoints have separate locks, so they can generate concurrently.

### Editor

- Domain-reload keep-alive moved into the editor assembly. The runtime side is
  a small handoff with no reflection; the native-handle and OrtEnv work lives
  where domain reloads actually happen. It reports unavailability instead of
  failing if a future ONNX Runtime moves the private members it uses.
- Embedding tables are no longer stashed across reload — they re-read in under
  a second, which did not justify a hand-packed blob of raw pointers.
- **Window → Qwen3 TTS → Model Status** replaces the Spark deployment tool.

### Removed

- The Spark-TTS BiCodec engine, its tokenizer service and model wrappers.
- Preset-speaker ("CustomVoice") entry points, which were the only readers of
  `speaker_ids.json`.
- `ORTModel`'s index-based input staging and preallocated-output paths: no Qwen
  graph has fixed output shapes. Load policy, execution-provider selection with
  CoreML caching and fallbacks, iOS handling and off-thread session opening are
  all retained.
