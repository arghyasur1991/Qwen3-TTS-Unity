# Changelog

## 0.1.0

First release as a standalone package. The ONNX inference path started life in
Spark-TTS-Unity's `qwen3-tts` branch; history is preserved here, and the
Spark-TTS engine that shared that repo is not carried over.

### Engine

- Qwen3-TTS 12 Hz **VoiceDesign** and **Base** checkpoints, addressed
  independently.
- **Streaming output.** `QwenVoice.SpeakStreamAsync` reports
  `SpeechChunk`es as audio is finished and still returns the whole
  `SpeechResult`. First audio at 928 ms against 8799 ms for the utterance on
  the test sentence. The codec decoder has no bounded receptive field, so
  chunks are not decoded as slices — the prefix is re-decoded and only new
  samples are handed over, which makes the concatenation match a single
  decode to 1.86e-6 rather than approximately. Opt-in; `SpeakAsync` is
  unchanged.
- **2.16x faster generation** (2101 -> 972 ms per second of audio), now
  slightly ahead of realtime. Profiling found half the wall clock outside the
  ONNX models entirely: the codec-embedding projections, sixteen 1024x2048
  matrix-vector products per output frame, running as scalar C#. Row-parallel
  with independent accumulators they are 14x faster (and the prefill text MLP
  23x). The KV cache copy that looked like the obvious O(T^2) culprit measured
  0.8% and was left alone.
- **Baked projection tables.** Fifteen of those sixteen products per frame are
  a pure function of export-time weights, so `bake_projected_tables.py` writes
  them and the engine reads instead of computing - another 13x on that stage,
  for ~138 MB per checkpoint. Verified byte-identical against on-demand
  projection on a greedy utterance. Exports without the files fall back to
  projecting on demand; a partial set is rejected rather than mixed.
- **`QwenPrecision.Int8`**, opt-in, resolved per graph with an fp32 fallback.
  1.40x end to end and less resident memory, since the talker is 2.35 GB rather
  than 5.67. Quantize with `Tools~/qwen3_tts_onnx/quantize_int8.py`, which
  holds the output projection and the outermost decoder layers in fp32: without
  that the talker passes every numerical check and still drops phonemes
  (Whisper reads "The Saner sees your ceiling"), and with it a five-line corpus
  transcribes at mean WER 0.017 against 0.000 for fp32. Both checkpoints
  quantize; on the cloning path int8 matches fp32 exactly at mean WER 0.017
  either way, and gains more speed (1.55x vs 1.40x) because its in-context
  prefill puts more of the work in the talker. The vocoder and the clone
  encoders stay fp32. Not bit-identical, so it stays a deliberate choice. fp16 is not offered - ONNX Runtime's CPU
  provider has no fast fp16 kernels for these ops on Apple silicon and measured
  17x slower.
- **`QwenTts.ProfilingEnabled` / `ProfileReport()`** for per-stage wall clock,
  off by default.
- **`QwenTtsSettings.IntraOpThreads`** to override ONNX Runtime's thread
  count. Left at 0 (ORT's choice) by default: forcing 12 on a 16-core M4 Max
  was 33% slower, because decode is bound by streaming weights from memory
  rather than by arithmetic.
- **One talker graph instead of two.** `talker_prefill` and `talker_decode`
  were the same 1.7B weights exported under two signatures, and both had to be
  resident for one utterance. A zero-length KV cache makes the decode graph a
  prefill, so `talker.onnx` serves both: resident memory for an utterance goes
  from 11.35 GB to 5.36 GB and disk from ~13 GB to ~8 GB per checkpoint, with
  outputs bit-exact against both graphs it replaces. Older installs keep the
  pair and still work.
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
