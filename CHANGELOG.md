# Changelog

## 0.1.0

First release. The ONNX inference path began as a branch of Spark-TTS-Unity and
that history is preserved here; the Spark-TTS engine is not part of this
package.

### Voices

- Qwen3-TTS 12.5 Hz **VoiceDesign** and **Base** checkpoints, addressed
  independently and loadable one at a time.
- Voice design from a natural-language description. The description is the
  whole specification; the package does not compose prose from enum knobs on
  the host's behalf.
- Cloning follows Qwen's in-context reference implementation: the prompt sums
  the text and codec streams position-aligned, the reference codes are decoded
  together with the generated ones and the reference portion trimmed off, and
  the speaker encoder's mel filterbank uses librosa's Slaney defaults.
- Cloned voices persist their derived prompt — speaker embedding plus reference
  codes — so reloading one is a file read rather than another speaker-encoder
  and tokenizer-encoder pass.

### API

- `QwenTts` facade with explicit residency: `WarmUpAsync`, `Evict`,
  `EvictAll`, `GetStatus`. A checkpoint is several gigabytes resident and the
  two are normally wanted in different phases, so holding both is a choice
  rather than the default.
- Configurable `ModelRoot`; weights need not live in StreamingAssets.
- Per-utterance `SpeechOptions`: language (ten plus automatic), sampling and
  sub-talker sampling, output rate, frame cap.
- `CancellationToken` and `IProgress<SpeechProgress>` on generation.
- **Streaming.** `SpeakStreamAsync` reports `SpeechChunk` values as audio
  finishes and still returns the whole `SpeechResult`. Chunks carry only
  samples not previously reported, and because the prefix is re-decoded rather
  than sliced, concatenating them reproduces a single decode to within ~2e-6.
- **`QwenPrecision.Int8`**, opt-in and resolved per graph with an fp32
  fallback: ~1.4× faster, with the talker resident at 2.35 GB instead of 5.67.
  fp16 is not offered, having measured far slower than fp32 on ONNX Runtime's
  CPU provider, which has no fast fp16 kernels for these operations on Apple
  silicon.
- Optional per-stage timing via `QwenTts.ProfilingEnabled` and
  `ProfileReport()`, off by default.
- `QwenTtsSettings.IntraOpThreads` to override ONNX Runtime's thread count.
  Defaults to ORT's own choice, which measured better than forcing a value.

### Performance

- The talker is one graph serving both prefill and decode, a zero-length KV
  cache making it a prefill. Exporting it twice, as separate prefill and decode
  graphs, meant the same weights resident twice for one utterance.
- The codec-embedding projections — sixteen matrix-vector products per output
  frame — are row-parallel, and fifteen of the sixteen are pre-applied at
  export time by `bake_projected_tables.py` and read rather than computed.
  Together these account for most of the package's generation speed.
- Projected rows that are not baked are filled on first use. Projecting every
  row of every table up front served sixteen rows per frame at the cost of
  tens of gigaflops and dominated load time.
- Talker decode and the code-predictor loop reuse KV buffers over the
  `OrtValue` API, rather than copying every output every step.
- Band-limited resampling. Linear interpolation folded 8–12 kHz back into the
  speech band when downsampling and left a stair-stepped spectrum going up,
  both of which move a reference away from the take being cloned.
- WAV reading honours the header rather than assuming 16-bit mono at 16 kHz,
  which would read a 24 kHz reference 1.5× slow.

### Editor

- **Window → Qwen3 TTS → Model Status** reports what is installed, what is
  loaded, and which files are missing.
- Opt-in domain-reload keep-alive for native ONNX allocations, so a script
  compile does not discard every session open. It reports unavailability rather
  than failing if a future ONNX Runtime moves the private members it uses.

### Tools

- `Tools~/qwen3_tts_onnx/` exports a HuggingFace checkpoint to the layout this
  package reads, validates each exported graph against the PyTorch module it
  came from, and can pre-apply the codec projections and quantize to int8.
