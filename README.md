# Qwen3-TTS for Unity

On-device text-to-speech using the **Qwen3-TTS 12.5 Hz 1.7B** checkpoints
through ONNX Runtime. Two capabilities, each a separate checkpoint:

| Checkpoint | What it does |
|---|---|
| **VoiceDesign** | Invents a speaker from a natural-language description. A different person every generate. |
| **Base** | Clones a speaker from a reference recording, using Qwen's in-context path (reference codes + transcript + speaker embedding). |

The usual flow is both: design until you like a take, then clone that take so
the voice stays put.

This package **does not include or download weights.** Export them yourself
with the scripts in `Tools~/qwen3_tts_onnx/` and point the package at the
result.

## Read this first: it is not a small model

Measured on an Apple-silicon laptop, CPU execution provider, fp32:

| | Per checkpoint |
|---|---|
| Disk | ~8 GB |
| Resident once loaded | ~7 GB (5.4 GB of ONNX sessions, ~1.5 GB of embedding tables) |
| Cold session open | ~11 s, ~3 s with a warm page cache |
| Generation | ~0.97× of real time — finishes just ahead of playback |

ONNX Runtime does **not** keep external `.onnx.data` lazily mapped, so resident
memory tracks file size roughly 1:1. Budget accordingly:

- **One checkpoint resident:** 16 GB machine is fine, 32 GB comfortable.
- **Both resident:** wants 32 GB, and is usually avoidable — see
  [Residency](#residency).
- **Mobile and XR are out of scope** at fp32. The package compiles anywhere,
  but these weights do not fit in a phone or headset app.

Developed against Unity 6000.x, Windows and macOS.

## Install

1. Export both checkpoints (see `Tools~/qwen3_tts_onnx/README.md`). You get one
   folder each, named `Qwen3-1.7B-VoiceDesign` and `Qwen3-1.7B-Base`.
2. Put them under a root of your choosing.
3. Point the package at that root and check what it found:

```csharp
QwenTts.Initialize(new QwenTtsSettings
{
    ModelRoot = Path.Combine(Application.persistentDataPath, "QwenTTS"),
    MemoryUsage = MemoryUsage.Balanced,
    LogLevel = LogLevel.INFO,
});

var status = QwenTts.GetStatus(QwenCheckpoint.Base);
Debug.Log(status);   // installed / loaded / missing files / bytes
```

`ModelRoot` defaults to `StreamingAssets/QwenTTS`, which is convenient in the
editor and usually wrong for a shipped player, because StreamingAssets is
copied into the build. Prefer a folder you install or download into.

**Window → Qwen3 TTS → Model Status** shows the same information without
writing code.

## Designing a voice

```csharp
var voice = await QwenTts.CreateDesignedVoiceAsync(new VoiceDesignSpec(
    "Male, thirties, warm and conversational, close-mic, not a narrator."));

var take = await voice.SpeakAsync("So you actually put it on.");
audioSource.clip = take.ToAudioClip();     // main thread
audioSource.Play();
```

The description *is* the voice. Calling `SpeakAsync` again gives you the same
style in a different person's mouth — that is what the checkpoint does, and
there is no seed that pins it. To keep a voice, clone it.

## Keeping a voice: clone the take

```csharp
// Render something long enough to characterise the speaker.
var line  = "I am Alex. I am your friend, and a really brilliant scientist.";
var take  = await voice.SpeakAsync(line);

// The transcript is required: it is what makes this in-context cloning
// rather than a generic speaker-embedding match.
var locked = await QwenTts.CreateClonedVoiceAsync(take.ToAudioClip(), line);

var next = await locked.SpeakAsync("Careful with that.");
```

Reference audio quality decides clone quality:

- **At least 4 seconds.** Shorter and there are too few frames to pin a
  speaker, so takes drift between utterances. The package warns below this.
- **Keep it at 24 kHz.** The speaker encoder reads mel up to 12 kHz, so a
  16 kHz reference has the top of the identifying band already missing.
  `SpeakAsync` returns 24 kHz unless you ask for something else, and the
  package warns if a reference is lower.
- **Pass the exact transcript.** Without it you get a stable voice that is not
  the one in your recording.

## Saving and reloading

```csharp
await locked.SaveAsync(folder, take);              // take is optional
var again = await QwenTts.LoadVoiceAsync(folder);
```

A saved clone stores its derived prompt — speaker embedding plus reference
codes — beside the reference audio, so reloading is a file read rather than
another speaker-encoder and tokenizer-encoder run. A prompt file from an
incompatible export is ignored and the prompt is re-derived.

## Streaming

Generation runs a little faster than playback, so audio can start long before
the line is finished.

```csharp
var player = new Progress<SpeechChunk>(chunk =>
{
    // Reported from a worker thread; marshal before touching an AudioClip.
    // chunk.Pcm holds only samples not reported before, so appending each
    // chunk in order reproduces the utterance.
    Enqueue(chunk.Pcm, chunk.SampleRate);
});

// Still returns the whole thing, for callers that also want to cache it.
var whole = await voice.SpeakStreamAsync("Careful with that.", player);
```

First audio arrives in about a second rather than at the end. Tune with
`SpeechOptions.FirstChunkFrames` (default 6, roughly half a second) and
`MaxChunkFrames` (48).

Chunks are not decoded independently. This codec decoder's output depends on
its whole input rather than a bounded window, so overlap-and-trim does not
apply here; instead the prefix is re-decoded for each chunk and only new
samples are handed over. That makes concatenated chunks match a single decode
to within about 2e-6 rather than approximately, at the cost of re-decoding —
which is why chunk sizes double rather than staying small.

## Precision

The talker and code predictor read every weight once per generated token, so
they are limited by memory bandwidth rather than arithmetic. int8 weights cut
that traffic and are about 1.4× faster end to end, with the talker resident at
2.35 GB instead of 5.67:

```csharp
QwenTts.Initialize(new QwenTtsSettings
{
    ModelRoot = "...",
    Precision = QwenPrecision.Int8,   // default is Float32
});
```

Produce the quantized graphs with `Tools~/qwen3_tts_onnx/quantize_int8.py`.
Precision is resolved per graph, so a checkpoint missing `talker_int8.onnx`
uses the fp32 talker rather than failing. Both checkpoints quantize, and clones
gain slightly more than designed voices (1.55× against 1.40×) because their
in-context prefill puts more of the work in the talker. The vocoder and the
speaker and tokenizer encoders stay fp32, so nothing about how a voice is
*captured* is quantized.

It is opt-in because it is not free. Quantized audio is not bit-identical to
fp32 and a voice can differ subtly, so listen before shipping it. Judge it by
transcribing the output rather than by numerical error: quantizing every layer
passes every numerical check and still drops phonemes, which is why
`quantize_int8.py` holds the output projection and the outermost decoder layers
back by default.

fp16 is deliberately not offered. ONNX Runtime's CPU provider has no fast fp16
kernels for these operations on Apple silicon, so it casts and computes
element-wise: measurably slower than fp32 while being numerically near-perfect.

## Residency

The two checkpoints are normally needed in *different phases* — VoiceDesign
while the player is picking a voice, Base for everything after — so load and
drop them explicitly instead of holding both:

```csharp
await QwenTts.WarmUpAsync(QwenCheckpoint.VoiceDesign);   // loading screen
// ... player auditions voices, picks one, you clone it ...
QwenTts.Evict(QwenCheckpoint.VoiceDesign);               // ~7 GB back
await QwenTts.WarmUpAsync(QwenCheckpoint.Base);
```

`WarmUpAsync` matters: without it the first utterance pays the whole session
open. `MemoryUsage` sets the default policy:

| Mode | Behaviour |
|---|---|
| `Performance` | Load eagerly, never drop. |
| `Balanced` | Load on first use, then keep. Default. |
| `Optimal` | Load per use and dispose after. Idle stays near the embedding tables (~1.5 GB), at the cost of reopening per utterance. |

## Generation options

```csharp
await voice.SpeakAsync(text, new SpeechOptions
{
    Language   = QwenLanguages.Default,   // 10 languages, or QwenLanguages.Auto
    SampleRate = 24000,                   // 0 keeps native
    Temperature = 0.9f, TopK = 50, TopP = 1f, RepetitionPenalty = 1.05f,
    MaxNewTokens = 2048,                  // frames at 12.5 Hz; 2048 is ~2.7 minutes
},
progress: new Progress<SpeechProgress>(p => Debug.Log($"{p.Seconds:0.0}s")),
cancellationToken: token);
```

Defaults match Qwen's own generate config. Two settings are worth knowing about
even if you never change them:

- **`RepetitionPenalty` above 1 is load-bearing.** Greedy decoding with the
  penalty disabled can loop without ever emitting end-of-speech.
- **`MaxNewTokens` bounds cost, not just length.** The per-step KV cache copy
  grows with the square of sequence length, so it is negligible for an
  utterance and expensive for a runaway one.

Pass a `CancellationToken` for anything the player can skip.

## Threading

`SpeakAsync`, `WarmUpAsync` and the create calls do their work on the thread
pool and are safe to await from the main thread. `AudioClip.GetData` and
`AudioClip.Create` are Unity main-thread APIs, so a reference clip is read on
the calling thread and `SpeechResult.ToAudioClip()` must be called on the main
thread — which is why generation returns PCM rather than a clip.

The two checkpoints have independent locks, so a designed and a cloned voice
can generate at the same time. Two utterances on the *same* checkpoint
serialise, because the talker reuses its KV and sampler buffers.

## Logging

ONNX Runtime's own diagnostics are routed into the Unity console, tagged with
the model they came from:

```
[ONNX-INFO][talker][onnxruntime] Session successfully initialized.
```

`QwenTtsSettings.LogLevel` sets the verbosity. Note that ONNX Runtime allows
**one environment per process**, and whichever library creates it owns the
logging sink for every library in the process. If another ONNX library in your
project initializes first, this package uses its sink and leaves the level
alone; if this package initializes first, the other library can attribute its
own models with `QwenTts.SetOnnxLogContext(name)`.

## Licence and attribution

Apache-2.0. The ONNX inference path derives from
[ElBruno.QwenTTS](https://github.com/elbruno/ElBruno.QwenTTS) (MIT), and prompt
construction follows Alibaba's `qwen-tts` reference implementation
(Apache-2.0). Model weights are Alibaba's, under their own licence. See
`THIRD_PARTY_NOTICES.md`.
