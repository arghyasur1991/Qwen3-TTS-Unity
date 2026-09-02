# Qwen3-TTS for Unity

On-device text-to-speech using the **Qwen3-TTS 12 Hz 1.7B** checkpoints through
ONNX Runtime. Two capabilities, each a separate model:

| Checkpoint | What it does |
|---|---|
| **VoiceDesign** | Invents a speaker from a natural-language instruct. A different person every generate. |
| **Base** | Clones a speaker from a reference recording, using Qwen's in-context path (reference codes + transcript + x-vector). |

The usual flow is both: design until you like a take, then clone that take so it
stays put.

This package **does not include or download weights.** Export them yourself with
`Tools~/qwen3_tts_onnx/export_all.py` and point the package at the result.

---

## Read this first: it is not a small model

Measured on an Apple-silicon Mac, CPU execution provider, FP32:

| | Per checkpoint |
|---|---|
| Disk | ~8 GB (plus ~138 MB of baked projection tables) |
| Resident when loaded | ~7.0 GB (5.4 GB of ONNX sessions + ~1.5 GB embedding tables) |
| Cold session open | ~11 s (~3 s with a warm page cache) |
| Generation speed | ~0.97x of real time (finishes just ahead of playback) |

ONNX Runtime does **not** keep the external `.onnx.data` lazily mapped — resident
memory tracks file size roughly 1:1. Budget accordingly:

- **One checkpoint resident:** 16 GB machine is fine, 32 GB comfortable.
- **Both resident:** wants 32 GB. Usually avoidable — see *Residency* below.

The talker is exported as a single graph that does both prefill and decode
(a zero-length KV cache makes it a prefill). Exports predating that carry a
`talker_prefill` + `talker_decode` pair, which is the same weights twice and
doubles both figures above; they are still read, but re-exporting with
`Tools~/qwen3_tts_onnx/export_talker_unified.py` halves the cost.
- **Mobile and XR are out of scope** at FP32. The package compiles everywhere,
  but 13 GB of weights does not fit in a phone or headset app.

Supported and tested: Unity 6000.x editor, Windows and macOS standalone.

## Install

1. Export a checkpoint (see `Tools~/qwen3_tts_onnx/`). You need one folder per
   checkpoint, named:
   - `Qwen3-1.7B-VoiceDesign`
   - `Qwen3-1.7B-Base`
2. Put those folders under a root of your choosing.
3. Point the package at the root and check what it found:

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
editor and usually wrong for a shipped player — StreamingAssets is copied into
the build. Prefer a folder you install or download into.

`Pocket Hamlet` style hosts can also use **Window → Qwen3 TTS → Model Status** to
see the same information without writing code.

## Designing a voice

```csharp
var voice = await QwenTts.CreateDesignedVoiceAsync(new VoiceDesignSpec(
    "Male, thirties, warm and conversational, close-mic, not a narrator."));

var take = await voice.SpeakAsync("So you actually put it on.");
audioSource.clip = take.ToAudioClip();     // main thread
audioSource.Play();
```

The instruct *is* the voice. Calling `SpeakAsync` again gives you the same style
in a different person's mouth — that is what the checkpoint does. There is no
seed that pins it.

## Keeping a voice: clone the take

```csharp
// Render something long enough to characterise the speaker.
var line  = "I am Alex. I am your friend, and a really brilliant scientist.";
var take  = await voice.SpeakAsync(line);

// The transcript is required: it is what makes this in-context cloning
// rather than a generic x-vector match.
var locked = await QwenTts.CreateClonedVoiceAsync(take.ToAudioClip(), line);

var next = await locked.SpeakAsync("Careful with that.");
```

Reference audio quality decides clone quality:

- **At least 4 seconds.** Shorter and there are too few 12 Hz frames to pin a
  speaker; takes drift between utterances. The package warns below this.
- **Keep it at 24 kHz.** The speaker encoder reads mel up to 12 kHz, so a 16 kHz
  reference has the top of the identifying band already missing. `SpeakAsync`
  returns 24 kHz unless you ask for something else. The package warns if a
  reference is lower.
- **Pass the exact transcript.** Without it you get a stable voice that is not
  the one in your recording.

## Streaming

Generation runs a little faster than playback, so audio can start long before
the line is finished.

```csharp
var player = new Progress<SpeechChunk>(chunk =>
{
    // Reported from a worker thread. Marshal before touching an AudioClip.
    // chunk.Pcm holds only samples not reported before, so appending each
    // chunk in order reproduces the utterance.
    Enqueue(chunk.Pcm, chunk.SampleRate);
});

// Still returns the whole thing, for callers that also want to cache it.
var whole = await voice.SpeakStreamAsync("Careful with that.", player);
```

First audio arrives in roughly a second instead of at the end. Tune with
`SpeechOptions.FirstChunkFrames` (default 6, about half a second) and
`MaxChunkFrames` (48).

The chunks are not decoded independently. This codec decoder's output depends
on its whole input rather than a bounded window — giving a chunk 24 frames of
preceding context still leaves errors of 0.59 on a signal in [-1, 1] — so
overlap-and-trim does not apply. Instead the prefix is re-decoded each time
and only new samples are handed over, which is exact rather than approximate:
concatenated chunks match a single decode to about 1.9e-6. The price is
re-decoding, so chunk sizes double rather than staying small, keeping total
decode work near twice a single pass.

## Saving and reloading

```csharp
await locked.SaveAsync(folder, take);              // take is optional
var again = await QwenTts.LoadVoiceAsync(folder);
```

A saved clone stores the derived prompt (x-vector plus reference codes) next to
the reference audio, so reloading is a file read rather than another
speaker-encoder and tokenizer-encoder run. If the prompt file is from an older
export it is ignored and the prompt is re-derived from the reference.

## Residency

Each checkpoint is ~13 GB, and the two are normally needed in *different phases*
— VoiceDesign while the player is picking a voice, Base for everything after. So
load and drop them explicitly instead of holding both:

```csharp
await QwenTts.WarmUpAsync(QwenCheckpoint.VoiceDesign);   // loading screen
// ... player auditions voices, picks one, you clone it ...
QwenTts.Evict(QwenCheckpoint.VoiceDesign);               // ~13 GB back
await QwenTts.WarmUpAsync(QwenCheckpoint.Base);
```

`WarmUpAsync` matters: without it the first utterance pays the whole session
open (~21 s cold). `MemoryUsage` controls the default policy:

| Mode | Behaviour |
|---|---|
| `Performance` | Load eagerly, never drop. |
| `Balanced` | Load on first use, keep. Default. |
| `Optimal` | Load per use, dispose after. Idle stays near the embedding tables (~1.5 GB) at the cost of reopening per utterance. |

## Generation options

```csharp
await voice.SpeakAsync(text, new SpeechOptions
{
    Language   = QwenLanguages.Default,   // 10 languages, or QwenLanguages.Auto
    SampleRate = 24000,                   // 0 keeps native
    Temperature = 0.9f, TopK = 50, TopP = 1f, RepetitionPenalty = 1.05f,
    MaxNewTokens = 2048,                  // 12 Hz frames; 2048 is ~2.7 minutes
},
progress: new Progress<SpeechProgress>(p => Debug.Log($"{p.Seconds:0.0}s")),
cancellationToken: token);
```

Defaults match Qwen's own generate config. Generation is ~4x slower than real
time, so pass a `CancellationToken` for anything the player can skip.

## Threading

`SpeakAsync`, `WarmUpAsync` and the create calls do their work on the thread
pool and are safe to await from the main thread. `AudioClip.GetData` and
`AudioClip.Create` are Unity main-thread APIs, so a reference clip is read on
the calling thread and `SpeechResult.ToAudioClip()` must be called on the main
thread — that is why generation returns PCM rather than a clip.

The two checkpoints have independent locks, so a designed and a cloned voice can
generate at the same time. Two utterances on the *same* checkpoint serialise,
because the talker reuses its KV and sampler buffers.

## Editor: holding models across a script compile

A domain reload would otherwise throw away ~22 s of session loading on every
compile. `QwenTTS.Editor` can detach the native ONNX allocations before the
reload and reattach them after. It is opt-in, does nothing unless the host asks
for it, and degrades to a normal reload if a future ONNX Runtime moves the
private members it relies on.

## Licence and attribution

Apache-2.0. The ONNX inference path derives from
[ElBruno.QwenTTS](https://github.com/elbruno/ElBruno.QwenTTS) (MIT), and the
prompt construction follows Alibaba's `qwen-tts` reference implementation
(Apache-2.0). Model weights are Alibaba's, under their own licence. See
`THIRD_PARTY_NOTICES.md`.
