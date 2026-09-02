using System;
using System.Collections.Generic;
using UnityEngine;

namespace QwenTTS
{
    /// <summary>
    /// Which Qwen3-TTS checkpoint a voice runs on. They are separate models
    /// with separate weights, and each is ~13 GB — see the package README for
    /// the memory budget before assuming both can be resident.
    /// </summary>
    public enum QwenCheckpoint
    {
        /// <summary>Speaker is sampled from a natural-language instruct. A new person every generate.</summary>
        VoiceDesign = 0,

        /// <summary>Speaker comes from a reference recording (in-context clone).</summary>
        Base = 1,
    }

    /// <summary>Engine-wide settings. Pass to <see cref="QwenTts.Initialize"/>.</summary>
    public sealed class QwenTtsSettings
    {
        /// <summary>
        /// Folder containing the checkpoint subfolders. Null uses
        /// <c>StreamingAssets/QwenTTS</c>, which is rarely right for a shipped
        /// player holding 13+ GB of weights.
        /// </summary>
        public string ModelRoot;

        public ExecutionProvider ExecutionProvider = ExecutionProvider.CPU;

        /// <summary>
        /// Controls when graphs load and whether they are disposed after use.
        /// <see cref="MemoryUsage.Optimal"/> keeps idle memory near the
        /// embedding tables only, at the cost of reopening the talkers per
        /// utterance.
        /// </summary>
        public MemoryUsage MemoryUsage = MemoryUsage.Balanced;

        public LogLevel LogLevel = LogLevel.INFO;

        /// <summary>Per-stage timing to the log.</summary>
        public bool LogTiming;

        /// <summary>
        /// ONNX Runtime intra-op threads. 0 leaves ORT's own choice alone.
        ///
        /// Autoregressive decode at batch 1 reads every weight once per token,
        /// so both talker and code predictor are bound by how fast weights can
        /// be streamed rather than by arithmetic. Thread count is therefore a
        /// bandwidth knob, and past a point more threads stop helping.
        /// </summary>
        public int IntraOpThreads;
    }

    /// <summary>Per-utterance generation controls. All optional.</summary>
    public sealed class SpeechOptions
    {
        /// <summary>One of <see cref="QwenLanguages.All"/>, or <see cref="QwenLanguages.Auto"/>.</summary>
        public string Language = QwenLanguages.Default;

        /// <summary>Output rate. 0 keeps the model's native 24 kHz.</summary>
        public int SampleRate;

        public float Temperature = 0.9f;
        public int TopK = 50;
        public float TopP = 1f;
        public float RepetitionPenalty = 1.05f;

        /// <summary>Cap on generated 12 Hz frames. 2048 frames is ~2.7 minutes.</summary>
        public int MaxNewTokens = 2048;

        // The code predictor ("sub-talker") samples the 15 residual codebooks.
        public float SubTalkerTemperature = 0.9f;
        public int SubTalkerTopK = 50;
        public float SubTalkerTopP = 1f;

        /// <summary>
        /// Frames in the first streamed chunk. Small means audio starts sooner.
        /// Only used by the streaming overload.
        /// </summary>
        public int FirstChunkFrames = 6;

        /// <summary>
        /// Ceiling on streamed chunk size. Chunks double from
        /// <see cref="FirstChunkFrames"/> up to this, because the codec has to
        /// re-decode the whole prefix for every chunk: fixed small chunks would
        /// make that cost grow with the square of utterance length, while
        /// doubling keeps the total near twice a single decode and still gets
        /// the first audio out early.
        /// </summary>
        public int MaxChunkFrames = 48;

        public static SpeechOptions Default() => new SpeechOptions();

        internal SpeechOptions Validated()
        {
            if (MaxNewTokens <= 0)
                throw new ArgumentOutOfRangeException(nameof(MaxNewTokens), "Must be positive.");
            if (TopK < 0)
                throw new ArgumentOutOfRangeException(nameof(TopK), "Must be zero (disabled) or positive.");
            if (TopP <= 0f || TopP > 1f)
                throw new ArgumentOutOfRangeException(nameof(TopP), "Must be in (0, 1].");
            if (SampleRate < 0)
                throw new ArgumentOutOfRangeException(nameof(SampleRate), "Use 0 for the native rate.");
            if (string.IsNullOrWhiteSpace(Language))
                throw new ArgumentException("Language is required; use QwenLanguages.Auto to let the model decide.");
            if (FirstChunkFrames <= 0)
                throw new ArgumentOutOfRangeException(nameof(FirstChunkFrames), "Must be positive.");
            if (MaxChunkFrames < FirstChunkFrames)
                throw new ArgumentOutOfRangeException(nameof(MaxChunkFrames),
                    "Must be at least FirstChunkFrames.");
            return this;
        }
    }

    /// <summary>Generated audio, as PCM plus its rate.</summary>
    /// <summary>
    /// A run of newly finished audio, handed over while the rest is still
    /// being generated.
    ///
    /// <see cref="Pcm"/> holds only samples not previously reported, so
    /// appending every chunk in order reconstructs the utterance exactly.
    /// Nothing is spliced or crossfaded: the codec decoder is re-run over the
    /// whole prefix each time and its output for already-emitted samples is
    /// stable, so a chunk boundary is not a discontinuity.
    /// </summary>
    public readonly struct SpeechChunk
    {
        public readonly float[] Pcm;
        public readonly int SampleRate;

        /// <summary>Index of the first 12.5 Hz frame this chunk covers.</summary>
        public readonly int FrameStart;

        public readonly int FrameCount;

        /// <summary>True for the chunk that completes the utterance.</summary>
        public readonly bool IsFinal;

        public SpeechChunk(float[] pcm, int sampleRate, int frameStart, int frameCount, bool isFinal)
        {
            Pcm = pcm;
            SampleRate = sampleRate;
            FrameStart = frameStart;
            FrameCount = frameCount;
            IsFinal = isFinal;
        }

        public float Duration => Pcm == null || Pcm.Length == 0
            ? 0f
            : (float)Pcm.Length / SampleRate;
    }

    public readonly struct SpeechResult
    {
        public readonly float[] Pcm;
        public readonly int SampleRate;

        public SpeechResult(float[] pcm, int sampleRate)
        {
            Pcm = pcm;
            SampleRate = sampleRate;
        }

        public bool IsEmpty => Pcm == null || Pcm.Length == 0;

        public float Duration => IsEmpty ? 0f : (float)Pcm.Length / SampleRate;

        /// <summary>Wraps the PCM in an AudioClip. Main thread only.</summary>
        public AudioClip ToAudioClip(string name = "QwenTtsSpeech")
        {
            if (IsEmpty)
                return null;
            var clip = AudioClip.Create(name, Pcm.Length, 1, SampleRate, false);
            clip.SetData(Pcm, 0);
            return clip;
        }
    }

    /// <summary>Progress of the autoregressive loop, in 12 Hz frames.</summary>
    public readonly struct SpeechProgress
    {
        /// <summary>Frames generated so far. Each is 80 ms of audio.</summary>
        public readonly int Frames;

        /// <summary>The <see cref="SpeechOptions.MaxNewTokens"/> ceiling, not a prediction.</summary>
        public readonly int MaxFrames;

        public SpeechProgress(int frames, int maxFrames)
        {
            Frames = frames;
            MaxFrames = maxFrames;
        }

        public float Seconds => Frames * 0.08f;
    }

    /// <summary>What a designed voice should sound like.</summary>
    public sealed class VoiceDesignSpec
    {
        /// <summary>
        /// Natural-language description — this *is* the voice. Every generate
        /// with the same instruct samples a different person; lock a take with
        /// <see cref="QwenTts.CreateClonedVoiceAsync"/> to keep one.
        /// </summary>
        public string Instruct;

        public string Language = QwenLanguages.Default;

        public VoiceDesignSpec() { }

        public VoiceDesignSpec(string instruct, string language = null)
        {
            Instruct = instruct;
            if (!string.IsNullOrWhiteSpace(language))
                Language = language;
        }
    }

    /// <summary>Install and residency state of one checkpoint.</summary>
    public sealed class CheckpointStatus
    {
        public QwenCheckpoint Checkpoint { get; internal set; }

        /// <summary>Every file the engine opens is on disk.</summary>
        public bool Installed { get; internal set; }

        /// <summary>Tables and graph wrappers are constructed in this process.</summary>
        public bool Loaded { get; internal set; }

        public string Directory { get; internal set; }

        public IReadOnlyList<string> MissingFiles { get; internal set; }

        public long InstalledBytes { get; internal set; }

        public override string ToString() =>
            $"{Checkpoint}: installed={Installed} loaded={Loaded} " +
            $"({InstalledBytes / (1024f * 1024f * 1024f):0.0} GB at {Directory})" +
            (Installed ? "" : $" missing {MissingFiles.Count} file(s)");
    }

    /// <summary>Languages the 12 Hz checkpoints accept.</summary>
    public static class QwenLanguages
    {
        public const string Auto = "auto";
        public const string Default = "english";

        public static readonly IReadOnlyList<string> All = new[]
        {
            "chinese", "english", "german", "italian", "portuguese",
            "spanish", "japanese", "korean", "french", "russian",
        };

        public static bool IsSupported(string language)
        {
            if (string.IsNullOrWhiteSpace(language))
                return false;
            var lower = language.Trim().ToLowerInvariant();
            if (lower == Auto)
                return true;
            for (int i = 0; i < All.Count; i++)
            {
                if (All[i] == lower)
                    return true;
            }
            return false;
        }
    }
}
