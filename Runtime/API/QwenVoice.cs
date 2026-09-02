using System;
using System.IO;
using System.Threading;
using System.Threading.Tasks;
using QwenTTS.Audio;
using QwenTTS.Engine;
using QwenTTS.Internal;

namespace QwenTTS
{
    /// <summary>
    /// A speaker you can talk with. Created by
    /// <see cref="QwenTts.CreateDesignedVoiceAsync"/>,
    /// <see cref="QwenTts.CreateClonedVoiceAsync"/> or
    /// <see cref="QwenTts.LoadVoiceAsync"/>.
    ///
    /// Designed voices carry an instruct and re-roll the speaker on every
    /// utterance. Cloned voices carry a fixed prompt and are stable.
    /// </summary>
    public sealed class QwenVoice
    {
        /// <summary>
        /// Applies the caller's output rate to each chunk, so streamed audio
        /// and the final <see cref="SpeechResult"/> agree. Resampling per chunk
        /// rather than once at the end introduces a boundary the whole-buffer
        /// path does not have; at 24 kHz to 48 kHz or 44.1 kHz it is far below
        /// audibility, and requesting the native rate avoids it entirely.
        /// </summary>
        sealed class ChunkRelay : IProgress<SpeechChunk>
        {
            readonly IProgress<SpeechChunk> _inner;
            readonly int _rate;

            public ChunkRelay(IProgress<SpeechChunk> inner, int rate)
            {
                _inner = inner;
                _rate = rate <= 0 ? QwenTts.NativeSampleRate : rate;
            }

            public void Report(SpeechChunk value)
            {
                if (_rate == QwenTts.NativeSampleRate)
                {
                    _inner.Report(value);
                    return;
                }
                var pcm = AudioResample.Resample(value.Pcm, QwenTts.NativeSampleRate, _rate);
                _inner.Report(new SpeechChunk(
                    pcm, _rate, value.FrameStart, value.FrameCount, value.IsFinal));
            }
        }

        readonly QwenTtsEngine _engine;
        readonly ClonePrompt _prompt;
        readonly float[] _referenceSamples24k;
        readonly int _referenceSourceRate;

        QwenVoice(QwenTtsEngine engine, QwenCheckpoint checkpoint, string instruct,
            string referenceText, string language, ClonePrompt prompt,
            float[] referenceSamples24k, int referenceSourceRate)
        {
            _engine = engine ?? throw new ArgumentNullException(nameof(engine));
            Checkpoint = checkpoint;
            Instruct = instruct;
            ReferenceText = referenceText;
            Language = language;
            _prompt = prompt;
            _referenceSamples24k = referenceSamples24k;
            _referenceSourceRate = referenceSourceRate;
        }

        internal static QwenVoice Designed(QwenTtsEngine engine, string instruct, string language) =>
            new QwenVoice(engine, QwenCheckpoint.VoiceDesign, instruct, null, language,
                default, null, 0);

        internal static QwenVoice Cloned(QwenTtsEngine engine, ClonePrompt prompt,
            string referenceText, string language, float[] referenceSamples24k, int referenceSourceRate) =>
            new QwenVoice(engine, QwenCheckpoint.Base, null, referenceText, language,
                prompt, referenceSamples24k, referenceSourceRate);

        /// <summary>Which checkpoint this voice generates on.</summary>
        public QwenCheckpoint Checkpoint { get; }

        public bool IsCloned => Checkpoint == QwenCheckpoint.Base;

        /// <summary>Natural-language description, for a designed voice.</summary>
        public string Instruct { get; }

        /// <summary>Transcript of the reference recording, for an in-context clone.</summary>
        public string ReferenceText { get; }

        /// <summary>Language used when none is given in <see cref="SpeechOptions"/>.</summary>
        public string Language { get; }

        /// <summary>True when this clone uses the in-context path rather than x-vector only.</summary>
        public bool UsesInContextCloning => IsCloned && _prompt.HasIclCodes &&
                                            !string.IsNullOrWhiteSpace(ReferenceText);

        /// <summary>
        /// Renders <paramref name="text"/>. Runs on a worker thread; the
        /// returned PCM can be turned into an AudioClip on the main thread with
        /// <see cref="SpeechResult.ToAudioClip"/>.
        /// </summary>
        /// <param name="options">Language, sampling and output rate. Null uses defaults.</param>
        /// <param name="progress">Reports 12 Hz frames as they are generated.</param>
        public Task<SpeechResult> SpeakAsync(string text, SpeechOptions options = null,
            IProgress<SpeechProgress> progress = null, CancellationToken cancellationToken = default)
            => SpeakInternalAsync(text, options, null, progress, cancellationToken);

        /// <summary>
        /// Renders <paramref name="text"/> and hands over audio in pieces as it
        /// becomes available, instead of only at the end.
        ///
        /// Each <see cref="SpeechChunk"/> carries only samples not reported
        /// before, so appending them in order reproduces the utterance; the
        /// returned <see cref="SpeechResult"/> is still the whole thing, for
        /// callers that also want to cache or save it. Chunks are reported from
        /// the worker thread, so marshal to the main thread before touching an
        /// AudioClip.
        ///
        /// Worth using because generation runs slightly faster than playback:
        /// the first chunk arrives in a fraction of a second and the rest keeps
        /// ahead of a listener, turning a multi-second wait into near-immediate
        /// speech. It costs some total throughput — see
        /// <see cref="SpeechOptions.MaxChunkFrames"/>.
        /// </summary>
        public Task<SpeechResult> SpeakStreamAsync(string text, IProgress<SpeechChunk> chunks,
            SpeechOptions options = null, IProgress<SpeechProgress> progress = null,
            CancellationToken cancellationToken = default)
        {
            if (chunks == null)
                throw new ArgumentNullException(nameof(chunks));
            return SpeakInternalAsync(text, options, chunks, progress, cancellationToken);
        }

        async Task<SpeechResult> SpeakInternalAsync(string text, SpeechOptions options,
            IProgress<SpeechChunk> chunkSink, IProgress<SpeechProgress> progress,
            CancellationToken cancellationToken)
        {
            if (string.IsNullOrWhiteSpace(text))
                throw new ArgumentException("Text is empty.", nameof(text));

            options = (options ?? SpeechOptions.Default()).Validated();
            string language = string.IsNullOrWhiteSpace(options.Language) ? Language : options.Language;
            var sampling = SamplingParams.From(options);

            var engine = _engine;
            var prompt = _prompt;
            string instruct = Instruct;
            string referenceText = ReferenceText;
            bool cloned = IsCloned;

            var stream = chunkSink == null
                ? default
                : new Engine.StreamRequest(
                    new ChunkRelay(chunkSink, options.SampleRate),
                    options.FirstChunkFrames, options.MaxChunkFrames);

            float[] pcm24 = await BackgroundWork.Run(() => cloned
                ? engine.SynthesizeCloned(text, prompt, referenceText, language, sampling,
                    progress, stream, cancellationToken)
                : engine.SynthesizeDesigned(text, instruct, language, sampling,
                    progress, stream, cancellationToken))
                .ConfigureAwait(false);

            Internal.GenerationProfiler.StopWall();

            if (pcm24 == null || pcm24.Length == 0)
                throw new InvalidOperationException("Generation produced no audio.");

            int rate = options.SampleRate <= 0 ? QwenTts.NativeSampleRate : options.SampleRate;
            float[] pcm = rate == QwenTts.NativeSampleRate
                ? pcm24
                : AudioResample.Resample(pcm24, QwenTts.NativeSampleRate, rate);
            return new SpeechResult(pcm, rate);
        }

        /// <summary>
        /// Writes the voice so <see cref="QwenTts.LoadVoiceAsync"/> can restore
        /// it. For a clone that includes the derived prompt, so reloading does
        /// not re-run the encoders, plus the reference audio so the prompt can
        /// be rebuilt if the encoder export ever changes.
        /// </summary>
        /// <param name="folder">Created if absent.</param>
        /// <param name="sample">
        /// Optional rendered take to store as <c>sample.wav</c>. Supply one if
        /// the voice is going to be auditioned or used as a clone reference
        /// later — nothing is written when it is null, and the manifest then
        /// does not claim a sample exists.
        /// </param>
        public async Task SaveAsync(string folder, SpeechResult? sample = null)
        {
            if (string.IsNullOrWhiteSpace(folder))
                throw new ArgumentException("Folder is required.", nameof(folder));
            Directory.CreateDirectory(folder);

            bool wroteSample = false;
            if (sample.HasValue && !sample.Value.IsEmpty)
            {
                var bytes = WavCodec.Encode(sample.Value.Pcm, sample.Value.SampleRate);
                await WriteAllBytesAsync(Path.Combine(folder, VoiceManifest.SampleFileName), bytes)
                    .ConfigureAwait(false);
                wroteSample = true;
            }

            bool wroteReference = false;
            if (IsCloned && _referenceSamples24k != null && _referenceSamples24k.Length > 0)
            {
                var bytes = WavCodec.Encode(_referenceSamples24k, QwenTts.NativeSampleRate);
                await WriteAllBytesAsync(Path.Combine(folder, VoiceManifest.ReferenceFileName), bytes)
                    .ConfigureAwait(false);
                wroteReference = true;
            }

            if (IsCloned && _prompt.IsValid)
            {
                var path = Path.Combine(folder, ClonePromptFile.FileName);
                await Task.Run(() => ClonePromptFile.Write(path, _prompt)).ConfigureAwait(false);
            }

            var manifest = new VoiceManifest
            {
                IsClone = IsCloned,
                Instruct = Instruct,
                ReferenceText = ReferenceText,
                Language = Language,
                HasSample = wroteSample,
                HasReference = wroteReference,
                ReferenceSourceSampleRate = _referenceSourceRate,
            };
            await manifest.WriteAsync(folder).ConfigureAwait(false);
        }

        static Task WriteAllBytesAsync(string path, byte[] bytes)
        {
#if UNITY_2021_2_OR_NEWER
            return File.WriteAllBytesAsync(path, bytes);
#else
            return Task.Run(() => File.WriteAllBytes(path, bytes));
#endif
        }
    }
}
