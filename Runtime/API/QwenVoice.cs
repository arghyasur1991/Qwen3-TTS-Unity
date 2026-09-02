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
        public async Task<SpeechResult> SpeakAsync(string text, SpeechOptions options = null,
            IProgress<SpeechProgress> progress = null, CancellationToken cancellationToken = default)
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

            float[] pcm24 = await BackgroundWork.Run(() => cloned
                ? engine.SynthesizeCloned(text, prompt, referenceText, language, sampling, progress, cancellationToken)
                : engine.SynthesizeDesigned(text, instruct, language, sampling, progress, cancellationToken))
                .ConfigureAwait(false);

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
