using System;
using System.Collections.Generic;
using System.IO;
using System.Threading;
using System.Threading.Tasks;
using QwenTTS.Audio;
using QwenTTS.Engine;
using QwenTTS.Internal;
using UnityEngine;

namespace QwenTTS
{
    /// <summary>
    /// Entry point for Qwen3-TTS. Two checkpoints, both optional:
    /// <see cref="QwenCheckpoint.VoiceDesign"/> invents a speaker from a
    /// natural-language instruct, <see cref="QwenCheckpoint.Base"/> clones one
    /// from a reference recording.
    ///
    /// Each checkpoint is roughly 13 GB on disk and resident when loaded, so
    /// residency is explicit: nothing loads until a call needs it, and
    /// <see cref="Evict"/> releases one without touching the other. Hosts that
    /// design a voice and then clone it — the usual flow — should evict
    /// VoiceDesign once a take is locked.
    ///
    /// This package never downloads weights. Point
    /// <see cref="QwenTtsSettings.ModelRoot"/> at an install and check
    /// <see cref="GetStatus"/>.
    /// </summary>
    public static class QwenTts
    {
        /// <summary>Rate the codec runs at. Ask for this to avoid a resample.</summary>
        public const int NativeSampleRate = QwenTtsEngine.NativeSampleRate;

        /// <summary>
        /// Below this a reference carries too few 12 Hz frames to pin a
        /// speaker, and clones of it vary noticeably between utterances.
        /// </summary>
        public const float MinRecommendedReferenceSeconds = 4f;

        static readonly object Gate = new object();
        static QwenTtsSettings _settings;
        static QwenTtsEngine _engine;
        static Task<QwenTtsEngine> _engineTask;

        public static bool IsInitialized { get; private set; }

        /// <summary>
        /// Applies settings. Cheap: no weights are touched here. Call again to
        /// change settings, which evicts anything already loaded because the
        /// model root and execution provider are baked into the sessions.
        /// </summary>
        public static void Initialize(QwenTtsSettings settings = null)
        {
            settings ??= new QwenTtsSettings();
            lock (Gate)
            {
                bool rebind = IsInitialized &&
                              (_settings.ExecutionProvider != settings.ExecutionProvider ||
                               _settings.ModelRoot != settings.ModelRoot);
                if (rebind)
                    UnloadInternal();

                _settings = settings;
                QwenLog.LogLevel = settings.LogLevel;
                QwenModelPaths.Root = settings.ModelRoot;
                // Order matters: memory usage decides each model's load
                // policy at construction, so it must be set before any
                // session wrapper exists.
                Onnx.ORTModel.SetMemoryUsage(settings.MemoryUsage);
                Onnx.ORTModel.InitializeEnvironment(settings.LogLevel);
                IsInitialized = true;

                QwenLog.Log(
                    $"[QwenTTS] Initialized (root {QwenModelPaths.Root}, EP {settings.ExecutionProvider}, " +
                    $"memory {settings.MemoryUsage}). " +
                    $"VoiceDesign installed={QwenModelPaths.IsPresent(QwenCheckpoint.VoiceDesign)}, " +
                    $"Base installed={QwenModelPaths.IsPresent(QwenCheckpoint.Base)}");
            }
        }

        /// <summary>Install and residency state, without loading anything.</summary>
        public static CheckpointStatus GetStatus(QwenCheckpoint checkpoint)
        {
            var missing = QwenModelPaths.MissingFiles(checkpoint);
            return new CheckpointStatus
            {
                Checkpoint = checkpoint,
                Installed = missing.Count == 0,
                Loaded = _engine != null && _engine.IsLoaded(checkpoint),
                Directory = QwenModelPaths.DirectoryFor(checkpoint),
                MissingFiles = missing,
                InstalledBytes = QwenModelPaths.InstalledBytes(checkpoint),
            };
        }

        public static bool IsLoaded(QwenCheckpoint checkpoint) =>
            _engine != null && _engine.IsLoaded(checkpoint);

        /// <summary>
        /// Opens a checkpoint's tables and ONNX sessions now, off the calling
        /// thread. Call this from a loading screen: the talker graphs alone are
        /// ~10 s to open warm and around twice that cold, and without a warm-up
        /// the first utterance pays all of it.
        /// </summary>
        public static async Task WarmUpAsync(QwenCheckpoint checkpoint,
            CancellationToken cancellationToken = default)
        {
            var engine = await GetEngineAsync().ConfigureAwait(false);
            await engine.PreloadAsync(checkpoint, cancellationToken).ConfigureAwait(false);
        }

        /// <summary>
        /// Releases one checkpoint's graphs and embedding tables. Voices that
        /// were created from it keep working — the next call reloads it — so
        /// this is safe to do between phases of a session.
        /// </summary>
        public static void Evict(QwenCheckpoint checkpoint) => _engine?.Evict(checkpoint);

        /// <summary>Releases both checkpoints but stays initialized.</summary>
        public static void EvictAll()
        {
            _engine?.Evict(QwenCheckpoint.VoiceDesign);
            _engine?.Evict(QwenCheckpoint.Base);
        }

        /// <summary>
        /// Drops the engine entirely. Existing <see cref="QwenVoice"/> handles
        /// become unusable. Stays initialized, so a later create rebuilds.
        /// </summary>
        public static void Unload()
        {
            lock (Gate)
                UnloadInternal();
            QwenLog.Log("[QwenTTS] Unloaded");
        }

        static void UnloadInternal()
        {
            _engine?.Dispose();
            _engine = null;
            _engineTask = null;
            KeepAliveHandoff.Clear();
        }

        /// <summary>Unload and forget the settings.</summary>
        public static void Shutdown()
        {
            lock (Gate)
            {
                UnloadInternal();
                IsInitialized = false;
                _settings = null;
            }
        }

        #region Voice creation

        /// <summary>
        /// A voice invented from an instruct. Every call samples a different
        /// person, even with the same instruct — that is what the checkpoint
        /// does. To keep one, render a take and pass it to
        /// <see cref="CreateClonedVoiceAsync"/>.
        /// </summary>
        public static async Task<QwenVoice> CreateDesignedVoiceAsync(VoiceDesignSpec spec,
            CancellationToken cancellationToken = default)
        {
            RequireInitialized();
            if (spec == null)
                throw new ArgumentNullException(nameof(spec));
            if (string.IsNullOrWhiteSpace(spec.Instruct))
                throw new ArgumentException(
                    "VoiceDesign needs an instruct — it is the voice. " +
                    "Describe the speaker in natural language.", nameof(spec));
            RequireLanguage(spec.Language);

            var engine = await GetEngineAsync().ConfigureAwait(false);
            return QwenVoice.Designed(engine, spec.Instruct.Trim(), spec.Language);
        }

        /// <summary>
        /// A voice cloned from a reference recording, using the in-context path
        /// when <paramref name="referenceText"/> is supplied — which is what
        /// Qwen does by default and what actually reproduces the reference
        /// speaker. Without the transcript this degrades to x-vector-only,
        /// which is a stable voice but not the one in the recording.
        /// </summary>
        /// <param name="reference">
        /// Reference audio. Best at <see cref="NativeSampleRate"/>: the speaker
        /// encoder reads mel up to 12 kHz, so a 16 kHz reference has the top of
        /// that band missing. At least
        /// <see cref="MinRecommendedReferenceSeconds"/> seconds.
        /// </param>
        /// <param name="referenceText">Exact transcript of the reference audio.</param>
        public static async Task<QwenVoice> CreateClonedVoiceAsync(AudioClip reference,
            string referenceText, string language = null,
            CancellationToken cancellationToken = default)
        {
            RequireInitialized();
            if (reference == null)
                throw new ArgumentNullException(nameof(reference));
            language = string.IsNullOrWhiteSpace(language) ? QwenLanguages.Default : language;
            RequireLanguage(language);

            WarnAboutReference(reference.frequency, reference.length);

            // AudioClip.GetData is main-thread only, so the samples come out
            // here and everything expensive happens on the pool.
            float[] samples = QwenTtsEngine.ClipToMono24k(reference);
            var engine = await GetEngineAsync().ConfigureAwait(false);

            bool icl = !string.IsNullOrWhiteSpace(referenceText);
            if (!icl)
            {
                QwenLog.LogWarning(
                    "[QwenTTS] Cloning without a reference transcript — x-vector only. " +
                    "Pass the transcript to use the in-context path.");
            }

            var prompt = await BackgroundWork.Run(
                () => engine.ExtractClonePrompt(samples, icl, cancellationToken)).ConfigureAwait(false);

            QwenLog.Log($"[QwenTTS] Clone prompt ready (x-vector dim={prompt.SpeakerEmbedding.Length}, " +
                        $"ref frames={prompt.ReferenceFrames}, {reference.length:0.00}s reference)");

            return QwenVoice.Cloned(engine, prompt, referenceText, language, samples, reference.frequency);
        }

        /// <summary>
        /// Reloads a voice previously written by <see cref="QwenVoice.SaveAsync"/>.
        /// A cloned voice restores its stored prompt if the file is current,
        /// and only falls back to re-running the encoders when it is not.
        /// </summary>
        public static async Task<QwenVoice> LoadVoiceAsync(string folder,
            CancellationToken cancellationToken = default)
        {
            RequireInitialized();
            if (string.IsNullOrWhiteSpace(folder) || !Directory.Exists(folder))
                throw new DirectoryNotFoundException("Voice folder not found: " + folder);

            var manifest = VoiceManifest.Read(folder);
            var engine = await GetEngineAsync().ConfigureAwait(false);

            if (!manifest.IsClone)
                return QwenVoice.Designed(engine, manifest.Instruct, manifest.Language);

            if (ClonePromptFile.TryRead(Path.Combine(folder, ClonePromptFile.FileName), out var stored))
            {
                QwenLog.LogVerbose($"[QwenTTS] Restored clone prompt from {folder} " +
                                   $"(ref frames={stored.ReferenceFrames})");
                return QwenVoice.Cloned(engine, stored, manifest.ReferenceText, manifest.Language, null, 0);
            }

            string referencePath = Path.Combine(folder, VoiceManifest.ReferenceFileName);
            if (!File.Exists(referencePath))
            {
                throw new FileNotFoundException(
                    "Cloned voice has neither a stored prompt nor its reference audio: " + folder);
            }

            QwenLog.Log($"[QwenTTS] No usable stored prompt in {folder}; re-deriving from the reference.");
            var bytes = await Task.Run(() => File.ReadAllBytes(referencePath), cancellationToken)
                .ConfigureAwait(false);
            if (!WavCodec.TryDecode(bytes, out float[] pcm, out int rate, out int channels))
                throw new InvalidDataException("Could not decode " + referencePath);

            float[] samples = WavCodec.ToMono24k(pcm, rate, channels);
            bool icl = !string.IsNullOrWhiteSpace(manifest.ReferenceText);
            var prompt = await BackgroundWork.Run(
                () => engine.ExtractClonePrompt(samples, icl, cancellationToken)).ConfigureAwait(false);
            return QwenVoice.Cloned(engine, prompt, manifest.ReferenceText, manifest.Language, samples, rate);
        }

        #endregion

        internal static MemoryUsage CurrentMemoryUsage => _settings?.MemoryUsage ?? MemoryUsage.Balanced;

        internal static bool LogTiming => _settings?.LogTiming ?? false;

        static void WarnAboutReference(int frequency, float seconds)
        {
            if (frequency > 0 && frequency < NativeSampleRate)
            {
                QwenLog.LogWarning(
                    $"[QwenTTS] Clone reference is {frequency} Hz. The speaker encoder reads mel to " +
                    $"12 kHz, so below {NativeSampleRate} Hz the top of the band that identifies the " +
                    "speaker is already gone and the clone will not match the recording.");
            }
            if (seconds > 0f && seconds < MinRecommendedReferenceSeconds)
            {
                QwenLog.LogWarning(
                    $"[QwenTTS] Clone reference is {seconds:0.00}s. In-context cloning conditions on the " +
                    $"reference codes, so under {MinRecommendedReferenceSeconds:0}s the speaker is weakly " +
                    "determined and takes vary between utterances.");
            }
        }

        static void RequireInitialized()
        {
            if (!IsInitialized)
                throw new InvalidOperationException("Call QwenTts.Initialize first.");
        }

        static void RequireLanguage(string language)
        {
            if (!QwenLanguages.IsSupported(language))
            {
                throw new ArgumentException(
                    $"Unsupported language '{language}'. Supported: " +
                    string.Join(", ", QwenLanguages.All) + ", " + QwenLanguages.Auto);
            }
        }

        static Task<QwenTtsEngine> GetEngineAsync()
        {
            lock (Gate)
            {
                if (_engine != null)
                    return Task.FromResult(_engine);
                if (_engineTask != null && !_engineTask.IsFaulted && !_engineTask.IsCanceled)
                    return _engineTask;

                var ep = _settings.ExecutionProvider;
                _engineTask = BackgroundWork.Run(() =>
                {
                    var engine = new QwenTtsEngine(ep);
                    var recovered = KeepAliveHandoff.TakeSessions();
                    if (recovered != null)
                        engine.AdoptNativeSessions(recovered);
                    _engine = engine;
                    return engine;
                });
                return _engineTask;
            }
        }

        /// <summary>Engine handle for the editor keep-alive. Null when nothing is loaded.</summary>
        internal static QwenTtsEngine EngineOrNull => _engine;
    }
}
