// One engine, two independent checkpoints. VoiceDesign samples a speaker from
// an instruct; Base clones one from a reference recording using the official
// in-context path (reference codes + reference text + speaker embedding).
//
// The halves are deliberately symmetric and separately lockable: each is ~13 GB
// resident, they are needed in different phases of a session, and a host that
// can load one and evict the other is the difference between a 32 GB and a
// 64 GB requirement.

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.ML.OnnxRuntime;
using QwenTTS.Audio;
using QwenTTS.Internal;
using QwenTTS.Onnx;
using UnityEngine;

namespace QwenTTS.Engine
{
    internal sealed class QwenTtsEngine : IDisposable
    {
        public const int NativeSampleRate = 24000;

        readonly ExecutionProvider _ep;
        readonly Half _voiceDesign;
        readonly Half _clone;
        bool _disposed;

        /// <summary>
        /// Everything belonging to one checkpoint, behind one lock. Two halves
        /// can generate at the same time; two calls into the same half cannot,
        /// because the talker reuses its KV and sampler buffers.
        /// </summary>
        sealed class Half : IDisposable
        {
            public readonly QwenCheckpoint Checkpoint;
            public readonly object Gate = new object();

            public TextTokenizer Tokenizer;
            public EmbeddingStore Embeddings;
            public LanguageModel Talker;
            public QwenVocoderModel Vocoder;
            public QwenSpeakerEncoderModel SpeakerEncoder;      // Base only
            public QwenTokenizerEncoderModel TokenizerEncoder;  // Base only

            public Half(QwenCheckpoint checkpoint) => Checkpoint = checkpoint;

            public bool IsLoaded => Talker != null;

            public void Dispose()
            {
                Tokenizer?.Dispose();
                Embeddings?.Dispose();
                Talker?.Dispose();
                Vocoder?.Dispose();
                SpeakerEncoder?.Dispose();
                TokenizerEncoder?.Dispose();
                Tokenizer = null;
                Embeddings = null;
                Talker = null;
                Vocoder = null;
                SpeakerEncoder = null;
                TokenizerEncoder = null;
            }
        }

        public QwenTtsEngine(ExecutionProvider executionProvider = ExecutionProvider.CPU)
        {
            _ep = executionProvider;
            _voiceDesign = new Half(QwenCheckpoint.VoiceDesign);
            _clone = new Half(QwenCheckpoint.Base);
        }

        Half HalfFor(QwenCheckpoint checkpoint) =>
            checkpoint == QwenCheckpoint.Base ? _clone : _voiceDesign;

        public bool IsLoaded(QwenCheckpoint checkpoint) => HalfFor(checkpoint).IsLoaded;

        /// <summary>True when the Base checkpoint's codec encoder is installed.</summary>
        public bool HasIclEncoder => _clone.TokenizerEncoder != null;

        #region Load / evict

        public void EnsureLoaded(QwenCheckpoint checkpoint)
        {
            ThrowIfDisposed();
            var half = HalfFor(checkpoint);
            lock (half.Gate)
                Load(half);
        }

        void Load(Half half)
        {
            if (half.IsLoaded)
                return;
            if (!QwenModelPaths.IsPresent(half.Checkpoint))
            {
                var missing = QwenModelPaths.MissingFiles(half.Checkpoint);
                throw new InvalidOperationException(
                    $"{half.Checkpoint} weights are not installed at " +
                    $"{QwenModelPaths.DirectoryFor(half.Checkpoint)} " +
                    $"(missing {missing.Count} file(s), first: {missing[0]}).");
            }

            var sw = Stopwatch.StartNew();
            var dir = QwenModelPaths.DirectoryFor(half.Checkpoint);
            var embeddingsDir = QwenModelPaths.EmbeddingsDir(half.Checkpoint);
            var configPath = Path.Combine(embeddingsDir, "config.json");

            half.Tokenizer = new TextTokenizer(QwenModelPaths.TokenizerDir(half.Checkpoint));

            // Embedding tables are re-read rather than stashed across domain
            // reload: measured 0.5-0.8 s, against ~22 s for the ONNX sessions,
            // which are what the editor keep-alive actually protects.
            half.Embeddings = new EmbeddingStore(embeddingsDir, configPath);
            QwenLog.Log($"[QwenTtsEngine] {half.Checkpoint} embeddings from {dir} " +
                        $"in {sw.ElapsedMilliseconds}ms");

            half.Talker = new LanguageModel(half.Embeddings, half.Checkpoint, _ep);
            half.Vocoder = new QwenVocoderModel(half.Checkpoint, timeMajor: false, _ep);

            if (half.Checkpoint == QwenCheckpoint.Base)
            {
                half.SpeakerEncoder = new QwenSpeakerEncoderModel(_ep);
                half.TokenizerEncoder = new QwenTokenizerEncoderModel(_ep);
            }
        }

        /// <summary>
        /// Drops one checkpoint's graphs and embedding tables. The other half
        /// keeps running. Safe to call when it was never loaded.
        /// </summary>
        public void Evict(QwenCheckpoint checkpoint)
        {
            if (_disposed)
                return;
            var half = HalfFor(checkpoint);
            lock (half.Gate)
            {
                if (!half.IsLoaded && half.Embeddings == null)
                    return;
                half.Dispose();
                QwenLog.Log($"[QwenTtsEngine] Evicted {checkpoint}");
            }
        }

        /// <summary>Opens the ONNX sessions for a checkpoint on a worker thread.</summary>
        public Task PreloadAsync(QwenCheckpoint checkpoint, CancellationToken cancellationToken = default)
        {
            ThrowIfDisposed();
            var half = HalfFor(checkpoint);
            return BackgroundWork.Run(() =>
            {
                lock (half.Gate)
                {
                    cancellationToken.ThrowIfCancellationRequested();
                    Load(half);
                    half.Talker.PreloadSessions();
                    half.Vocoder.GetSession();
                    half.SpeakerEncoder?.GetSession();
                    half.TokenizerEncoder?.GetSession();
                }
            });
        }

        #endregion

        #region VoiceDesign

        /// <summary>
        /// Speech from a natural-language instruct. The instruct *is* the
        /// speaker, and a new one is sampled every call.
        /// </summary>
        public float[] SynthesizeDesigned(string text, string instruct, string language,
            SamplingParams sampling, IProgress<SpeechProgress> progress = null,
            StreamRequest stream = default,
            CancellationToken cancellationToken = default)
        {
            ThrowIfDisposed();
            RequireText(text);

            lock (_voiceDesign.Gate)
            {
                Load(_voiceDesign);
                cancellationToken.ThrowIfCancellationRequested();

                var assistantIds = _voiceDesign.Tokenizer.BuildAssistantPrompt(text);
                var instructIds = _voiceDesign.Tokenizer.BuildInstructTokens(instruct);
                QwenLog.LogVerbose(
                    $"[QwenTtsEngine] VoiceDesign tokens assistant={assistantIds.Length} " +
                    $"instruct={instructIds.Length}");

                long[,,] codes;
                using (stream.Begin(_voiceDesign.Talker, _voiceDesign.Vocoder,
                    prefixCodes: null, cancellationToken))
                {
                    codes = _voiceDesign.Talker.GenerateVoiceDesign(
                        assistantIds, instructIds, language, sampling, progress, cancellationToken);
                }
                var pcm = _voiceDesign.Vocoder.Decode(codes, cancellationToken);
                QwenLog.Log($"[QwenTtsEngine] VoiceDesign codes T={codes.GetLength(2)} wav={pcm.Length} @24k");
                return pcm;
            }
        }

        #endregion

        #region Base clone

        /// <summary>
        /// The reusable part of a clone: the 2048-d speaker vector and, for
        /// in-context cloning, the codec frames of the reference. Both are pure
        /// functions of the reference audio, so a host should persist them
        /// rather than re-derive them on every load.
        /// </summary>
        public ClonePrompt ExtractClonePrompt(float[] samples24k, bool withIclCodes,
            CancellationToken cancellationToken = default)
        {
            ThrowIfDisposed();
            if (samples24k == null || samples24k.Length == 0)
                throw new ArgumentException("Reference audio is empty.", nameof(samples24k));

            lock (_clone.Gate)
            {
                Load(_clone);
                cancellationToken.ThrowIfCancellationRequested();

                var embedding = _clone.SpeakerEncoder.Encode(samples24k);
                if (embedding.Length == 0)
                    throw new InvalidOperationException("speaker_encoder.onnx returned an empty embedding.");

                long[,,] codes = null;
                if (withIclCodes)
                {
                    cancellationToken.ThrowIfCancellationRequested();
                    codes = _clone.TokenizerEncoder.Encode(samples24k);
                }
                return new ClonePrompt(embedding, codes);
            }
        }

        public float[] SynthesizeCloned(string text, ClonePrompt prompt, string refText, string language,
            SamplingParams sampling, IProgress<SpeechProgress> progress = null,
            StreamRequest stream = default,
            CancellationToken cancellationToken = default)
        {
            ThrowIfDisposed();
            RequireText(text);
            if (!prompt.IsValid)
                throw new ArgumentException("Clone prompt is empty.", nameof(prompt));

            lock (_clone.Gate)
            {
                Load(_clone);
                cancellationToken.ThrowIfCancellationRequested();

                var tokenIds = _clone.Tokenizer.BuildClonePrompt(text, speaker: null, language, instruct: null);
                if (tokenIds.Length < 8)
                    throw new InvalidOperationException("Prompt tokenization produced too few tokens.");

                bool icl = prompt.HasIclCodes && !string.IsNullOrEmpty(refText);
                long[,,] codes;
                // Streaming the ICL path has to carry the reference frames into
                // every decode, for the same reason the finished utterance does.
                using (stream.Begin(_clone.Talker, _clone.Vocoder,
                    icl ? prompt.ReferenceCodes : null, cancellationToken))
                {
                    if (icl)
                    {
                        var refTokenIds = _clone.Tokenizer.BuildIclRefTextTokens(refText);
                        codes = _clone.Talker.GenerateWithSpeakerEmbeddingAndRefText(
                            tokenIds, prompt.SpeakerEmbedding, language, refTokenIds, prompt.ReferenceCodes,
                            sampling, progress, cancellationToken);
                        QwenLog.Log($"[QwenTtsEngine] Base ICL clone codes T={codes.GetLength(2)} " +
                                    $"refT={prompt.ReferenceFrames}");
                    }
                    else
                    {
                        codes = _clone.Talker.GenerateWithSpeakerEmbedding(
                            tokenIds, prompt.SpeakerEmbedding, language, sampling, progress, cancellationToken);
                        QwenLog.Log($"[QwenTtsEngine] Base x-vector clone codes T={codes.GetLength(2)}");
                    }
                }

                if (!icl)
                {
                    var plain = _clone.Vocoder.Decode(codes, cancellationToken);
                    QwenLog.Log($"[QwenTtsEngine] Base clone wav={plain.Length} @24k icl=False");
                    return plain;
                }

                // The generated codes continue the reference, so the codec has
                // to decode both together and the leading reference is then
                // dropped by frame proportion. Decoding the tail alone starts
                // the decoder cold and the first word comes out as a different
                // voice. This mirrors Qwen's generate_voice_clone.
                int refFrames = prompt.ReferenceFrames;
                var joined = PrependReferenceCodes(prompt.ReferenceCodes, codes);
                int totalFrames = joined.GetLength(2);
                var full = _clone.Vocoder.Decode(joined, cancellationToken);
                int cut = (int)((long)refFrames * full.Length / Math.Max(totalFrames, 1));
                if (cut >= full.Length)
                    cut = 0;
                var pcm = new float[full.Length - cut];
                Array.Copy(full, cut, pcm, 0, pcm.Length);
                QwenLog.Log($"[QwenTtsEngine] Base clone wav={pcm.Length} @24k icl=True " +
                            $"(decoded {totalFrames} frames, dropped {cut} ref samples)");
                return pcm;
            }
        }

        /// <summary>
        /// Reference frames (1, T, 16) ahead of generated codes (1, 16, T'), as
        /// one quantizer-major block for the codec.
        /// </summary>
        static long[,,] PrependReferenceCodes(long[,,] refTimeMajor, long[,,] generated)
        {
            int quantizers = generated.GetLength(1);
            int refFrames = refTimeMajor.GetLength(1);
            int genFrames = generated.GetLength(2);
            var joined = new long[1, quantizers, refFrames + genFrames];
            for (int q = 0; q < quantizers; q++)
            {
                for (int t = 0; t < refFrames; t++)
                    joined[0, q, t] = refTimeMajor[0, t, q];
                for (int t = 0; t < genFrames; t++)
                    joined[0, q, refFrames + t] = generated[0, q, t];
            }
            return joined;
        }

        #endregion

        #region Audio helpers

        /// <summary>
        /// Mono 24 kHz float PCM from a clip. Must run on the main thread —
        /// <c>AudioClip.GetData</c> is a Unity API.
        /// </summary>
        public static float[] ClipToMono24k(AudioClip clip)
        {
            if (clip == null)
                throw new ArgumentNullException(nameof(clip));

            var raw = new float[clip.samples * clip.channels];
            clip.GetData(raw, 0);

            float[] mono;
            if (clip.channels <= 1)
            {
                mono = raw;
            }
            else
            {
                mono = new float[clip.samples];
                for (int i = 0; i < clip.samples; i++)
                {
                    float sum = 0f;
                    for (int c = 0; c < clip.channels; c++)
                        sum += raw[i * clip.channels + c];
                    mono[i] = sum / clip.channels;
                }
            }

            if (clip.frequency == NativeSampleRate)
                return mono;
            return AudioResample.Resample(mono, clip.frequency, NativeSampleRate);
        }

        #endregion

        void RequireText(string text)
        {
            if (string.IsNullOrEmpty(text))
                throw new ArgumentException("Text cannot be empty.", nameof(text));
            if (text.Length > 10000)
                throw new ArgumentException("Text exceeds maximum length of 10,000 characters.", nameof(text));
        }

        void ThrowIfDisposed()
        {
            if (_disposed)
                throw new ObjectDisposedException(nameof(QwenTtsEngine));
        }

        public void Dispose()
        {
            if (_disposed)
                return;
            _disposed = true;
            lock (_voiceDesign.Gate)
                _voiceDesign.Dispose();
            lock (_clone.Gate)
                _clone.Dispose();
        }

        #region Editor keep-alive support

        internal void CollectOnnxModels(List<ORTModel> list)
        {
            CollectFrom(_voiceDesign, list);
            CollectFrom(_clone, list);
        }

        static void CollectFrom(Half half, List<ORTModel> list)
        {
            lock (half.Gate)
            {
                half.Talker?.CollectOnnxModels(list);
                if (half.Vocoder != null)
                    list.Add(half.Vocoder);
                if (half.SpeakerEncoder != null)
                    list.Add(half.SpeakerEncoder);
                if (half.TokenizerEncoder != null)
                    list.Add(half.TokenizerEncoder);
            }
        }

        internal void AdoptNativeSessions(Dictionary<string, InferenceSession> sessions)
        {
            if (sessions == null || sessions.Count == 0)
                return;
            var models = new List<ORTModel>();
            CollectOnnxModels(models);
            foreach (var model in models)
            {
                if (sessions.TryGetValue(model.SessionKeepAliveKey, out var session))
                {
                    model.AdoptSession(session);
                    sessions.Remove(model.SessionKeepAliveKey);
                    QwenLog.Log("[QwenTtsEngine] Adopted " + model.SessionKeepAliveKey);
                }
            }
        }

        #endregion
    }

    /// <summary>
    /// The speaker identity extracted from a reference recording: an x-vector
    /// always, plus the reference codec frames when in-context cloning is used.
    /// </summary>
    internal readonly struct ClonePrompt
    {
        public readonly float[] SpeakerEmbedding;
        public readonly long[,,] ReferenceCodes;

        public ClonePrompt(float[] speakerEmbedding, long[,,] referenceCodes)
        {
            SpeakerEmbedding = speakerEmbedding;
            ReferenceCodes = referenceCodes;
        }

        public bool IsValid => SpeakerEmbedding != null && SpeakerEmbedding.Length > 0;

        public bool HasIclCodes => ReferenceCodes != null && ReferenceCodes.GetLength(1) > 0;

        public int ReferenceFrames => ReferenceCodes?.GetLength(1) ?? 0;
    }
}
