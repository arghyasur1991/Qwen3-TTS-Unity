using System;
using System.IO;
using QwenTTS.Engine;

namespace QwenTTS.Internal
{
    /// <summary>
    /// On-disk form of a clone prompt.
    ///
    /// The speaker embedding and the reference codec frames are pure functions of the
    /// reference wav, but deriving them costs a speaker-encoder run plus a
    /// tokenizer-encoder run behind a ~370 MB session. Persisting them turns
    /// every subsequent voice load into a file read.
    ///
    /// Versioned, because a change to the encoder export invalidates the
    /// contents — on a version or shape mismatch the caller re-derives from
    /// the reference wav instead of failing.
    /// </summary>
    internal static class ClonePromptFile
    {
        public const string FileName = "clone_prompt.bin";

        const int Magic = 0x51435031; // QCP1
        const int Version = 1;

        public static void Write(string path, ClonePrompt prompt)
        {
            if (!prompt.IsValid)
                throw new ArgumentException("Clone prompt is empty.", nameof(prompt));

            using var stream = new FileStream(path, FileMode.Create, FileAccess.Write);
            using var w = new BinaryWriter(stream);
            w.Write(Magic);
            w.Write(Version);

            w.Write(prompt.SpeakerEmbedding.Length);
            for (int i = 0; i < prompt.SpeakerEmbedding.Length; i++)
                w.Write(prompt.SpeakerEmbedding[i]);

            int frames = prompt.ReferenceFrames;
            int quantizers = prompt.HasIclCodes ? prompt.ReferenceCodes.GetLength(2) : 0;
            w.Write(frames);
            w.Write(quantizers);
            for (int t = 0; t < frames; t++)
                for (int q = 0; q < quantizers; q++)
                    w.Write(prompt.ReferenceCodes[0, t, q]);
        }

        /// <summary>False when the file is absent, stale or malformed; re-derive in that case.</summary>
        public static bool TryRead(string path, out ClonePrompt prompt)
        {
            prompt = default;
            if (!File.Exists(path))
                return false;

            try
            {
                using var stream = new FileStream(path, FileMode.Open, FileAccess.Read);
                using var r = new BinaryReader(stream);
                if (r.ReadInt32() != Magic || r.ReadInt32() != Version)
                    return false;

                int embLen = r.ReadInt32();
                if (embLen <= 0 || embLen > 1 << 16)
                    return false;
                var embedding = new float[embLen];
                for (int i = 0; i < embLen; i++)
                    embedding[i] = r.ReadSingle();

                int frames = r.ReadInt32();
                int quantizers = r.ReadInt32();
                if (frames < 0 || quantizers < 0 || frames > 1 << 20 || quantizers > 64)
                    return false;

                long[,,] codes = null;
                if (frames > 0 && quantizers > 0)
                {
                    codes = new long[1, frames, quantizers];
                    for (int t = 0; t < frames; t++)
                        for (int q = 0; q < quantizers; q++)
                            codes[0, t, q] = r.ReadInt64();
                }

                prompt = new ClonePrompt(embedding, codes);
                return true;
            }
            catch (Exception e)
            {
                QwenLog.LogWarning($"[QwenTTS] Ignoring unreadable clone prompt at {path}: {e.Message}");
                return false;
            }
        }
    }
}
