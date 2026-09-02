using System;
using System.Collections.Generic;
using System.IO;
using UnityEngine;

namespace QwenTTS.Engine
{
    /// <summary>
    /// Where the exported Qwen3-TTS graphs live, and whether a checkpoint is
    /// complete. This package never downloads weights.
    ///
    /// The root defaults to <c>StreamingAssets/QwenTTS</c> but is settable —
    /// a shipped game usually keeps 13+ GB of weights beside the player, in
    /// DLC, or in <see cref="Application.persistentDataPath"/> after a
    /// post-install download, none of which are StreamingAssets.
    /// </summary>
    public static class QwenModelPaths
    {
        public const string StreamingAssetsSubfolder = "QwenTTS";

        /// <summary>Folder name for the VoiceDesign checkpoint.</summary>
        public const string VoiceDesignFolderName = "Qwen3-1.7B-VoiceDesign";

        /// <summary>Folder name for the Base (voice clone) checkpoint.</summary>
        public const string BaseFolderName = "Qwen3-1.7B-Base";

        // Graph file stems, shared by both checkpoints unless noted.
        /// <summary>
        /// One graph serving both phases: a zero-length past makes it a
        /// prefill. Halves the talker weights, which are otherwise the same
        /// 1.7B exported twice and both resident for a single utterance.
        /// </summary>
        public const string GraphTalker = "talker";

        // Superseded by GraphTalker; still read when only these are installed.
        public const string GraphTalkerPrefill = "talker_prefill";
        public const string GraphTalkerDecode = "talker_decode";
        public const string GraphCodePredictor = "code_predictor";
        public const string GraphVocoder = "vocoder";
        public const string GraphSpeakerEncoder = "speaker_encoder";    // Base only
        public const string GraphTokenizerEncoder = "tokenizer_encoder"; // Base only

        static string _root;

        /// <summary>
        /// Root folder holding the checkpoint subfolders. Set before
        /// <c>QwenTts.Initialize</c>; null or empty restores the
        /// StreamingAssets default.
        /// </summary>
        public static string Root
        {
            get => string.IsNullOrEmpty(_root)
                ? Path.Combine(Application.streamingAssetsPath, StreamingAssetsSubfolder)
                : _root;
            set => _root = value;
        }

        /// <summary>True when <see cref="Root"/> has been pointed somewhere explicit.</summary>
        public static bool RootIsExplicit => !string.IsNullOrEmpty(_root);

        public static string FolderName(QwenCheckpoint checkpoint) => checkpoint switch
        {
            QwenCheckpoint.VoiceDesign => VoiceDesignFolderName,
            QwenCheckpoint.Base => BaseFolderName,
            _ => throw new ArgumentOutOfRangeException(nameof(checkpoint)),
        };

        public static string DirectoryFor(QwenCheckpoint checkpoint) =>
            Path.Combine(Root, FolderName(checkpoint));

        public static string GraphPath(QwenCheckpoint checkpoint, string stem) =>
            Path.Combine(DirectoryFor(checkpoint), stem + ".onnx");

        public static string EmbeddingsDir(QwenCheckpoint checkpoint) =>
            Path.Combine(DirectoryFor(checkpoint), "embeddings");

        public static string TokenizerDir(QwenCheckpoint checkpoint) =>
            Path.Combine(DirectoryFor(checkpoint), "tokenizer");

        /// <summary>
        /// Files the engine actually opens. Deliberately not "everything the
        /// exporter wrote" — <c>codec_head_weight.npy</c> for instance is
        /// produced but never read, and listing it would fail installs that
        /// are in fact complete.
        /// </summary>
        /// <summary>
        /// True when the checkpoint has the single unified talker. Exports made
        /// before it existed have the prefill/decode pair instead, and both
        /// layouts are supported so an install does not have to be redone.
        /// </summary>
        public static bool HasUnifiedTalker(QwenCheckpoint checkpoint)
        {
            var dir = DirectoryFor(checkpoint);
            return File.Exists(Path.Combine(dir, GraphTalker + ".onnx"))
                && File.Exists(Path.Combine(dir, GraphTalker + ".onnx.data"));
        }

        public static IReadOnlyList<string> ExpectedFiles(QwenCheckpoint checkpoint)
        {
            var files = new List<string>
            {
                GraphCodePredictor + ".onnx", GraphCodePredictor + ".onnx.data",
                GraphVocoder + ".onnx", GraphVocoder + ".onnx.data",
                "embeddings/config.json",
                "embeddings/talker_codec_embedding.npy",
                "embeddings/text_embedding.npy",
                "embeddings/text_projection_fc1_weight.npy",
                "embeddings/text_projection_fc1_bias.npy",
                "embeddings/text_projection_fc2_weight.npy",
                "embeddings/text_projection_fc2_bias.npy",
                "tokenizer/vocab.json",
                "tokenizer/merges.txt",
            };
            for (int i = 0; i < EmbeddingStore.CpGroupCount; i++)
                files.Add($"embeddings/cp_codec_embedding_{i}.npy");
            files.Add("embeddings/cp_projection_weight.npy");
            files.Add("embeddings/cp_projection_bias.npy");

            if (HasUnifiedTalker(checkpoint))
            {
                files.Add(GraphTalker + ".onnx");
                files.Add(GraphTalker + ".onnx.data");
            }
            else
            {
                files.Add(GraphTalkerPrefill + ".onnx");
                files.Add(GraphTalkerPrefill + ".onnx.data");
                files.Add(GraphTalkerDecode + ".onnx");
                files.Add(GraphTalkerDecode + ".onnx.data");
            }

            if (checkpoint == QwenCheckpoint.Base)
            {
                // Clone needs the x-vector encoder and the 12 Hz tokenizer.
                files.Add(GraphSpeakerEncoder + ".onnx");
                files.Add(GraphSpeakerEncoder + ".onnx.data");
                files.Add(GraphTokenizerEncoder + ".onnx");
                files.Add(GraphTokenizerEncoder + ".onnx.data");
            }
            return files;
        }

        public static List<string> MissingFiles(QwenCheckpoint checkpoint)
        {
            var dir = DirectoryFor(checkpoint);
            var missing = new List<string>();
            foreach (var rel in ExpectedFiles(checkpoint))
            {
                if (!File.Exists(Path.Combine(dir, rel)))
                    missing.Add(rel);
            }
            return missing;
        }

        public static bool IsPresent(QwenCheckpoint checkpoint) =>
            MissingFiles(checkpoint).Count == 0;

        /// <summary>Bytes on disk for the files the engine will open.</summary>
        public static long InstalledBytes(QwenCheckpoint checkpoint)
        {
            var dir = DirectoryFor(checkpoint);
            long total = 0;
            foreach (var rel in ExpectedFiles(checkpoint))
            {
                var path = Path.Combine(dir, rel);
                if (File.Exists(path))
                    total += new FileInfo(path).Length;
            }
            return total;
        }
    }
}
