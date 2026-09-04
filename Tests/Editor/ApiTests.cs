using System;
using System.IO;
using NUnit.Framework;
using QwenTTS.Engine;
using QwenTTS.Internal;

namespace QwenTTS.Tests
{
    public class ClonePromptFileTests
    {
        static ClonePrompt MakePrompt(int frames, int quantizers = 16)
        {
            var embedding = new float[2048];
            for (int i = 0; i < embedding.Length; i++)
                embedding[i] = i * 0.001f;

            long[,,] codes = null;
            if (frames > 0)
            {
                codes = new long[1, frames, quantizers];
                for (int t = 0; t < frames; t++)
                    for (int q = 0; q < quantizers; q++)
                        codes[0, t, q] = (t * 31 + q * 7) % 2048;
            }
            return new ClonePrompt(embedding, codes);
        }

        [Test]
        public void Round_trip_preserves_the_x_vector_and_the_codes()
        {
            var path = Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString("N") + ".bin");
            var original = MakePrompt(37);
            try
            {
                ClonePromptFile.Write(path, original);
                Assert.IsTrue(ClonePromptFile.TryRead(path, out var back));

                Assert.AreEqual(original.SpeakerEmbedding.Length, back.SpeakerEmbedding.Length);
                for (int i = 0; i < original.SpeakerEmbedding.Length; i++)
                    Assert.AreEqual(original.SpeakerEmbedding[i], back.SpeakerEmbedding[i]);

                Assert.AreEqual(original.ReferenceFrames, back.ReferenceFrames);
                Assert.IsTrue(back.HasIclCodes);
                for (int t = 0; t < original.ReferenceFrames; t++)
                    for (int q = 0; q < 16; q++)
                        Assert.AreEqual(original.ReferenceCodes[0, t, q], back.ReferenceCodes[0, t, q]);
            }
            finally
            {
                File.Delete(path);
            }
        }

        [Test]
        public void An_x_vector_only_prompt_round_trips_without_codes()
        {
            var path = Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString("N") + ".bin");
            try
            {
                ClonePromptFile.Write(path, MakePrompt(0));
                Assert.IsTrue(ClonePromptFile.TryRead(path, out var back));
                Assert.IsTrue(back.IsValid);
                Assert.IsFalse(back.HasIclCodes);
            }
            finally
            {
                File.Delete(path);
            }
        }

        [Test]
        public void A_missing_or_corrupt_file_is_a_miss_not_a_throw()
        {
            // The caller re-derives from the reference audio on a miss, so this
            // must never throw: a stale prompt from an older export is expected.
            Assert.IsFalse(ClonePromptFile.TryRead(
                Path.Combine(Path.GetTempPath(), "definitely-not-here.bin"), out _));

            var path = Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString("N") + ".bin");
            try
            {
                File.WriteAllBytes(path, new byte[] { 1, 2, 3, 4, 5, 6, 7, 8 });
                Assert.IsFalse(ClonePromptFile.TryRead(path, out _));
            }
            finally
            {
                File.Delete(path);
            }
        }

        [Test]
        public void Writing_an_empty_prompt_is_rejected()
        {
            var path = Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString("N") + ".bin");
            Assert.Throws<ArgumentException>(() => ClonePromptFile.Write(path, default));
        }
    }

    public class SpeechOptionsTests
    {
        [Test]
        public void Defaults_match_the_reference_generate_config()
        {
            var o = SpeechOptions.Default();
            Assert.AreEqual(0.9f, o.Temperature);
            Assert.AreEqual(50, o.TopK);
            Assert.AreEqual(1f, o.TopP);
            Assert.AreEqual(1.05f, o.RepetitionPenalty);
            Assert.AreEqual(2048, o.MaxNewTokens);
            Assert.AreEqual(QwenLanguages.Default, o.Language);
            Assert.AreEqual(0, o.SampleRate, "0 means the model's native rate");
        }

        [Test]
        public void Invalid_values_are_rejected_before_a_generate_starts()
        {
            Assert.Throws<ArgumentOutOfRangeException>(() =>
                new SpeechOptions { MaxNewTokens = 0 }.Validated());
            Assert.Throws<ArgumentOutOfRangeException>(() =>
                new SpeechOptions { TopP = 0f }.Validated());
            Assert.Throws<ArgumentOutOfRangeException>(() =>
                new SpeechOptions { TopP = 1.5f }.Validated());
            Assert.Throws<ArgumentOutOfRangeException>(() =>
                new SpeechOptions { SampleRate = -1 }.Validated());
            Assert.Throws<ArgumentException>(() =>
                new SpeechOptions { Language = "" }.Validated());
        }

        [Test]
        public void Sampling_params_carry_the_sub_talker_knobs_through()
        {
            var o = new SpeechOptions { SubTalkerTemperature = 0.4f, SubTalkerTopK = 12, SubTalkerTopP = 0.8f };
            var s = SamplingParams.From(o);
            Assert.AreEqual(0.4f, s.SubTemperature);
            Assert.AreEqual(12, s.SubTopK);
            Assert.AreEqual(0.8f, s.SubTopP);
        }
    }

    public class LanguageTests
    {
        [Test]
        public void The_ten_supported_languages_plus_auto_are_accepted()
        {
            Assert.AreEqual(10, QwenLanguages.All.Count);
            foreach (var language in QwenLanguages.All)
                Assert.IsTrue(QwenLanguages.IsSupported(language), language);
            Assert.IsTrue(QwenLanguages.IsSupported(QwenLanguages.Auto));
            Assert.IsTrue(QwenLanguages.IsSupported("ENGLISH"), "matching is case-insensitive");
        }

        [Test]
        public void Anything_else_is_rejected()
        {
            Assert.IsFalse(QwenLanguages.IsSupported("klingon"));
            Assert.IsFalse(QwenLanguages.IsSupported(""));
            Assert.IsFalse(QwenLanguages.IsSupported(null));
        }
    }

    public class ModelPathTests
    {
        [Test]
        public void The_root_is_overridable_and_falls_back_to_streaming_assets()
        {
            string original = QwenModelPaths.RootIsExplicit ? QwenModelPaths.Root : null;
            try
            {
                QwenModelPaths.Root = "/tmp/some/models";
                Assert.IsTrue(QwenModelPaths.RootIsExplicit);
                Assert.AreEqual("/tmp/some/models", QwenModelPaths.Root);

                QwenModelPaths.Root = null;
                Assert.IsFalse(QwenModelPaths.RootIsExplicit);
                StringAssert.Contains("QwenTTS", QwenModelPaths.Root);
            }
            finally
            {
                QwenModelPaths.Root = original;
            }
        }

        [Test]
        public void Each_checkpoint_has_its_own_folder_and_the_name_says_which()
        {
            Assert.AreNotEqual(
                QwenModelPaths.FolderName(QwenCheckpoint.VoiceDesign),
                QwenModelPaths.FolderName(QwenCheckpoint.Base));
            StringAssert.Contains("VoiceDesign", QwenModelPaths.FolderName(QwenCheckpoint.VoiceDesign));
            StringAssert.Contains("Base", QwenModelPaths.FolderName(QwenCheckpoint.Base));
        }

        [Test]
        public void Only_the_base_checkpoint_requires_the_clone_encoders()
        {
            var design = QwenModelPaths.ExpectedFiles(QwenCheckpoint.VoiceDesign);
            var clone = QwenModelPaths.ExpectedFiles(QwenCheckpoint.Base);

            Assert.IsFalse(Contains(design, "speaker_encoder.onnx"));
            Assert.IsFalse(Contains(design, "tokenizer_encoder.onnx"));
            Assert.IsTrue(Contains(clone, "speaker_encoder.onnx"));
            Assert.IsTrue(Contains(clone, "tokenizer_encoder.onnx"));
        }

        [Test]
        public void The_checklist_does_not_demand_files_the_engine_never_reads()
        {
            // codec_head_weight.npy is produced by the exporter and never
            // loaded; listing it would fail installs that are in fact complete.
            foreach (var checkpoint in new[] { QwenCheckpoint.VoiceDesign, QwenCheckpoint.Base })
            {
                Assert.IsFalse(Contains(QwenModelPaths.ExpectedFiles(checkpoint), "codec_head_weight.npy"));
                Assert.IsFalse(Contains(QwenModelPaths.ExpectedFiles(checkpoint), "speaker_ids.json"));
            }
        }

        static bool Contains(System.Collections.Generic.IReadOnlyList<string> files, string needle)
        {
            for (int i = 0; i < files.Count; i++)
            {
                if (files[i].EndsWith(needle, StringComparison.Ordinal))
                    return true;
            }
            return false;
        }
    }
}
