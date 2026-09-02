using System;
using NUnit.Framework;
using QwenTTS.Audio;

namespace QwenTTS.Tests
{
    /// <summary>
    /// The audio front end, which is where clone quality is won or lost. Each
    /// of these covers a defect that shipped once: a WAV read at the wrong
    /// rate, and a resampler with no anti-aliasing.
    /// </summary>
    public class WavCodecTests
    {
        static float[] Tone(int samples, int rate, float hz)
        {
            var pcm = new float[samples];
            for (int i = 0; i < samples; i++)
                pcm[i] = 0.5f * Mathf.Sin(2f * Mathf.PI * hz * i / rate);
            return pcm;
        }

        [Test]
        public void Encode_then_decode_preserves_rate_and_length()
        {
            var pcm = Tone(2400, 24000, 220f);
            var bytes = WavCodec.Encode(pcm, 24000);

            Assert.IsTrue(WavCodec.TryDecode(bytes, out var back, out int rate, out int channels));
            Assert.AreEqual(24000, rate, "the decoder must read the rate from the header, not assume one");
            Assert.AreEqual(1, channels);
            Assert.AreEqual(pcm.Length, back.Length);
        }

        [Test]
        public void Decode_reports_the_header_rate_not_a_default()
        {
            // The regression: a 24 kHz reference read as 16 kHz plays 1.5x slow,
            // silently corrupting the signal a clone is derived from.
            foreach (int rate in new[] { 16000, 22050, 24000, 44100, 48000 })
            {
                var bytes = WavCodec.Encode(Tone(1000, rate, 100f), rate);
                Assert.IsTrue(WavCodec.TryDecode(bytes, out _, out int decoded, out _));
                Assert.AreEqual(rate, decoded);
            }
        }

        [Test]
        public void Round_trip_is_accurate_to_16_bit_quantisation()
        {
            var pcm = Tone(4096, 24000, 440f);
            WavCodec.TryDecode(WavCodec.Encode(pcm, 24000), out var back, out _, out _);

            double worst = 0;
            for (int i = 0; i < pcm.Length; i++)
                worst = Math.Max(worst, Math.Abs(pcm[i] - back[i]));
            // One 16-bit step is ~3.05e-5; rounding keeps us inside half of that
            // plus a margin for the 32767/32768 asymmetry.
            Assert.Less(worst, 4e-5, "16-bit round trip should be within one quantisation step");
        }

        [Test]
        public void Encode_clamps_out_of_range_input()
        {
            var pcm = new[] { 2f, -2f, 0f };
            WavCodec.TryDecode(WavCodec.Encode(pcm, 24000), out var back, out _, out _);
            Assert.LessOrEqual(back[0], 1f);
            Assert.GreaterOrEqual(back[1], -1f);
        }

        [Test]
        public void Decode_rejects_non_wav_bytes()
        {
            Assert.IsFalse(WavCodec.TryDecode(new byte[64], out _, out _, out _));
            Assert.IsFalse(WavCodec.TryDecode(null, out _, out _, out _));
        }

        [Test]
        public void ToMono24k_averages_channels_and_resamples()
        {
            // Two channels, 48 kHz, both carrying the same signal.
            int frames = 4800;
            var interleaved = new float[frames * 2];
            for (int i = 0; i < frames; i++)
            {
                float v = 0.25f;
                interleaved[i * 2] = v;
                interleaved[i * 2 + 1] = v;
            }

            var mono = WavCodec.ToMono24k(interleaved, 48000, 2);
            Assert.AreEqual(frames / 2, mono.Length, 2, "48 kHz to 24 kHz halves the sample count");
            // Away from the edges the constant should survive.
            Assert.AreEqual(0.25f, mono[mono.Length / 2], 1e-3f);
        }
    }

    public class AudioResampleTests
    {
        [Test]
        public void Same_rate_returns_the_input_untouched()
        {
            var pcm = new[] { 1f, 2f, 3f };
            Assert.AreSame(pcm, AudioResample.Resample(pcm, 24000, 24000));
        }

        [Test]
        public void Output_length_follows_the_rate_ratio()
        {
            var pcm = new float[24000];
            Assert.AreEqual(16000, AudioResample.Resample(pcm, 24000, 16000).Length);
            Assert.AreEqual(48000, AudioResample.Resample(pcm, 24000, 48000).Length);
        }

        [Test]
        public void Constant_signal_keeps_its_level()
        {
            // Unity gain at DC: the kernel is normalised by its own sum, so a
            // constant must come out at the same value away from the edges.
            var pcm = new float[4800];
            for (int i = 0; i < pcm.Length; i++)
                pcm[i] = 0.7f;

            var down = AudioResample.Resample(pcm, 24000, 16000);
            Assert.AreEqual(0.7f, down[down.Length / 2], 1e-4f);

            var up = AudioResample.Resample(pcm, 16000, 24000);
            Assert.AreEqual(0.7f, up[up.Length / 2], 1e-4f);
        }

        [Test]
        public void Downsampling_attenuates_content_above_the_new_nyquist()
        {
            // The point of the band-limited kernel: an 11 kHz tone cannot be
            // represented at 16 kHz and must be filtered out rather than folded
            // back into the speech band.
            int rate = 24000;
            var pcm = new float[rate];
            for (int i = 0; i < pcm.Length; i++)
                pcm[i] = Mathf.Sin(2f * Mathf.PI * 11000f * i / rate);

            var down = AudioResample.Resample(pcm, rate, 16000);

            double energy = 0;
            int from = down.Length / 4, to = down.Length * 3 / 4;
            for (int i = from; i < to; i++)
                energy += down[i] * down[i];
            double rms = Math.Sqrt(energy / (to - from));

            Assert.Less(rms, 0.2, "an 11 kHz tone should be largely removed when moving to 16 kHz");
        }

        [Test]
        public void Empty_input_is_handled()
        {
            Assert.AreEqual(0, AudioResample.Resample(Array.Empty<float>(), 24000, 16000).Length);
        }
    }

    // Local stand-in so the tests do not depend on UnityEngine.Mathf overload
    // resolution differing between editor versions.
    static class Mathf
    {
        public const float PI = 3.14159265358979f;
        public static float Sin(float x) => (float)Math.Sin(x);
    }
}
