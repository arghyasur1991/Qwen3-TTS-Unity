using System;
using System.Collections.Generic;
using UnityEngine;

namespace QwenTTS.Audio
{
    /// <summary>
    /// PCM assembly helpers for the audio this package produces and consumes.
    ///
    /// Generation hands back a `float[]`, and a caller almost always has to do
    /// something with it before playback — downmix a reference recording,
    /// match a target rate, or stitch several takes together with a beat
    /// between them. The alternative to putting these here is every host
    /// writing the same downmix-and-resample loop, so they live with the thing
    /// that hands out the samples.
    ///
    /// The `float[]` overloads are thread-agnostic and are the ones to prefer.
    /// The <see cref="AudioClip"/> overloads exist for convenience and must be
    /// called on Unity's main thread, because `AudioClip.Create`, `SetData` and
    /// `GetData` all require it.
    /// </summary>
    public static class QwenAudio
    {
        /// <summary>Interleaved multi-channel samples averaged down to one channel.</summary>
        public static float[] ToMono(float[] interleaved, int channels)
        {
            if (interleaved == null || interleaved.Length == 0)
                return Array.Empty<float>();
            if (channels <= 1)
                return interleaved;

            int frames = interleaved.Length / channels;
            var mono = new float[frames];
            for (int i = 0; i < frames; i++)
            {
                float sum = 0f;
                for (int c = 0; c < channels; c++)
                    sum += interleaved[i * channels + c];
                mono[i] = sum / channels;
            }
            return mono;
        }

        /// <summary>
        /// Band-limited resample. Returns the input unchanged when the rates
        /// already match.
        ///
        /// Windowed sinc rather than linear interpolation, which folds
        /// 8-12 kHz back into the speech band on the way down and leaves a
        /// stair-stepped spectrum on the way up — either of which moves a
        /// clone reference away from the voice being cloned.
        /// </summary>
        public static float[] Resample(float[] samples, int sourceRate, int targetRate)
        {
            if (samples == null || samples.Length == 0)
                return Array.Empty<float>();
            if (sourceRate <= 0 || targetRate <= 0)
                throw new ArgumentOutOfRangeException(nameof(targetRate), "Rates must be positive.");
            return sourceRate == targetRate ? samples : AudioResample.Resample(samples, sourceRate, targetRate);
        }

        /// <summary>Silent mono samples of the given duration.</summary>
        public static float[] Silence(int sampleRate, float seconds)
        {
            if (sampleRate <= 0)
                throw new ArgumentOutOfRangeException(nameof(sampleRate), "Rate must be positive.");
            if (seconds <= 0f)
                return Array.Empty<float>();
            return new float[SampleCount(seconds, sampleRate)];
        }

        /// <summary>
        /// Seconds to samples, rounded rather than truncated and computed in
        /// double.
        ///
        /// `(int)(0.01f * 1000)` is 9, not 10: the product is 9.999999 in
        /// single precision and the cast floors it. Truncating a requested gap
        /// or duration to one sample short is the kind of thing that never
        /// gets noticed and never gets fixed.
        /// </summary>
        static int SampleCount(float seconds, int sampleRate) =>
            (int)Math.Round((double)seconds * sampleRate);

        /// <summary>
        /// Mono segments joined in order, with <paramref name="gapSeconds"/> of
        /// silence between them but not before the first or after the last.
        /// Null and empty segments are skipped rather than producing a gap.
        /// </summary>
        public static float[] Concatenate(IReadOnlyList<float[]> segments, int sampleRate,
            float gapSeconds = 0f)
        {
            if (segments == null || segments.Count == 0)
                return Array.Empty<float>();
            if (sampleRate <= 0)
                throw new ArgumentOutOfRangeException(nameof(sampleRate), "Rate must be positive.");

            int gap = gapSeconds > 0f ? SampleCount(gapSeconds, sampleRate) : 0;

            int total = 0, kept = 0;
            foreach (var seg in segments)
            {
                if (seg == null || seg.Length == 0)
                    continue;
                total += seg.Length;
                kept++;
            }
            if (kept == 0)
                return Array.Empty<float>();
            total += gap * (kept - 1);

            var joined = new float[total];
            int at = 0, written = 0;
            foreach (var seg in segments)
            {
                if (seg == null || seg.Length == 0)
                    continue;
                if (written > 0 && gap > 0)
                    at += gap;   // the array is already zeroed
                Array.Copy(seg, 0, joined, at, seg.Length);
                at += seg.Length;
                written++;
            }
            return joined;
        }

        /// <summary>Silent mono clip. Main thread.</summary>
        public static AudioClip SilenceClip(int sampleRate, float seconds,
            string name = "QwenTtsSilence")
        {
            var samples = Silence(sampleRate, seconds);
            if (samples.Length == 0)
                samples = new float[1];
            var clip = AudioClip.Create(name, samples.Length, 1, sampleRate, false);
            clip.SetData(samples, 0);
            return clip;
        }

        /// <summary>
        /// Clips joined into one mono clip at <paramref name="sampleRate"/>,
        /// downmixing and resampling each as needed. Main thread, because it
        /// reads and writes clip data.
        /// </summary>
        public static AudioClip Concatenate(IReadOnlyList<AudioClip> clips, int sampleRate,
            float gapSeconds = 0f, string name = "QwenTtsConcat")
        {
            if (clips == null || clips.Count == 0)
                return null;

            var segments = new List<float[]>(clips.Count);
            foreach (var clip in clips)
            {
                if (clip == null || clip.samples == 0)
                    continue;
                var interleaved = new float[clip.samples * clip.channels];
                clip.GetData(interleaved, 0);
                segments.Add(Resample(ToMono(interleaved, clip.channels), clip.frequency, sampleRate));
            }

            var joined = Concatenate(segments, sampleRate, gapSeconds);
            if (joined.Length == 0)
                return null;

            var output = AudioClip.Create(name, joined.Length, 1, sampleRate, false);
            output.SetData(joined, 0);
            return output;
        }
    }
}
