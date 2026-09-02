using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Text;

namespace QwenTTS.Internal
{
    /// <summary>
    /// Wall-clock accumulator for the synthesis path, off by default.
    ///
    /// Generation is ~16 ONNX runs per output frame with buffer shuffling
    /// between them, so "it feels slow" is not actionable — the question is
    /// always which of the sixteen. Enable via
    /// <see cref="QwenTts.ProfilingEnabled"/>, synthesise, then read
    /// <see cref="QwenTts.ProfileReport"/>.
    ///
    /// Stopwatch.GetTimestamp is tens of nanoseconds against stages measured in
    /// milliseconds, and the counters are pre-seeded so a measured run does not
    /// allocate. It is still gated, because a disabled profiler should cost a
    /// branch and nothing else.
    /// </summary>
    internal static class GenerationProfiler
    {
        public const string PrefillBuild = "prefill.build_embedding";
        public const string PrefillRun = "prefill.talker_run";
        public const string TalkerRun = "talker.decode_run";
        public const string TalkerKvCopy = "talker.kv_copy";
        public const string TalkerOutCopy = "talker.logits_hidden_copy";
        public const string SampleGroup0 = "sample.group0";
        public const string CpRun = "cp.run";
        public const string CpKvCopy = "cp.kv_copy";
        public const string CpSample = "cp.sample";
        public const string CpEmbed = "cp.embedding_lookup";
        public const string NextInput = "talker.next_input_sum";
        public const string Vocoder = "vocoder";
        public const string StreamVocoder = "vocoder.streamed_prefix";
        public const string SpeakerEncoder = "clone.speaker_encoder";
        public const string TokenizerEncoder = "clone.tokenizer_encoder";

        static readonly string[] Stages =
        {
            PrefillBuild, PrefillRun, TalkerRun, TalkerKvCopy, TalkerOutCopy,
            SampleGroup0, CpRun, CpKvCopy, CpSample, CpEmbed, NextInput,
            Vocoder, StreamVocoder, SpeakerEncoder, TokenizerEncoder,
        };

        static readonly Dictionary<string, long> Ticks = NewCounters();
        static readonly Dictionary<string, long> Calls = NewCounters();
        static readonly object Gate = new object();
        static long _wallStart;
        static long _wallTicks;
        static int _frames;

        public static bool Enabled;

        static Dictionary<string, long> NewCounters()
        {
            var d = new Dictionary<string, long>(Stages.Length);
            foreach (var s in Stages) d[s] = 0;
            return d;
        }

        public static void Reset()
        {
            lock (Gate)
            {
                foreach (var s in Stages) { Ticks[s] = 0; Calls[s] = 0; }
                _frames = 0;
                _wallTicks = 0;
                _wallStart = Stopwatch.GetTimestamp();
            }
        }

        /// <summary>Records the frame count; does not stop the clock.</summary>
        public static void SetFrames(int frames)
        {
            if (!Enabled) return;
            lock (Gate) { _frames = frames; }
        }

        /// <summary>
        /// Stops the wall clock. Called once synthesis is fully done, vocoder
        /// included — stopping it when the frame loop ends left the vocoder
        /// counted in the stages but not in the total, which showed up as a
        /// negative unattributed row.
        /// </summary>
        public static void StopWall()
        {
            if (!Enabled) return;
            lock (Gate) { _wallTicks = Stopwatch.GetTimestamp() - _wallStart; }
        }

        public static Scope Measure(string stage) => new Scope(stage);

        public static void Add(string stage, long ticks)
        {
            if (!Enabled) return;
            lock (Gate)
            {
                Ticks[stage] += ticks;
                Calls[stage] += 1;
            }
        }

        internal readonly struct Scope : IDisposable
        {
            readonly string _stage;
            readonly long _start;

            public Scope(string stage)
            {
                if (!Enabled) { _stage = null; _start = 0; return; }
                _stage = stage;
                _start = Stopwatch.GetTimestamp();
            }

            public void Dispose()
            {
                if (_stage == null) return;
                Add(_stage, Stopwatch.GetTimestamp() - _start);
            }
        }

        public static string Report()
        {
            lock (Gate)
            {
                double perTick = 1000.0 / Stopwatch.Frequency;
                double wall = _wallTicks * perTick;
                double accounted = 0;
                foreach (var s in Stages) accounted += Ticks[s] * perTick;

                var sb = new StringBuilder();
                sb.AppendLine($"[QwenTts] {_frames} frames ({_frames / 12.5:F2}s audio) in {wall:F0} ms");
                sb.AppendLine($"{"stage",-30} {"ms",9} {"calls",8} {"ms/call",10} {"share",7}");
                foreach (var s in Stages)
                {
                    long calls = Calls[s];
                    if (calls == 0) continue;
                    double ms = Ticks[s] * perTick;
                    sb.AppendLine($"{s,-30} {ms,9:F1} {calls,8} {ms / calls,10:F3} "
                                  + $"{(wall > 0 ? ms / wall : 0),7:P1}");
                }
                double other = wall - accounted;
                sb.AppendLine($"{"(unattributed)",-30} {other,9:F1} {"",8} {"",10} "
                              + $"{(wall > 0 ? other / wall : 0),7:P1}");
                if (_frames > 0)
                    sb.AppendLine($"realtime factor: {wall / (_frames / 12.5 * 1000.0):F2}x slower than playback");
                return sb.ToString();
            }
        }
    }
}
