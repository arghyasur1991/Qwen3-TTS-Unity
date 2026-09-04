using System;
using System.Collections.Generic;
using System.Threading;
using QwenTTS.Internal;

namespace QwenTTS.Engine
{
    /// <summary>
    /// Hands over finished audio while the rest of the utterance is still
    /// being generated.
    ///
    /// The 12.5 Hz codec decoder cannot be run on a slice in isolation. Its
    /// output depends on the whole input, not a bounded left window: giving a
    /// chunk 24 frames of preceding context still leaves a max error of 0.59
    /// on a signal that lives in [-1, 1] — audible as a click at every
    /// boundary. Overlap-and-trim, the usual answer, does not apply.
    ///
    /// What is true of this decoder is that <em>prefixes are stable</em>.
    /// Decoding frames [0, k) yields, for every sample it covers, the same
    /// values as decoding [0, T) for T &gt; k — measured at ~1e-5 worst case,
    /// about -90 dB. So instead of decoding slices, decode the whole prefix
    /// each time and pass on only the samples past the previous high-water
    /// mark. Nothing is ever spliced, so there is no seam to conceal, and the
    /// concatenation of the chunks is the same waveform a single decode of the
    /// finished utterance would have produced.
    ///
    /// The cost of that choice is re-decoding the prefix once per chunk. Fixed
    /// small chunks would make the total decode work grow with the square of
    /// the utterance, so chunk size doubles from a small first chunk up to a
    /// cap: the first audio still arrives after a fraction of a second, while
    /// total decode work stays near twice a single pass.
    /// </summary>
    internal sealed class StreamingVocode
    {
        readonly QwenVocoderModel _vocoder;
        readonly IProgress<SpeechChunk> _sink;
        readonly int _maxChunkFrames;
        readonly int _leadingFramesToDrop;
        readonly long[,,] _prefixCodes;
        readonly CancellationToken _token;

        int _nextChunkFrames;
        int _emittedSamples;
        int _emittedFrames;

        /// <param name="prefixCodes">
        /// Frames the decoder needs but the caller must not hear — the
        /// reference codes on the in-context clone path. Decoded every time and
        /// never emitted. Time-major <c>[1, T, quantizers]</c>, matching what
        /// the tokenizer encoder produces, unlike the generated codes.
        /// </param>
        public StreamingVocode(QwenVocoderModel vocoder, IProgress<SpeechChunk> sink,
            int firstChunkFrames, int maxChunkFrames, long[,,] prefixCodes,
            CancellationToken token)
        {
            _vocoder = vocoder;
            _sink = sink;
            _nextChunkFrames = Math.Max(1, firstChunkFrames);
            _maxChunkFrames = Math.Max(_nextChunkFrames, maxChunkFrames);
            _prefixCodes = prefixCodes;
            _leadingFramesToDrop = prefixCodes?.GetLength(1) ?? 0;
            _token = token;
        }

        /// <summary>Frames handed over so far, reference frames excluded.</summary>
        public int EmittedFrames => _emittedFrames;

        /// <summary>
        /// Called per generated frame. Decodes and emits once enough new frames
        /// have accumulated, otherwise returns immediately.
        /// </summary>
        public void OnFrame(List<long[]> framesSoFar)
        {
            if (framesSoFar.Count - _emittedFrames < _nextChunkFrames)
                return;
            Emit(framesSoFar, final: false);
            _nextChunkFrames = Math.Min(_maxChunkFrames, _nextChunkFrames * 2);
        }

        /// <summary>
        /// Emits whatever is left. The caller still gets the whole waveform
        /// from the non-streaming return value, so this is about the last chunk
        /// arriving, not about completeness.
        /// </summary>
        public void Finish(List<long[]> framesSoFar)
        {
            if (framesSoFar.Count > _emittedFrames || _emittedFrames == 0)
                Emit(framesSoFar, final: true);
        }

        void Emit(List<long[]> framesSoFar, bool final)
        {
            _token.ThrowIfCancellationRequested();

            int generated = framesSoFar.Count;
            if (generated == 0)
                return;

            var codes = BuildCodes(framesSoFar);
            float[] decoded;
            using (GenerationProfiler.Measure(GenerationProfiler.StreamVocoder))
                decoded = _vocoder.Decode(codes, _token);

            // Everything before the reference is the decoder warming up on
            // audio the listener already heard from the original speaker.
            int samplesPerFrame = decoded.Length / Math.Max(1, generated + _leadingFramesToDrop);
            int skip = _leadingFramesToDrop * samplesPerFrame;
            if (skip >= decoded.Length)
                return;

            int from = skip + _emittedSamples;
            if (from >= decoded.Length)
                return;

            int count = decoded.Length - from;
            var pcm = new float[count];
            Array.Copy(decoded, from, pcm, 0, count);

            int frameStart = _emittedFrames;
            _emittedSamples += count;
            _emittedFrames = generated;

            _sink.Report(new SpeechChunk(
                pcm, QwenTtsEngine.NativeSampleRate, frameStart,
                generated - frameStart, final));
        }

        long[,,] BuildCodes(List<long[]> framesSoFar)
        {
            int groups = framesSoFar[0].Length;
            int generated = framesSoFar.Count;
            int total = generated + _leadingFramesToDrop;
            var codes = new long[1, groups, total];

            // Reference codes arrive time-major; the vocoder wants
            // quantizer-major, same as PrependReferenceCodes does it.
            for (int f = 0; f < _leadingFramesToDrop; f++)
                for (int g = 0; g < groups; g++)
                    codes[0, g, f] = _prefixCodes[0, f, g];

            for (int f = 0; f < generated; f++)
            {
                var frame = framesSoFar[f];
                for (int g = 0; g < groups; g++)
                    codes[0, g, _leadingFramesToDrop + f] = frame[g];
            }
            return codes;
        }
    }
}
