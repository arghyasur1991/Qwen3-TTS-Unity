using System;
using System.Collections.Generic;
using System.Threading;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using QwenTTS.Onnx;

namespace QwenTTS.Engine
{
    /// <summary>
    /// The codec decoder: 16 code groups per frame in, 24 kHz waveform out at
    /// 1920 samples per frame. Both checkpoints use the same graph.
    /// </summary>
    internal sealed class QwenVocoderModel : QwenOnnxModel
    {
        public const int SamplesPerFrame = 1920;
        public const int SampleRate = 24000;

        private readonly bool _timeMajor;

        public QwenVocoderModel(QwenCheckpoint checkpoint, bool timeMajor = false,
            ExecutionProvider executionProvider = ExecutionProvider.CPU)
            : base(QwenModelPaths.GraphVocoder, checkpoint, executionProvider)
        {
            _timeMajor = timeMajor;
        }

        public float[] Decode(long[,,] codesQuantizerMajor, CancellationToken cancellationToken = default)
        {
            int batch = codesQuantizerMajor.GetLength(0);
            int quantizers = codesQuantizerMajor.GetLength(1);
            int timesteps = codesQuantizerMajor.GetLength(2);
            var flat = new long[batch * quantizers * timesteps];
            int n = 0;
            if (_timeMajor)
            {
                for (int b = 0; b < batch; b++)
                    for (int t = 0; t < timesteps; t++)
                        for (int q = 0; q < quantizers; q++)
                            flat[n++] = codesQuantizerMajor[b, q, t];
            }
            else
            {
                for (int b = 0; b < batch; b++)
                    for (int q = 0; q < quantizers; q++)
                        for (int t = 0; t < timesteps; t++)
                            flat[n++] = codesQuantizerMajor[b, q, t];
            }

            int[] shape = _timeMajor
                ? new[] { batch, timesteps, quantizers }
                : new[] { batch, quantizers, timesteps };
            return DecodeFlat(flat, shape, timesteps, cancellationToken);
        }

        public float[] Decode(long[,] codesTimeMajor, CancellationToken cancellationToken = default)
        {
            int t = codesTimeMajor.GetLength(0);
            int groups = codesTimeMajor.GetLength(1);
            if (t == 0)
                return Array.Empty<float>();

            var flat = new long[t * groups];
            int n = 0;
            for (int i = 0; i < t; i++)
                for (int g = 0; g < groups; g++)
                    flat[n++] = codesTimeMajor[i, g];

            return DecodeFlat(flat, new[] { 1, t, groups }, t, cancellationToken);
        }

        private float[] DecodeFlat(long[] flat, int[] shape, int timesteps, CancellationToken cancellationToken)
        {
            cancellationToken.ThrowIfCancellationRequested();
            string inputName = ResolveInputName("audio_codes", "codes");
            var feeds = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor(inputName, new DenseTensor<long>(flat, shape))
            };

            using var _prof = QwenTTS.Internal.GenerationProfiler.Measure(
                QwenTTS.Internal.GenerationProfiler.Vocoder);
            using var results = Run(feeds);
            var wav = CopyFloat(results[0]);
            // The flat buffer is the waveform. Do not read Dimensions[1]: the
            // output is [batch, 1, samples], so that axis is channels, not time.
            int wavLen = wav.Length;

            int target = timesteps * SamplesPerFrame;
            if (target > wavLen)
                target = wavLen;
            if (results.Count > 1)
            {
                var lengths = CopyLong(results[1]);
                // lengths[0] is sample count when it is in a plausible range.
                // Frame counts (T, or 1) must not trim a 24 kHz buffer to silence.
                if (lengths.Length > 0 && lengths[0] >= SamplesPerFrame / 2 && lengths[0] < wavLen)
                    target = Math.Min(target, (int)lengths[0]);
            }

            if (!_timeMajor && wav.Length != timesteps * SamplesPerFrame && wav.Length < target)
            {
                throw new InvalidOperationException(
                    $"Vocoder output mismatch: expected {timesteps * SamplesPerFrame} samples, got {wav.Length}.");
            }

            if (target >= wav.Length)
                return wav;
            var trimmed = new float[target];
            Array.Copy(wav, trimmed, target);
            return trimmed;
        }
    }
}
