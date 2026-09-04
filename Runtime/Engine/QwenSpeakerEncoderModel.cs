using System;
using System.Collections.Generic;
using Microsoft.ML.OnnxRuntime.Tensors;
using QwenTTS.Engine;
using QwenTTS.Onnx;
using QwenTTS.Audio;

namespace QwenTTS.Engine
{
    internal sealed class QwenSpeakerEncoderModel : QwenOnnxModel
    {
        public QwenSpeakerEncoderModel(ExecutionProvider executionProvider = ExecutionProvider.CPU)
            : base(QwenModelPaths.GraphSpeakerEncoder, QwenCheckpoint.Base, executionProvider)
        {
        }

        public float[] Encode(float[] samples24k)
        {
            if (samples24k == null || samples24k.Length == 0)
                throw new ArgumentException("Reference audio is empty.", nameof(samples24k));

            var mel = MelSpectrogram.Extract(samples24k);
            int tMel = mel.GetLength(0);
            int nMels = mel.GetLength(1);
            var flat = MelSpectrogram.FlattenTimeFirst(mel);

            string inputName = ResolveInputName("mels", "mel_spectrogram");
            var feeds = new List<Microsoft.ML.OnnxRuntime.NamedOnnxValue>
            {
                Microsoft.ML.OnnxRuntime.NamedOnnxValue.CreateFromTensor(
                    inputName, new DenseTensor<float>(flat, new[] { 1, tMel, nMels }))
            };

            using var results = Run(feeds);
            return CopyFloat(results[0]);
        }
    }
}
