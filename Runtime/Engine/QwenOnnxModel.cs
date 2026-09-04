using System;
using System.Collections.Generic;
using System.IO;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using QwenTTS.Onnx;

namespace QwenTTS.Engine
{
    /// <summary>
    /// A Qwen graph on top of <see cref="ORTModel"/>, which owns load policy,
    /// execution-provider selection and session lifetime.
    ///
    /// Autoregressive generation calls Run() synchronously after EnsureLoaded.
    /// Do not wrap an individual decode step in a Task: there are sixteen per
    /// output frame and the scheduling costs more than the step.
    /// </summary>
    internal class QwenOnnxModel : ORTModel
    {
        // An expected export is a few MB of protobuf beside a sibling
        // .onnx.data. A multi-gigabyte .onnx means a single-file export, which
        // ONNX Runtime rejects as an invalid protobuf past 2 GB — worth failing
        // on with a clear message rather than a parse error.
        private const long MaxOnnxBytes = 64_000_000;

        public readonly QwenCheckpoint Checkpoint;

        public QwenOnnxModel(string modelName, QwenCheckpoint checkpoint,
            ExecutionProvider executionProvider = ExecutionProvider.CPU)
            : base(modelName, QwenModelPaths.FolderName(checkpoint),
                   ResolvePrecision(modelName, checkpoint),
                   executionProvider, deferLoad: true)
        {
            Checkpoint = checkpoint;
        }

        /// <summary>
        /// int8 per graph, not per checkpoint. Only the talker and code
        /// predictor are worth quantizing — they are the two that stream their
        /// whole weight set once per token — and asking for int8 must not stop
        /// the vocoder or the encoders loading, so a missing quantized file is
        /// a silent downgrade rather than an error.
        /// </summary>
        static Precision ResolvePrecision(string modelName, QwenCheckpoint checkpoint)
        {
            if (QwenModelPaths.Precision != QwenPrecision.Int8)
                return Precision.FP32;
            if (!QwenModelPaths.HasInt8(checkpoint, modelName))
                return Precision.FP32;
            return Precision.Int8;
        }

        public new void EnsureLoaded()
        {
            string path = ModelFilePath;
            var info = new FileInfo(path);
            if (info.Exists && info.Length > MaxOnnxBytes)
                throw new InvalidOperationException($"ONNX file too large ({info.Length / 1e9:F2} GB): {path}");
            base.EnsureLoaded();
        }

        public IDisposableReadOnlyCollection<DisposableNamedOnnxValue> Run(IReadOnlyCollection<NamedOnnxValue> inputs)
        {
            EnsureLoaded();
            SetLoggingParam(ModelName);
            return Session.Run(inputs);
        }

        public string ResolveInputName(params string[] candidates)
        {
            EnsureLoaded();
            foreach (var c in candidates)
            {
                if (Session.InputMetadata.ContainsKey(c))
                    return c;
            }

            foreach (var key in Session.InputMetadata.Keys)
                return key;
            return candidates.Length > 0 ? candidates[0] : "input";
        }

        public IReadOnlyList<string> GraphInputNames
        {
            get
            {
                EnsureLoaded();
                return InputNames;
            }
        }

        public InferenceSession GetSession()
        {
            EnsureLoaded();
            return Session;
        }

        public static float[] CopyFloat(DisposableNamedOnnxValue value)
        {
            if (value.Value is DenseTensor<float> dense)
                return dense.Buffer.ToArray();
            return ToArray(value.AsEnumerable<float>());
        }

        public static long[] CopyLong(DisposableNamedOnnxValue value)
        {
            if (value.Value is DenseTensor<long> dense)
                return dense.Buffer.ToArray();
            return ToArray(value.AsEnumerable<long>());
        }

        public static DisposableNamedOnnxValue FindNamed(
            IDisposableReadOnlyCollection<DisposableNamedOnnxValue> outputs, string name)
        {
            foreach (var o in outputs)
            {
                if (o.Name == name)
                    return o;
            }
            throw new InvalidOperationException($"ONNX output '{name}' not found.");
        }

        public static float[] LastHidden(DisposableNamedOnnxValue value, int hidden)
        {
            var all = CopyFloat(value);
            if (all.Length == hidden)
                return all;
            var last = new float[hidden];
            Array.Copy(all, all.Length - hidden, last, 0, hidden);
            return last;
        }

        private static T[] ToArray<T>(IEnumerable<T> src)
        {
            if (src is T[] arr)
                return arr;
            var list = new List<T>();
            foreach (var v in src)
                list.Add(v);
            return list.ToArray();
        }
    }
}
