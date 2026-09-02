// Ported from ElBruno.QwenTTS (MIT) — https://github.com/elbruno/ElBruno.QwenTTS
// Qwen3-TTS ONNX inference. Public API lives in Runtime/Api.

using System;
using System.Collections.Generic;
using System.IO;
using System.Threading.Tasks;
using Newtonsoft.Json;
using QwenTTS.Internal;

namespace QwenTTS.Engine
{
    /// <summary>
    /// Embedding matrices in AllocHGlobal. Editor keep-alive stashes the
    /// pointers across domain reload so 1.5 GB npy + CP projection tables
    /// are not rebuilt.
    /// </summary>
    internal sealed class EmbeddingStore : IDisposable
    {
        public const int CpGroupCount = 15;

        NativeFloatBuffer _textEmbedding;
        NativeFloatBuffer _fc1Weight;
        NativeFloatBuffer _fc1Bias;
        NativeFloatBuffer _fc2Weight;
        NativeFloatBuffer _fc2Bias;
        NativeFloatBuffer _talkerCodecEmbedding;
        readonly NativeFloatBuffer[] _cpCodecEmbeddings = new NativeFloatBuffer[CpGroupCount];

        NativeFloatBuffer _cpProjectionWeight;
        NativeFloatBuffer _cpProjectionBias;
        NativeFloatBuffer[] _projectedCpCodecEmbeddings;
        NativeFloatBuffer _projectedTalkerCodecEmbedding;
        // Projected rows are filled on first use. Eagerly projecting every row
        // of all 16 codec tables is ~71 GFLOP (13+ seconds) to serve the ~16
        // rows a frame actually reads.
        bool[] _projectedTalkerReady;
        bool[][] _projectedCpReady;

        readonly int _textHiddenSize;
        readonly int _fc1OutSize;
        readonly int _hiddenSize;
        readonly int _cpHiddenSize;
        readonly int _cpModelHiddenSize;

        public ModelConfig Config { get; }

        public int HiddenSize => _hiddenSize;
        public int TextHiddenSize => _textHiddenSize;
        public int CpHiddenSize => _cpHiddenSize;
        public bool HasCpProjection => _cpProjectionWeight != null && !_cpProjectionWeight.IsEmpty;
        public int CpModelHiddenSize => _cpModelHiddenSize;

        public EmbeddingStore(string embeddingsDir, string configPath)
        {
            Config = LoadConfig(configPath);

            NativeFloatBuffer text = null, fc1w = null, fc1b = null, fc2w = null, fc2b = null, talker = null;
            var cpLocal = new NativeFloatBuffer[CpGroupCount];
            Parallel.Invoke(
                () => text = NpyReader.ReadNative2D(Path.Combine(embeddingsDir, "text_embedding.npy")),
                () => fc1w = NpyReader.ReadNative2D(Path.Combine(embeddingsDir, "text_projection_fc1_weight.npy")),
                () => fc1b = NpyReader.ReadNative1D(Path.Combine(embeddingsDir, "text_projection_fc1_bias.npy")),
                () => fc2w = NpyReader.ReadNative2D(Path.Combine(embeddingsDir, "text_projection_fc2_weight.npy")),
                () => fc2b = NpyReader.ReadNative1D(Path.Combine(embeddingsDir, "text_projection_fc2_bias.npy")),
                () => talker = NpyReader.ReadNative2D(Path.Combine(embeddingsDir, "talker_codec_embedding.npy")),
                () =>
                {
                    Parallel.For(0, CpGroupCount, i =>
                    {
                        cpLocal[i] = NpyReader.ReadNative2D(
                            Path.Combine(embeddingsDir, $"cp_codec_embedding_{i}.npy"));
                    });
                });
            _textEmbedding = text;
            _fc1Weight = fc1w;
            _fc1Bias = fc1b;
            _fc2Weight = fc2w;
            _fc2Bias = fc2b;
            _talkerCodecEmbedding = talker;
            for (int i = 0; i < CpGroupCount; i++)
                _cpCodecEmbeddings[i] = cpLocal[i];

            _textHiddenSize = _textEmbedding.Cols;
            _fc1OutSize = _fc1Weight.Rows;
            _hiddenSize = _fc2Weight.Rows;
            _cpHiddenSize = _cpCodecEmbeddings[0].Cols;
            _cpModelHiddenSize = Config.code_predictor.hidden_size > 0
                ? Config.code_predictor.hidden_size
                : _cpHiddenSize;

            var projWeightPath = Path.Combine(embeddingsDir, "cp_projection_weight.npy");
            var projBiasPath = Path.Combine(embeddingsDir, "cp_projection_bias.npy");
            if (File.Exists(projWeightPath) && File.Exists(projBiasPath))
            {
                _cpProjectionWeight = NpyReader.ReadNative2D(projWeightPath);
                _cpProjectionBias = NpyReader.ReadNative1D(projBiasPath);
                if (_cpProjectionWeight.Rows != _cpProjectionBias.Rows)
                    throw new InvalidDataException(
                        $"CP projection dimension mismatch: weight rows ({_cpProjectionWeight.Rows}) != bias length ({_cpProjectionBias.Rows})");
                if (_cpProjectionWeight.Cols != _hiddenSize)
                    throw new InvalidDataException(
                        $"CP projection input mismatch: weight columns ({_cpProjectionWeight.Cols}) != hidden_size ({_hiddenSize})");
                AllocateProjected();
            }
        }

        static ModelConfig LoadConfig(string configPath)
        {
            var configJson = File.ReadAllText(configPath);
            return JsonConvert.DeserializeObject<ModelConfig>(configJson)
                ?? throw new InvalidDataException("Failed to parse config.json");
        }

        void AllocateProjected()
        {
            int projOutDim = _cpProjectionWeight.Rows;

            _projectedCpCodecEmbeddings = new NativeFloatBuffer[CpGroupCount];
            _projectedCpReady = new bool[CpGroupCount][];
            for (int g = 0; g < CpGroupCount; g++)
            {
                int vocab = _cpCodecEmbeddings[g].Rows;
                _projectedCpCodecEmbeddings[g] = NativeFloatBuffer.Alloc(vocab, projOutDim);
                _projectedCpReady[g] = new bool[vocab];
            }

            int talkerVocab = _talkerCodecEmbedding.Rows;
            _projectedTalkerCodecEmbedding = NativeFloatBuffer.Alloc(talkerVocab, projOutDim);
            _projectedTalkerReady = new bool[talkerVocab];
        }

        static unsafe void ProjectRow(
            IntPtr weight, IntPtr bias, IntPtr src, IntPtr dst,
            int t, int srcDim, int projOutDim, int wRows, int wCols)
        {
            float* inRow = (float*)src + (long)t * srcDim;
            float* outRow = (float*)dst + (long)t * projOutDim;
            MatVec((float*)weight, wRows, wCols, inRow, outRow, (float*)bias);
        }

        public void TextEmbedding(int tokenId, Span<float> output)
        {
            if (output.Length != _textHiddenSize)
                throw new ArgumentException($"Output must be length {_textHiddenSize}");
            _textEmbedding.CopyRow(tokenId, output);
        }

        public void TextProjection(ReadOnlySpan<float> input, Span<float> output)
        {
            if (input.Length != _textHiddenSize)
                throw new ArgumentException($"Input must be length {_textHiddenSize}");
            if (output.Length != _hiddenSize)
                throw new ArgumentException($"Output must be length {_hiddenSize}");

            var hidden = new float[_fc1OutSize];
            MatMul(_fc1Weight, input, hidden);
            for (int i = 0; i < _fc1OutSize; i++)
                hidden[i] = SiLU(hidden[i] + At(_fc1Bias, i));
            MatMul(_fc2Weight, hidden, output);
            for (int i = 0; i < _hiddenSize; i++)
                output[i] += At(_fc2Bias, i);
        }

        public void TalkerCodecEmbedding(int tokenId, Span<float> output)
        {
            if (output.Length != _hiddenSize)
                throw new ArgumentException($"Output must be length {_hiddenSize}");
            _talkerCodecEmbedding.CopyRow(tokenId, output);
        }

        public void CpCodecEmbedding(int groupIndex, int tokenId, Span<float> output)
        {
            if (groupIndex < 0 || groupIndex >= CpGroupCount)
                throw new ArgumentException($"groupIndex must be 0-14, got {groupIndex}");
            if (output.Length != _cpHiddenSize)
                throw new ArgumentException($"Output must be length {_cpHiddenSize}");
            _cpCodecEmbeddings[groupIndex].CopyRow(tokenId, output);
        }

        public void CpProjection(ReadOnlySpan<float> input, Span<float> output)
        {
            if (!HasCpProjection)
                throw new InvalidOperationException("CP projection weights not loaded");
            if (input.Length < _cpProjectionWeight.Cols)
                throw new ArgumentException(
                    $"CP projection input too short: got {input.Length}, need {_cpProjectionWeight.Cols}");
            if (output.Length < _cpProjectionWeight.Rows)
                throw new ArgumentException(
                    $"CP projection output too short: got {output.Length}, need {_cpProjectionWeight.Rows}");

            MatMul(_cpProjectionWeight, input, output);
            for (int i = 0; i < _cpProjectionWeight.Rows; i++)
                output[i] += At(_cpProjectionBias, i);
        }

        // Callers hold the engine lock, so these fills are serialized.
        public void ProjectedCpCodecEmbedding(int groupIndex, int tokenId, Span<float> output)
        {
            if (_projectedCpCodecEmbeddings == null)
                throw new InvalidOperationException("Projected CP codec embeddings not available");
            if (groupIndex < 0 || groupIndex >= CpGroupCount)
                throw new ArgumentException($"groupIndex must be 0-14, got {groupIndex}");

            var ready = _projectedCpReady[groupIndex];
            if (!ready[tokenId])
            {
                ProjectRow(
                    _cpProjectionWeight.Ptr, _cpProjectionBias.Ptr,
                    _cpCodecEmbeddings[groupIndex].Ptr, _projectedCpCodecEmbeddings[groupIndex].Ptr,
                    tokenId, _cpHiddenSize, _cpProjectionWeight.Rows,
                    _cpProjectionWeight.Rows, _cpProjectionWeight.Cols);
                ready[tokenId] = true;
            }
            _projectedCpCodecEmbeddings[groupIndex].CopyRow(tokenId, output);
        }

        public void ProjectedTalkerCodecEmbedding(int tokenId, Span<float> output)
        {
            if (_projectedTalkerCodecEmbedding == null)
                throw new InvalidOperationException("Projected talker codec embedding not available");

            if (!_projectedTalkerReady[tokenId])
            {
                ProjectRow(
                    _cpProjectionWeight.Ptr, _cpProjectionBias.Ptr,
                    _talkerCodecEmbedding.Ptr, _projectedTalkerCodecEmbedding.Ptr,
                    tokenId, _hiddenSize, _cpProjectionWeight.Rows,
                    _cpProjectionWeight.Rows, _cpProjectionWeight.Cols);
                _projectedTalkerReady[tokenId] = true;
            }
            _projectedTalkerCodecEmbedding.CopyRow(tokenId, output);
        }

        public void Dispose()
        {
            _textEmbedding?.Free();
            _fc1Weight?.Free();
            _fc1Bias?.Free();
            _fc2Weight?.Free();
            _fc2Bias?.Free();
            _talkerCodecEmbedding?.Free();
            for (int i = 0; i < CpGroupCount; i++)
                _cpCodecEmbeddings[i]?.Free();
            _cpProjectionWeight?.Free();
            _cpProjectionBias?.Free();
            if (_projectedCpCodecEmbeddings != null)
            {
                for (int i = 0; i < _projectedCpCodecEmbeddings.Length; i++)
                    _projectedCpCodecEmbeddings[i]?.Free();
            }
            _projectedTalkerCodecEmbedding?.Free();
        }

        static float SiLU(float x) => x / (1.0f + MathF.Exp(-x));

        static unsafe float At(NativeFloatBuffer buf, int i)
        {
            return ((float*)buf.Ptr)[i];
        }

        static unsafe void MatMul(NativeFloatBuffer weight, ReadOnlySpan<float> input, Span<float> output)
        {
            MatMul((float*)weight.Ptr, weight.Rows, weight.Cols, input, output);
        }

        static unsafe void MatMul(float* weight, int M, int N, ReadOnlySpan<float> input, Span<float> output)
        {
            fixed (float* inPtr = input)
            fixed (float* outPtr = output)
                MatVec(weight, M, N, inPtr, outPtr, null);
        }

        // Below this many rows a Parallel.For fork costs more than it saves.
        const int ParallelRowFloor = 64;

        /// <summary>
        /// output[i] = dot(weight[i], input), plus bias[i] when bias is given.
        ///
        /// The hottest arithmetic in the package: sixteen of these run per
        /// output frame (fifteen codec-embedding projections and the text MLP),
        /// and as a plain scalar loop they measured 7.25 s of a 14.6 s
        /// utterance — half the wall clock, more than the talker and
        /// code-predictor ONNX runs put together.
        ///
        /// Unity's Mono reports Vector.IsHardwareAccelerated false, so
        /// System.Numerics vectors would take a software path and lose to
        /// scalar code. What does help is four independent accumulators (float
        /// addition is not reassociable, so a single accumulator serialises on
        /// FP-add latency rather than issue throughput) and splitting rows over
        /// cores, each output element being an independent dot product with
        /// nothing shared between them.
        /// </summary>
        static unsafe void MatVec(float* weight, int M, int N, float* input, float* output, float* bias)
        {
            if (M < ParallelRowFloor)
            {
                for (int i = 0; i < M; i++)
                    output[i] = DotRow(weight + (long)i * N, input, N) + (bias == null ? 0f : bias[i]);
                return;
            }

            // A lambda cannot close over a pointer, but it can close over the
            // address as an IntPtr.
            var w = (IntPtr)weight;
            var inp = (IntPtr)input;
            var outp = (IntPtr)output;
            var bia = (IntPtr)bias;
            int cols = N;

            Parallel.For(0, M, i =>
            {
                float* bp = (float*)bia;
                ((float*)outp)[i] =
                    DotRow((float*)w + (long)i * cols, (float*)inp, cols)
                    + (bp == null ? 0f : bp[i]);
            });
        }

        static unsafe float DotRow(float* row, float* input, int n)
        {
            float s0 = 0f, s1 = 0f, s2 = 0f, s3 = 0f;
            int j = 0;
            int limit = n - 3;
            for (; j < limit; j += 4)
            {
                s0 += row[j] * input[j];
                s1 += row[j + 1] * input[j + 1];
                s2 += row[j + 2] * input[j + 2];
                s3 += row[j + 3] * input[j + 3];
            }
            float sum = (s0 + s1) + (s2 + s3);
            for (; j < n; j++)
                sum += row[j] * input[j];
            return sum;
        }
    }

    internal sealed class ModelConfig
    {
        public TalkerConfig talker { get; set; } = new();
        public CodePredictorConfig code_predictor { get; set; } = new();
        public TtsConfig tts { get; set; } = new();
        public Dictionary<string, int> language_ids { get; set; } = new();
        public Dictionary<string, object> speaker_dialect { get; set; } = new();
    }

    internal sealed class TalkerConfig
    {
        public int codec_eos_token_id { get; set; }
        public int codec_pad_id { get; set; }
        public int codec_bos_id { get; set; }
        public int codec_think_id { get; set; }
        public int codec_nothink_id { get; set; }
        public int codec_think_bos_id { get; set; }
        public int codec_think_eos_id { get; set; }
        public int num_code_groups { get; set; }
        public int hidden_size { get; set; }
        public int text_hidden_size { get; set; }
        public int num_hidden_layers { get; set; }
        public int num_key_value_heads { get; set; }
        public int head_dim { get; set; }
        public int vocab_size { get; set; }
    }

    internal sealed class CodePredictorConfig
    {
        public int num_hidden_layers { get; set; }
        public int num_key_value_heads { get; set; }
        public int head_dim { get; set; }
        public int vocab_size { get; set; }
        public int hidden_size { get; set; }
    }

    internal sealed class TtsConfig
    {
        public int tts_bos_token_id { get; set; }
        public int tts_eos_token_id { get; set; }
        public int tts_pad_token_id { get; set; }
    }
}
