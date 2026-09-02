namespace QwenTTS.Engine
{
    /// <summary>
    /// Sampling knobs for one generate, flattened out of
    /// <see cref="SpeechOptions"/> so the engine does not depend on the public
    /// mutable class. Defaults match Qwen's own generate config.
    /// </summary>
    internal readonly struct SamplingParams
    {
        public readonly int MaxNewTokens;
        public readonly float Temperature;
        public readonly int TopK;
        public readonly float TopP;
        public readonly float RepetitionPenalty;

        // The code predictor samples the 15 residual codebooks and Qwen gives
        // it its own knobs ("subtalker_*").
        public readonly float SubTemperature;
        public readonly int SubTopK;
        public readonly float SubTopP;

        SamplingParams(int maxNewTokens, float temperature, int topK, float topP,
            float repetitionPenalty, float subTemperature, int subTopK, float subTopP)
        {
            MaxNewTokens = maxNewTokens;
            Temperature = temperature;
            TopK = topK;
            TopP = topP;
            RepetitionPenalty = repetitionPenalty;
            SubTemperature = subTemperature;
            SubTopK = subTopK;
            SubTopP = subTopP;
        }

        public static SamplingParams Default => new SamplingParams(
            2048, 0.9f, 50, 1f, 1.05f, 0.9f, 50, 1f);

        public static SamplingParams From(SpeechOptions options)
        {
            if (options == null)
                return Default;
            return new SamplingParams(
                options.MaxNewTokens,
                options.Temperature,
                options.TopK,
                options.TopP,
                options.RepetitionPenalty,
                options.SubTalkerTemperature,
                options.SubTalkerTopK,
                options.SubTalkerTopP);
        }
    }
}
