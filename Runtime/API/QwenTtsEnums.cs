namespace QwenTTS
{
    /// <summary>ONNX Runtime execution provider.</summary>
    public enum ExecutionProvider
    {
        /// <summary>
        /// CPU execution provider - universal compatibility, moderate performance
        /// </summary>
        CPU,
            
        /// <summary>
        /// CUDA execution provider - GPU acceleration for NVIDIA cards, high performance
        /// </summary>
        CUDA,
            
        /// <summary>
        /// CoreML execution provider - Apple Silicon/macOS acceleration, optimized for Apple hardware
        /// </summary>
        CoreML
    }

    /// <summary>
    /// Memory usage patterns for model loading
    /// </summary>
    public enum MemoryUsage
    {
        /// <summary>
        /// Load all models at startup for fastest inference. Higher memory usage.
        /// </summary>
        Performance,
        
        /// <summary>
        /// Load models on demand and keep them alive. Balanced approach.
        /// </summary>
        Balanced,
        
        /// <summary>
        /// Load models on demand and dispose after use. Lowest memory usage but slower.
        /// </summary>
        Optimal
    }
}
