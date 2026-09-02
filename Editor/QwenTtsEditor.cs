#if UNITY_EDITOR
namespace QwenTTS.Editor
{
    /// <summary>
    /// Editor-only controls a host needs. Kept out of the runtime API on
    /// purpose: holding native allocations across a domain reload is a
    /// scripting-workflow concern, not something a shipped player can do.
    /// </summary>
    public static class QwenTtsEditor
    {
        /// <summary>
        /// Detach the ONNX sessions before a script compile and reattach them
        /// after, instead of paying ~22 s to reopen them. Off by default.
        /// </summary>
        public static bool HoldModelsAcrossReload
        {
            get => NativeSessionKeepAlive.KeepRequested;
            set => NativeSessionKeepAlive.KeepRequested = value;
        }

        /// <summary>
        /// False when this ONNX Runtime build has moved the private members the
        /// hold relies on. Hold then does nothing and models reload normally.
        /// </summary>
        public static bool HoldIsSupported => NativeSessionKeepAlive.IsSupported;
    }
}
#endif
