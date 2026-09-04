#if UNITY_EDITOR
using System;
using UnityEditor;
using QwenTTS.Onnx;
using QwenLog = QwenTTS.Internal.QwenLog;

namespace QwenTTS.Editor
{
    /// <summary>
    /// Releases ONNX Runtime deterministically before a domain reload.
    ///
    /// A reload destroys the managed wrappers without releasing what they own.
    /// Sessions, the environment, and the unmanaged buffer the logging sink
    /// reads would each be orphaned once per compile — the environment
    /// unconditionally, since nothing runs a finalizer for it on the way out.
    /// A few gigabytes of session per orphan is the part that matters.
    ///
    /// Order is load-bearing: sessions hold the environment's logger, so they
    /// go first, and the buffer goes last because the sink dereferences it
    /// while the environment lives.
    ///
    /// This blocks the reload until any in-flight generation finishes, because
    /// unloading takes the engine's per-checkpoint locks. Unity is already
    /// waiting at that point, and the alternative — pulling the environment out
    /// from under a running session — is worse.
    /// </summary>
    [InitializeOnLoad]
    internal static class OnnxReloadCleanup
    {
        static OnnxReloadCleanup()
        {
            AssemblyReloadEvents.beforeAssemblyReload += ReleaseBeforeReload;
        }

        static void ReleaseBeforeReload()
        {
            try
            {
                bool hadEngine = QwenTts.IsInitialized;
                QwenTts.Unload();
                ORTModel.ReleaseEnvironment();
                if (hadEngine)
                    QwenLog.Log("[QwenTTS] Released ONNX sessions and environment before reload.");
            }
            catch (Exception e)
            {
                // Never let cleanup take the reload down with it.
                QwenLog.LogWarning("[QwenTTS] Pre-reload cleanup: " + e.Message);
            }
        }
    }
}
#endif
