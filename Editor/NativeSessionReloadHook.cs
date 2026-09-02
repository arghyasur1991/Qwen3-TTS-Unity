#if UNITY_EDITOR
using UnityEditor;

namespace QwenTTS.Editor
{
    /// <summary>
    /// Hooks the domain reload so a script compile does not throw away
    /// multi-gigabyte ONNX sessions. Only acts when the host has asked to hold
    /// them (see <c>NativeSessionKeepAlive.KeepRequested</c>).
    /// </summary>
    [InitializeOnLoad]
    internal static class NativeSessionReloadHook
    {
        static NativeSessionReloadHook()
        {
            AssemblyReloadEvents.beforeAssemblyReload += NativeSessionKeepAlive.StashBeforeReload;
            AssemblyReloadEvents.afterAssemblyReload += NativeSessionKeepAlive.RestoreAfterReload;
            // Also on first load of a fresh domain, where the reload event for
            // this domain has already been and gone.
            NativeSessionKeepAlive.RestoreAfterReload();
        }
    }
}
#endif
