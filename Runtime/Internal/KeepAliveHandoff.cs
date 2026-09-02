using System.Collections.Generic;
using Microsoft.ML.OnnxRuntime;

namespace QwenTTS.Internal
{
    /// <summary>
    /// Where the editor hands recovered ONNX sessions back to the engine after
    /// a domain reload.
    ///
    /// The runtime side is deliberately this small: a couple of static slots
    /// with no reflection and no platform assumptions, inert in a player
    /// because nothing ever fills them. All of the machinery — stashing native
    /// handles past the AppDomain teardown, reattaching OrtEnv — lives in the
    /// editor assembly, which is the only place a domain reload happens.
    /// </summary>
    internal static class KeepAliveHandoff
    {
        static Dictionary<string, InferenceSession> _sessions;

        /// <summary>Sessions are waiting to be adopted by a rebuilt engine.</summary>
        internal static bool HasSessions => _sessions != null && _sessions.Count > 0;

        /// <summary>Called by the editor after it reattaches native handles.</summary>
        internal static void OfferSessions(Dictionary<string, InferenceSession> sessions)
        {
            _sessions = sessions != null && sessions.Count > 0 ? sessions : null;
        }

        /// <summary>Claims the offered sessions. The caller owns them from here.</summary>
        internal static Dictionary<string, InferenceSession> TakeSessions()
        {
            var s = _sessions;
            _sessions = null;
            return s;
        }

        internal static void Clear() => _sessions = null;
    }
}
