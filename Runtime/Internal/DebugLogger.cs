using UnityEngine;

namespace QwenTTS
{
    /// <summary>Verbosity for this package's own logging.</summary>
    public enum LogLevel
    {
        VERBOSE,
        INFO,
        WARNING,
        ERROR,

        /// <summary>Nothing, including errors. Diagnose with the return values instead.</summary>
        NONE,
    }
}

namespace QwenTTS.Internal
{
    /// <summary>
    /// Package logging. Named so it does not collide with a consumer's own
    /// <c>Logger</c> when both are in scope.
    /// </summary>
    internal static class QwenLog
    {
        public static LogLevel LogLevel { get; set; } = LogLevel.INFO;

        public static bool IsVerbose => LogLevel <= LogLevel.VERBOSE;

        public static void LogVerbose(string message)
        {
            if (LogLevel <= LogLevel.VERBOSE)
                Debug.Log(message);
        }

        public static void Log(string message)
        {
            if (LogLevel <= LogLevel.INFO)
                Debug.Log(message);
        }

        public static void LogWarning(string message)
        {
            if (LogLevel <= LogLevel.WARNING)
                Debug.LogWarning(message);
        }

        public static void LogError(string message)
        {
            if (LogLevel <= LogLevel.ERROR)
                Debug.LogError(message);
        }
    }
}
