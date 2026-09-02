#if UNITY_EDITOR
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Runtime.Serialization;
using System.Text;
using Microsoft.ML.OnnxRuntime;
using QwenTTS.Internal;
using QwenTTS.Onnx;
using QwenLog = QwenTTS.Internal.QwenLog;

namespace QwenTTS.Editor
{
    /// <summary>
    /// Keeps OrtEnv and InferenceSession native allocations alive across an
    /// editor domain reload. Managed wrappers die with the AppDomain; the
    /// process-wide ONNX allocations do not, provided nobody calls
    /// OrtReleaseSession or OrtReleaseEnv on the way out.
    ///
    /// Worth the reflection because the talker graphs take ~10 s each to open
    /// and a script compile would otherwise pay that again. Embedding tables
    /// are deliberately *not* stashed — they re-read in well under a second,
    /// which is not worth a hand-packed blob of AllocHGlobal pointers.
    ///
    /// Editor-only by construction: a player never reloads the domain, so none
    /// of this exists outside the editor assembly.
    /// </summary>
    internal static class NativeSessionKeepAlive
    {
        const int Magic = 0x514B4131; // QKA1
        const int MaxSessions = 24;

        static readonly FieldInfo SessionHandleField = typeof(InferenceSession).GetField(
            "_nativeHandle", BindingFlags.Instance | BindingFlags.NonPublic);
        static readonly FieldInfo SessionDisposedField = typeof(InferenceSession).GetField(
            "_disposed", BindingFlags.Instance | BindingFlags.NonPublic);
        static readonly MethodInfo InitWithHandle = typeof(InferenceSession).GetMethod(
            "InitWithSessionHandle", BindingFlags.Instance | BindingFlags.NonPublic);
        static readonly ConstructorInfo OrtEnvCtor = typeof(OrtEnv).GetConstructor(
            BindingFlags.Instance | BindingFlags.NonPublic,
            null,
            new[] { typeof(IntPtr), typeof(OrtLoggingLevel) },
            null);
        static readonly FieldInfo OrtEnvInstanceField = typeof(OrtEnv).GetField(
            "_instance", BindingFlags.Static | BindingFlags.NonPublic);

        static bool _envInstalled;

        /// <summary>
        /// True when every reflected member resolved. ONNX Runtime is a
        /// third-party package and these are its private members, so an
        /// upgrade can break them — in which case Hold degrades to reloading
        /// rather than crashing.
        /// </summary>
        internal static bool IsSupported =>
            SessionHandleField != null && SessionDisposedField != null &&
            InitWithHandle != null && OrtEnvCtor != null && OrtEnvInstanceField != null;

        /// <summary>Host preference: survive the next reload rather than release.</summary>
        internal static bool KeepRequested
        {
            get
            {
                try { return File.Exists(KeepFlagPath()); }
                catch { return false; }
            }
            set
            {
                try
                {
                    if (value)
                        File.WriteAllText(KeepFlagPath(), "1");
                    else if (File.Exists(KeepFlagPath()))
                        File.Delete(KeepFlagPath());
                }
                catch (Exception ex)
                {
                    QwenLog.LogWarning("[QwenTTS] Keep-alive flag: " + ex.Message);
                }
            }
        }

        #region Orchestration

        /// <summary>
        /// Steals the loaded sessions out of the live engine and records their
        /// native handles somewhere the next domain can find them. Called from
        /// <c>AssemblyReloadEvents.beforeAssemblyReload</c>.
        /// </summary>
        internal static void StashBeforeReload()
        {
            if (!KeepRequested)
                return;
            if (!IsSupported)
            {
                QwenLog.LogWarning(
                    "[QwenTTS] Keep-alive is unavailable on this ONNX Runtime build " +
                    "(private members moved); models will reload after the compile.");
                return;
            }

            var engine = QwenTts.EngineOrNull;
            if (engine == null)
                return;

            var models = new List<ORTModel>();
            engine.CollectOnnxModels(models);

            var stolen = new List<(string key, IntPtr handle)>();
            foreach (var model in models)
            {
                if (!model.TryReleaseSessionForKeepAlive(out var key, out var session))
                    continue;
                var handle = DetachSessionHandle(session);
                if (handle != IntPtr.Zero)
                    stolen.Add((key, handle));
            }

            if (stolen.Count == 0)
                return;

            var env = DetachOrtEnv();
            Stash(env, stolen);
        }

        /// <summary>
        /// Rewraps the stashed handles and offers them to the next engine.
        /// Called from <c>afterAssemblyReload</c> and from the static hook so a
        /// fresh domain picks them up even without a reload event.
        /// </summary>
        internal static void RestoreAfterReload()
        {
            if (!IsSupported)
                return;
            if (!TryRestore())
                return;
            QwenLog.Log("[QwenTTS] Native ONNX sessions restored; awaiting adoption by the engine.");
        }

        #endregion

        #region Native plumbing

        static IntPtr DetachSessionHandle(InferenceSession session)
        {
            if (session == null || SessionHandleField == null)
                return IntPtr.Zero;
            var handle = (IntPtr)SessionHandleField.GetValue(session);
            SessionHandleField.SetValue(session, IntPtr.Zero);
            GC.SuppressFinalize(session);
            return handle;
        }

        static void Stash(IntPtr envHandle, List<(string key, IntPtr handle)> sessions)
        {
            ClearTokenFile();
            if (envHandle == IntPtr.Zero || sessions == null || sessions.Count == 0)
                return;
            if (sessions.Count > MaxSessions)
            {
                QwenLog.LogError("[QwenTTS] Keep-alive: too many sessions to stash.");
                return;
            }

            int bytes = 4 + 4 + 8 + 8 + 4;
            var keysUtf8 = new byte[sessions.Count][];
            for (int i = 0; i < sessions.Count; i++)
            {
                keysUtf8[i] = Encoding.UTF8.GetBytes(sessions[i].key ?? "");
                bytes += 8 + 4 + keysUtf8[i].Length;
            }

            IntPtr blob = Marshal.AllocHGlobal(bytes);
            IntPtr p = blob;
            WriteI32(ref p, Magic);
            WriteI32(ref p, Process.GetCurrentProcess().Id);
            WriteI64(ref p, Process.GetCurrentProcess().StartTime.ToUniversalTime().Ticks);
            WriteI64(ref p, envHandle.ToInt64());
            WriteI32(ref p, sessions.Count);
            for (int i = 0; i < sessions.Count; i++)
            {
                WriteI64(ref p, sessions[i].handle.ToInt64());
                WriteI32(ref p, keysUtf8[i].Length);
                Marshal.Copy(keysUtf8[i], 0, p, keysUtf8[i].Length);
                p = IntPtr.Add(p, keysUtf8[i].Length);
            }

            File.WriteAllText(TokenPath(), blob.ToInt64().ToString());
            QwenLog.Log($"[QwenTTS] Stashed {sessions.Count} ONNX session(s) across domain reload.");
            for (int i = 0; i < sessions.Count; i++)
                QwenLog.LogVerbose("[QwenTTS] stash " + sessions[i].key);
        }

        static bool TryRestore()
        {
            if (KeepAliveHandoff.HasSessions)
                return true;

            string path = TokenPath();
            if (!File.Exists(path))
                return false;
            string raw = File.ReadAllText(path).Trim();
            ClearTokenFile();
            if (string.IsNullOrEmpty(raw) || !long.TryParse(raw, out long addr) || addr == 0)
                return false;

            var blob = new IntPtr(addr);
            try
            {
                IntPtr p = blob;
                int magic = ReadI32(ref p);
                int pid = ReadI32(ref p);
                long startTicks = ReadI64(ref p);
                long envBits = ReadI64(ref p);
                int count = ReadI32(ref p);
                var proc = Process.GetCurrentProcess();
                // The blob is a raw process address, so it is only meaningful
                // inside the same process instance that wrote it.
                if (magic != Magic || pid != proc.Id ||
                    startTicks != proc.StartTime.ToUniversalTime().Ticks ||
                    count < 1 || count > MaxSessions)
                {
                    QwenLog.LogWarning("[QwenTTS] Keep-alive blob is stale; ignoring.");
                    return false;
                }

                if (!InstallOrtEnv(new IntPtr(envBits)))
                    return false;

                var pending = new Dictionary<string, InferenceSession>(count);
                for (int i = 0; i < count; i++)
                {
                    var sessionHandle = new IntPtr(ReadI64(ref p));
                    int keyLen = ReadI32(ref p);
                    if (keyLen < 0 || keyLen > 2048)
                        throw new InvalidOperationException("Keep-alive key length is invalid.");
                    var keyBytes = new byte[keyLen];
                    Marshal.Copy(p, keyBytes, 0, keyLen);
                    p = IntPtr.Add(p, keyLen);
                    string key = Encoding.UTF8.GetString(keyBytes);
                    var wrapped = WrapSession(sessionHandle);
                    if (wrapped == null)
                        throw new InvalidOperationException("Failed to wrap InferenceSession for " + key);
                    pending[key] = wrapped;
                }

                KeepAliveHandoff.OfferSessions(pending);
                QwenLog.Log($"[QwenTTS] Restored {pending.Count} ONNX session(s) after domain reload.");
                return true;
            }
            catch (Exception ex)
            {
                QwenLog.LogError("[QwenTTS] Keep-alive restore failed: " + ex.Message);
                return false;
            }
            finally
            {
                Marshal.FreeHGlobal(blob);
            }
        }

        static bool InstallOrtEnv(IntPtr envHandle)
        {
            if (_envInstalled && OrtEnv.IsCreated)
                return true;
            if (envHandle == IntPtr.Zero)
                return false;

            if (OrtEnv.IsCreated)
            {
                QwenLog.LogError(
                    "[QwenTTS] OrtEnv already created before keep-alive restore; " +
                    "stashed sessions belong to a different env and will not be adopted.");
                return false;
            }

            var env = (OrtEnv)OrtEnvCtor.Invoke(new object[]
            {
                envHandle, OrtLoggingLevel.ORT_LOGGING_LEVEL_WARNING
            });
            OrtEnvInstanceField.SetValue(null, new Lazy<OrtEnv>(() => env));
            _envInstalled = true;
            QwenLog.Log("[QwenTTS] Reattached OrtEnv after domain reload.");
            return true;
        }

        static IntPtr DetachOrtEnv()
        {
            if (!OrtEnv.IsCreated)
                return IntPtr.Zero;
            var env = OrtEnv.Instance();
            IntPtr handle = env.DangerousGetHandle();
            env.SetHandleAsInvalid();
            return handle;
        }

        static InferenceSession WrapSession(IntPtr nativeHandle)
        {
            if (nativeHandle == IntPtr.Zero || InitWithHandle == null)
                return null;
            var session = (InferenceSession)FormatterServices.GetUninitializedObject(typeof(InferenceSession));
            SessionDisposedField?.SetValue(session, false);
            InitWithHandle.Invoke(session, new object[] { nativeHandle });
            return session;
        }

        #endregion

        #region Token files

        static string KeepFlagPath() => TempPath("keep");

        static string TokenPath() => TempPath("ptr");

        static string TempPath(string extension) => Path.Combine(
            Path.GetTempPath(),
            "QwenTTS-KeepAlive-" + Process.GetCurrentProcess().Id + "." + extension);

        static void ClearTokenFile()
        {
            try
            {
                if (File.Exists(TokenPath()))
                    File.Delete(TokenPath());
            }
            catch (Exception ex)
            {
                QwenLog.LogWarning("[QwenTTS] Keep-alive token cleanup: " + ex.Message);
            }
        }

        static void WriteI32(ref IntPtr p, int value)
        {
            Marshal.WriteInt32(p, value);
            p = IntPtr.Add(p, 4);
        }

        static void WriteI64(ref IntPtr p, long value)
        {
            Marshal.WriteInt64(p, value);
            p = IntPtr.Add(p, 8);
        }

        static int ReadI32(ref IntPtr p)
        {
            int value = Marshal.ReadInt32(p);
            p = IntPtr.Add(p, 4);
            return value;
        }

        static long ReadI64(ref IntPtr p)
        {
            long value = Marshal.ReadInt64(p);
            p = IntPtr.Add(p, 8);
            return value;
        }

        #endregion
    }
}
#endif
