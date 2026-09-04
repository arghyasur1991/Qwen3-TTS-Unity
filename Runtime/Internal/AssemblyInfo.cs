using System.Runtime.CompilerServices;

// The editor assembly releases ONNX Runtime before a domain reload (sessions,
// then the environment) and draws the model-status window, both of which
// reach internal engine and session types. One-way: nothing in Runtime
// references the editor assembly.
[assembly: InternalsVisibleTo("QwenTTS.Editor")]

// Tests cover the pure helpers (WAV codec, resampler, clone-prompt format),
// which are internal because they are not part of the supported surface.
[assembly: InternalsVisibleTo("QwenTTS.Tests")]
