using System.Runtime.CompilerServices;

// The editor assembly drives the domain-reload keep-alive, which needs to
// reach the engine's session wrappers. One-way: nothing in Runtime references
// the editor assembly.
[assembly: InternalsVisibleTo("QwenTTS.Editor")]

// Tests cover the pure helpers (WAV codec, resampler, clone-prompt format),
// which are internal because they are not part of the supported surface.
[assembly: InternalsVisibleTo("QwenTTS.Tests")]
