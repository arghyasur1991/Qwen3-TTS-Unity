using System;
using System.Threading;

namespace QwenTTS.Engine
{
    /// <summary>
    /// Opt-in streaming for one synthesis call.
    ///
    /// Default-constructed means "do not stream", which is what the
    /// non-streaming callers pass, so the ordinary path stays a plain
    /// generate-then-decode: no vocoder work inside the frame loop and no
    /// behaviour change for anyone who has not asked for chunks.
    /// </summary>
    internal readonly struct StreamRequest
    {
        readonly IProgress<SpeechChunk> _sink;
        readonly int _firstChunkFrames;
        readonly int _maxChunkFrames;

        public StreamRequest(IProgress<SpeechChunk> sink, int firstChunkFrames, int maxChunkFrames)
        {
            _sink = sink;
            _firstChunkFrames = firstChunkFrames;
            _maxChunkFrames = maxChunkFrames;
        }

        public bool IsEnabled => _sink != null;

        /// <summary>
        /// Attaches a frame sink to <paramref name="talker"/> for the life of
        /// the returned scope, and flushes the last chunk on dispose. The
        /// caller must already hold the checkpoint's lock:
        /// <see cref="LanguageModel.FrameSink"/> is shared state guarded by it.
        /// </summary>
        public Binding Begin(LanguageModel talker, QwenVocoderModel vocoder,
            long[,,] prefixCodes, CancellationToken token)
        {
            if (!IsEnabled)
                return default;
            return new Binding(talker, new StreamingVocode(
                vocoder, _sink, _firstChunkFrames, _maxChunkFrames, prefixCodes, token));
        }

        internal readonly struct Binding : IDisposable
        {
            readonly LanguageModel _talker;
            readonly StreamingVocode _vocode;

            public Binding(LanguageModel talker, StreamingVocode vocode)
            {
                _talker = talker;
                _vocode = vocode;
                talker.LastFrames = null;
                talker.FrameSink = vocode.OnFrame;
            }

            public void Dispose()
            {
                if (_talker == null)
                    return;
                try
                {
                    // Generation stops on EOS or the token cap, either of which
                    // can leave frames that never reached a chunk boundary.
                    var frames = _talker.LastFrames;
                    if (frames != null)
                        _vocode.Finish(frames);
                }
                finally
                {
                    _talker.FrameSink = null;
                    _talker.LastFrames = null;
                }
            }
        }
    }
}
