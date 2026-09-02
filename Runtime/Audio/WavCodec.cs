using System;
using UnityEngine;

namespace QwenTTS.Audio
{
    /// <summary>
    /// Minimal PCM WAV read/write, plus the conversion the engine needs.
    ///
    /// Reads the header rather than assuming it: a 24 kHz reference decoded as
    /// 16 kHz plays 1.5x slow, which is silent corruption of exactly the signal
    /// a clone is derived from.
    /// </summary>
    public static class WavCodec
    {
        /// <summary>
        /// Decodes PCM (8/16/24/32-bit) or IEEE float WAV into interleaved
        /// samples. False when the bytes are not a WAV this can read.
        /// </summary>
        public static bool TryDecode(byte[] data, out float[] samples, out int sampleRate, out int channels)
        {
            samples = Array.Empty<float>();
            sampleRate = 0;
            channels = 0;

            if (data == null || data.Length < 44)
                return false;
            if (data[0] != 'R' || data[1] != 'I' || data[2] != 'F' || data[3] != 'F')
                return false;
            if (data[8] != 'W' || data[9] != 'A' || data[10] != 'V' || data[11] != 'E')
                return false;

            int format = 0, bits = 0, dataOffset = -1, dataLength = 0;

            int pos = 12;
            while (pos + 8 <= data.Length)
            {
                string id = "" + (char)data[pos] + (char)data[pos + 1] + (char)data[pos + 2] + (char)data[pos + 3];
                int size = BitConverter.ToInt32(data, pos + 4);
                int body = pos + 8;
                if (size < 0 || body + size > data.Length)
                    size = data.Length - body;

                if (id == "fmt ")
                {
                    format = BitConverter.ToInt16(data, body);
                    channels = BitConverter.ToInt16(data, body + 2);
                    sampleRate = BitConverter.ToInt32(data, body + 4);
                    bits = BitConverter.ToInt16(data, body + 14);
                }
                else if (id == "data")
                {
                    dataOffset = body;
                    dataLength = size;
                }

                pos = body + size + (size & 1); // chunks are word aligned
            }

            if (dataOffset < 0 || channels <= 0 || sampleRate <= 0 || bits <= 0)
                return false;

            int bytesPerSample = bits / 8;
            int count = dataLength / bytesPerSample;
            var outBuf = new float[count];

            if (format == 3 && bits == 32)
            {
                for (int i = 0; i < count; i++)
                    outBuf[i] = BitConverter.ToSingle(data, dataOffset + i * 4);
            }
            else if (format == 1 || format == 0xFFFE)
            {
                switch (bits)
                {
                    case 8:
                        for (int i = 0; i < count; i++)
                            outBuf[i] = (data[dataOffset + i] - 128) / 128f;
                        break;
                    case 16:
                        for (int i = 0; i < count; i++)
                            outBuf[i] = BitConverter.ToInt16(data, dataOffset + i * 2) / 32768f;
                        break;
                    case 24:
                        for (int i = 0; i < count; i++)
                        {
                            int o = dataOffset + i * 3;
                            int v = data[o] | (data[o + 1] << 8) | ((sbyte)data[o + 2] << 16);
                            outBuf[i] = v / 8388608f;
                        }
                        break;
                    case 32:
                        for (int i = 0; i < count; i++)
                            outBuf[i] = BitConverter.ToInt32(data, dataOffset + i * 4) / 2147483648f;
                        break;
                    default:
                        return false;
                }
            }
            else
            {
                return false;
            }

            samples = outBuf;
            return true;
        }

        /// <summary>16-bit mono PCM WAV. Rounds rather than truncates, and clamps.</summary>
        public static byte[] Encode(float[] samples, int sampleRate)
        {
            if (samples == null)
                throw new ArgumentNullException(nameof(samples));
            if (sampleRate <= 0)
                throw new ArgumentOutOfRangeException(nameof(sampleRate));

            int dataBytes = samples.Length * 2;
            var bytes = new byte[44 + dataBytes];
            int p = 0;

            void PutAscii(string s) { for (int i = 0; i < s.Length; i++) bytes[p++] = (byte)s[i]; }
            void PutI32(int v) { BitConverter.GetBytes(v).CopyTo(bytes, p); p += 4; }
            void PutI16(short v) { BitConverter.GetBytes(v).CopyTo(bytes, p); p += 2; }

            PutAscii("RIFF");
            PutI32(36 + dataBytes);
            PutAscii("WAVE");
            PutAscii("fmt ");
            PutI32(16);
            PutI16(1);              // PCM
            PutI16(1);              // mono
            PutI32(sampleRate);
            PutI32(sampleRate * 2); // byte rate
            PutI16(2);              // block align
            PutI16(16);             // bits
            PutAscii("data");
            PutI32(dataBytes);

            for (int i = 0; i < samples.Length; i++)
            {
                // Round, don't truncate: a clone reference makes several of
                // these round trips and the codec's quantizer is sensitive
                // enough that truncation bias shifts reference codes.
                float clamped = Mathf.Clamp(samples[i], -1f, 1f);
                PutI16((short)Mathf.RoundToInt(clamped * 32767f));
            }

            return bytes;
        }

        /// <summary>Interleaved samples at any rate to mono 24 kHz.</summary>
        public static float[] ToMono24k(float[] interleaved, int sampleRate, int channels)
        {
            if (interleaved == null || interleaved.Length == 0)
                return Array.Empty<float>();

            float[] mono;
            if (channels <= 1)
            {
                mono = interleaved;
            }
            else
            {
                int frames = interleaved.Length / channels;
                mono = new float[frames];
                for (int i = 0; i < frames; i++)
                {
                    float sum = 0f;
                    for (int c = 0; c < channels; c++)
                        sum += interleaved[i * channels + c];
                    mono[i] = sum / channels;
                }
            }

            return sampleRate == 24000 ? mono : AudioResample.Resample(mono, sampleRate, 24000);
        }

        /// <summary>Decoded WAV bytes as an AudioClip at the rate the file declares.</summary>
        public static AudioClip ToAudioClip(byte[] wavBytes, string name = "QwenTtsWav")
        {
            if (!TryDecode(wavBytes, out float[] samples, out int rate, out int channels))
                return null;
            int frames = samples.Length / Math.Max(1, channels);
            var clip = AudioClip.Create(name, frames, channels, rate, false);
            clip.SetData(samples, 0);
            return clip;
        }
    }
}
