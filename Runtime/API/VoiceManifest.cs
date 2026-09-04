using System.IO;
using System.Threading.Tasks;
using Newtonsoft.Json;

namespace QwenTTS
{
    /// <summary>
    /// <c>voice.json</c> — what a saved voice folder contains.
    ///
    /// The <c>Has*</c> flags exist so the manifest never names a file that is
    /// not there. The previous format always recorded a sample filename, and a
    /// consumer that trusted it would build a path to a missing wav.
    /// </summary>
    internal sealed class VoiceManifest
    {
        public const string FileName = "voice.json";
        public const string SampleFileName = "sample.wav";
        public const string ReferenceFileName = "reference.wav";

        [JsonProperty("clone")] public bool IsClone { get; set; }
        [JsonProperty("instruct")] public string Instruct { get; set; }
        [JsonProperty("referenceText")] public string ReferenceText { get; set; }
        [JsonProperty("language")] public string Language { get; set; }

        /// <summary>A rendered take is present as <see cref="SampleFileName"/>.</summary>
        [JsonProperty("hasSample")] public bool HasSample { get; set; }

        /// <summary>The clone reference audio is present as <see cref="ReferenceFileName"/>.</summary>
        [JsonProperty("hasReference")] public bool HasReference { get; set; }

        /// <summary>Rate of the reference before it was resampled to 24 kHz. 0 when unknown.</summary>
        [JsonProperty("referenceSourceSampleRate")] public int ReferenceSourceSampleRate { get; set; }

        public static VoiceManifest Read(string folder)
        {
            var path = Path.Combine(folder, FileName);
            if (!File.Exists(path))
                throw new FileNotFoundException("Voice manifest not found: " + path);
            var manifest = JsonConvert.DeserializeObject<VoiceManifest>(File.ReadAllText(path));
            if (manifest == null)
                throw new InvalidDataException("Could not parse " + path);
            if (string.IsNullOrWhiteSpace(manifest.Language))
                manifest.Language = QwenLanguages.Default;
            return manifest;
        }

        public Task WriteAsync(string folder)
        {
            var json = JsonConvert.SerializeObject(this, Formatting.Indented);
            var path = Path.Combine(folder, FileName);
#if UNITY_2021_2_OR_NEWER
            return File.WriteAllTextAsync(path, json);
#else
            return Task.Run(() => File.WriteAllText(path, json));
#endif
        }

        /// <summary>Full path to the sample, or null when there isn't one.</summary>
        public string SamplePath(string folder) =>
            HasSample ? Path.Combine(folder, SampleFileName) : null;
    }
}
