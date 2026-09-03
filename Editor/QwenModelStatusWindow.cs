#if UNITY_EDITOR
using System.IO;
using UnityEditor;
using UnityEngine;

namespace QwenTTS.Editor
{
    /// <summary>
    /// Shows where the package is looking for weights, what is installed, and
    /// what is currently resident. Read-only apart from Hold and Evict: the
    /// package does not copy or download models, so there is nothing to
    /// "deploy".
    /// </summary>
    internal sealed class QwenModelStatusWindow : EditorWindow
    {
        const double RefreshSeconds = 1.0;

        double _nextRefresh;
        CheckpointStatus _voiceDesign;
        CheckpointStatus _base;
        string _root;

        [MenuItem("Window/Qwen3 TTS/Model Status")]
        static void Open()
        {
            var window = GetWindow<QwenModelStatusWindow>();
            window.titleContent = new GUIContent("Qwen3 TTS");
            window.minSize = new Vector2(460, 320);
            window.Show();
        }

        void OnEnable() => Refresh();

        void Update()
        {
            if (EditorApplication.timeSinceStartup < _nextRefresh)
                return;
            _nextRefresh = EditorApplication.timeSinceStartup + RefreshSeconds;
            Refresh();
            Repaint();
        }

        // Everything drawn must come from these cached fields: OnGUI runs on
        // every repaint and file existence checks over ~40 paths per checkpoint
        // would make the window janky.
        void Refresh()
        {
            _root = Engine.QwenModelPaths.Root;
            _voiceDesign = QwenTts.GetStatus(QwenCheckpoint.VoiceDesign);
            _base = QwenTts.GetStatus(QwenCheckpoint.Base);
        }

        void OnGUI()
        {
            EditorGUILayout.Space();
            EditorGUILayout.LabelField("Model root", EditorStyles.boldLabel);
            EditorGUILayout.SelectableLabel(_root, EditorStyles.textField,
                GUILayout.Height(EditorGUIUtility.singleLineHeight));
            if (!Engine.QwenModelPaths.RootIsExplicit)
            {
                EditorGUILayout.HelpBox(
                    "Defaulting to StreamingAssets. Set QwenTtsSettings.ModelRoot before " +
                    "Initialize to keep 13+ GB of weights out of your build.",
                    MessageType.Info);
            }
            using (new EditorGUI.DisabledScope(!Directory.Exists(_root)))
            {
                if (GUILayout.Button("Reveal root"))
                    EditorUtility.RevealInFinder(_root);
            }

            EditorGUILayout.Space();
            DrawCheckpoint(_voiceDesign, "Invents a speaker from an instruct.");
            EditorGUILayout.Space();
            DrawCheckpoint(_base, "Clones a speaker from a reference recording.");

            EditorGUILayout.Space();
            using (new EditorGUI.DisabledScope(!QwenTts.IsInitialized))
            {
                if (GUILayout.Button("Unload everything"))
                    QwenTts.Unload();
            }
        }

        void DrawCheckpoint(CheckpointStatus status, string blurb)
        {
            if (status == null)
                return;

            using var box = new EditorGUILayout.VerticalScope(EditorStyles.helpBox);
            EditorGUILayout.LabelField(status.Checkpoint.ToString(), EditorStyles.boldLabel);
            EditorGUILayout.LabelField(blurb, EditorStyles.miniLabel);

            string size = status.InstalledBytes > 0
                ? $"{status.InstalledBytes / (1024f * 1024f * 1024f):0.00} GB"
                : "nothing found";
            EditorGUILayout.LabelField("Installed", status.Installed ? $"yes ({size})" : $"no ({size})");
            EditorGUILayout.LabelField("Resident", status.Loaded ? "yes (~12.9 GB)" : "no");

            if (!status.Installed)
            {
                int show = Mathf.Min(4, status.MissingFiles.Count);
                var lines = new string[show];
                for (int i = 0; i < show; i++)
                    lines[i] = status.MissingFiles[i];
                EditorGUILayout.HelpBox(
                    $"Missing {status.MissingFiles.Count} file(s), e.g.\n  " +
                    string.Join("\n  ", lines) +
                    "\n\nExport with Tools~/qwen3_tts_onnx/export_all.py.",
                    MessageType.Warning);
                return;
            }

            using (new EditorGUILayout.HorizontalScope())
            {
                using (new EditorGUI.DisabledScope(!QwenTts.IsInitialized || status.Loaded))
                {
                    if (GUILayout.Button("Warm up"))
                        _ = QwenTts.WarmUpAsync(status.Checkpoint);
                }
                using (new EditorGUI.DisabledScope(!status.Loaded))
                {
                    if (GUILayout.Button("Evict"))
                        QwenTts.Evict(status.Checkpoint);
                }
            }
        }
    }
}
#endif
