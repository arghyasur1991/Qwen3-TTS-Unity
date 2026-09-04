using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Threading.Tasks;
using UnityEngine;

namespace QwenTTS.Onnx
{
    using QwenTTS.Engine;
    using QwenTTS.Internal;

    /// <summary>
    /// Model load policy for controlling when models are loaded and disposed
    /// </summary>
    internal enum ModelLoadPolicy
    {
        /// <summary>
        /// Load model at startup. Never dispose until shutdown.
        /// </summary>
        OnStartup,
        
        /// <summary>
        /// Load model on first use. Keep alive after loading.
        /// </summary>
        OnDemandKeepAlive,
        
        /// <summary>
        /// Load model on each use. Dispose after use.
        /// </summary>
        OnDemand
    }

    /// <summary>
    /// Base ONNX model wrapper: load policy, execution provider, session lifetime.
    /// Lazy background load with a cached session, input/output name
    /// management, and a shared ONNX Runtime environment whose diagnostics are
    /// routed into the Unity console.
    /// </summary>
    internal abstract class ORTModel : IDisposable
    {
        #region Private Fields
        
        private readonly ModelConfig _config;
        private InferenceSession _session;
        private List<string> _inputNames = new();
        protected Task<InferenceSession> _loadTask = null;
        private bool _disposed = false;
        private ModelLoadPolicy _loadPolicy = ModelLoadPolicy.OnDemandKeepAlive;
        
        // Static memory usage configuration
        private static MemoryUsage _memoryUsage = MemoryUsage.Balanced;
        
        // Static logging configuration
        private static bool _loggingInitialized = false;
        private static IntPtr _loggingParam = IntPtr.Zero;
        private static OrtLoggingLevel _ortLogLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_WARNING;
        
        #endregion

        #region Protected Properties

        /// <summary>
        /// Gets whether the model has been successfully initialized.
        /// </summary>
        public bool IsInitialized { get; protected set; } = false;

        /// <summary>
        /// Gets the loading task for the model. Can be awaited to ensure model is loaded.
        /// </summary>
        public Task LoadTask => _loadTask;

        /// <summary>
        /// Gets the current ONNX Runtime logging level.
        /// </summary>
        protected static OrtLoggingLevel OrtLogLevel => _ortLogLevel;

        protected enum Precision
        {
            FP32,
            FP16,
            Int8
        }

        #endregion

        #region Constructor

        /// <summary>
        /// Initializes a new instance of the ORTModel class with the specified configuration.
        /// </summary>
        /// <param name="modelName">The name of the model file (without extension)</param>
        /// <param name="modelFolder">The folder containing the model (from QwenModelPaths)</param>
        /// <param name="precision">Weight precision; selects the _fp16 / _int8 file variant</param>
        /// <param name="executionProvider">The execution provider for the model</param>
        protected ORTModel(
            string modelName, 
            string modelFolder, 
            Precision precision = Precision.FP32,
            ExecutionProvider executionProvider = ExecutionProvider.CPU,
            bool deferLoad = false)
        {
            if (string.IsNullOrEmpty(modelName))
                throw new ArgumentNullException(nameof(modelName));
            if (modelFolder == null)
                throw new ArgumentNullException(nameof(modelFolder));

            _config = new ModelConfig
            {
                ModelName = modelName,
                Precision = precision,
                ExecutionProvider = executionProvider,
                // Resolved through QwenModelPaths so a shipped player can put
                // ~8 GB of weights per checkpoint somewhere other than StreamingAssets.
                ModelPath = Path.Combine(QwenModelPaths.Root, modelFolder)
            };
            
            // Set load policy based on global memory usage setting
            _loadPolicy = _memoryUsage switch
            {
                MemoryUsage.Performance => ModelLoadPolicy.OnStartup,
                MemoryUsage.Balanced => ModelLoadPolicy.OnDemandKeepAlive,
                MemoryUsage.Optimal => ModelLoadPolicy.OnDemand,
                _ => ModelLoadPolicy.OnDemandKeepAlive
            };

            // Multi-GB talker graphs must not all load at factory init even in Performance mode.
            if (deferLoad && _loadPolicy == ModelLoadPolicy.OnStartup)
                _loadPolicy = ModelLoadPolicy.OnDemandKeepAlive;
            
            // Start loading immediately in Performance mode
            if (_loadPolicy == ModelLoadPolicy.OnStartup)
            {
                StartLoadingAsync();
            }
        }

        #endregion
        
        #region Public Methods
        
        /// <summary>
        /// Sets the execution provider for the ONNX model.
        /// </summary>
        /// <param name="executionProvider">The execution provider to use for the model.</param>
        public void SetExecutionProvider(ExecutionProvider executionProvider)
        {
            _config.ExecutionProvider = executionProvider;
        }
        
        #endregion

        #region Public Methods - Input Loading

        /// <summary>
        /// Starts the asynchronous loading operation.
        /// </summary>
        /// <returns>A task that represents the asynchronous loading operation</returns>
        public void StartLoadingAsync()
        {
            if (IsInitialized || _loadTask != null)
                return;
            _disposed = false;
            _loadTask = BackgroundWork.Run(LoadSession);
        }

        #endregion

        #region Static Methods - Environment Management

        /// <summary>
        /// Initializes the ONNX Runtime environment with the specified logging level.
        /// </summary>
        /// <param name="logLevel">The logging level for ONNX Runtime operations</param>
        public static void InitializeEnvironment(LogLevel logLevel = LogLevel.WARNING)
        {
            _ortLogLevel = logLevel switch
            {
                LogLevel.ERROR => OrtLoggingLevel.ORT_LOGGING_LEVEL_ERROR,
                LogLevel.WARNING => OrtLoggingLevel.ORT_LOGGING_LEVEL_WARNING,
                LogLevel.INFO => OrtLoggingLevel.ORT_LOGGING_LEVEL_INFO,
                LogLevel.VERBOSE => OrtLoggingLevel.ORT_LOGGING_LEVEL_VERBOSE,
                _ => OrtLoggingLevel.ORT_LOGGING_LEVEL_WARNING
            };
            InitializeOnnxLogging();
        }

        /// <summary>
        /// Sets the global memory usage mode for all models.
        /// Must be called before any models are created.
        /// </summary>
        /// <param name="memoryUsage">The memory usage mode to use</param>
        public static void SetMemoryUsage(MemoryUsage memoryUsage)
        {
            _memoryUsage = memoryUsage;
            QwenLog.Log($"[ORTModel] Memory usage mode set to: {memoryUsage}");
        }

        /// <summary>
        /// Gets the current memory usage mode.
        /// </summary>
        public static MemoryUsage CurrentMemoryUsage => _memoryUsage;

        #endregion

        #region Protected Methods - Utilities

        /// <summary>
        /// Sets the full path to a model file within the model root.
        /// </summary>
        /// <param name="modelFolder">The subfolder for the model</param>
        /// <param name="modelName">The model name</param>
        protected string GetModelPath(string modelName)
        {
            if (_config.Precision == Precision.FP16)
            {
                modelName = $"{modelName}_fp16";
            }
            else if (_config.Precision == Precision.Int8)
            {
                modelName = $"{modelName}_int8";
            }
            return Path.Combine(
                _config.ModelPath,
                $"{modelName}.onnx");
        }

        /// <summary>
        /// Creates optimized SessionOptions for ONNX Runtime.
        /// </summary>
        /// <returns>A configured SessionOptions object</returns>
        /// <summary>
        /// Intra-op thread count applied to every session. 0 keeps ORT's
        /// default. Set from <see cref="QwenTtsSettings.IntraOpThreads"/>.
        /// </summary>
        internal static int IntraOpThreads;

        protected static SessionOptions CreateSessionOptions()
        {
            var options = new SessionOptions
            {
                LogSeverityLevel = _ortLogLevel,
                GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL
            };

            if (IntraOpThreads > 0)
                options.IntraOpNumThreads = IntraOpThreads;

            return options;
        }

        /// <summary>
        /// Blocks until the session is loaded. Call from a worker thread (or after
        /// <see cref="StartLoadingAsync"/> has finished). Loading uses
        /// <see cref="TaskScheduler.Default"/> so it does not run on the Unity main thread.
        /// </summary>
        protected void EnsureLoaded()
        {
            StartLoadingAsync();
            if (_loadTask != null && !_loadTask.IsCompleted)
                _loadTask.GetAwaiter().GetResult();
            if (_session == null)
                throw new InvalidOperationException($"[{_config.ModelName}] Session failed to load.");
        }

        /// <summary>Loaded session. Call EnsureLoaded first.</summary>
        protected InferenceSession Session
        {
            get
            {
                if (_session == null)
                    throw new InvalidOperationException($"[{_config.ModelName}] Session is not loaded.");
                return _session;
            }
        }

        protected IReadOnlyList<string> InputNames
        {
            get
            {
                EnsureLoaded();
                return _inputNames;
            }
        }

        protected string ModelName => _config.ModelName;

        protected string ModelFilePath => GetModelPath(_config.ModelName);


        internal bool HasLoadedSession => _session != null && !_disposed;


        /// <summary>
        /// Sets the logging parameter context for ONNX Runtime operations.
        /// </summary>
        /// <param name="modelName">The name of the model currently being processed</param>
        protected static void SetLoggingParam(string modelName) => SetLogContext(modelName);

        /// <summary>
        /// Same as <see cref="SetLoggingParam"/>, reachable by another library
        /// sharing this process's ONNX Runtime environment. See
        /// <c>QwenTts.SetOnnxLogContext</c>.
        /// </summary>
        internal static void SetLogContext(string modelName)
        {
            if (string.IsNullOrEmpty(modelName))
                return;
            // Nothing to attribute through if someone else created the
            // environment; their sink reads their own buffer, not this one.
            if (_loggingParam == IntPtr.Zero)
                return;

            Marshal.StructureToPtr(new LoadingInfo { ModelName = modelName }, _loggingParam, false);
        }

        #endregion

        #region Private Methods

        /// <summary>
        /// Asynchronously loads the ONNX model and initializes input/output metadata.
        /// </summary>
        /// <returns>A task containing the loaded InferenceSession</returns>
        private InferenceSession LoadSession()
        {
            string modelPath = GetModelPath(_config.ModelName);
            if (!File.Exists(modelPath))
            {
                QwenLog.LogError($"[{_config.ModelName}] Model file not found: {modelPath}");
                throw new FileNotFoundException($"Model file not found: {modelPath}");
            }
            QwenLog.Log($"[{_config.ModelName}] Loading model: {_config.ModelName}");
            // Attribute the native lines the session open itself produces —
            // arena growth, provider selection, graph optimisation. Without
            // this they arrive with an empty model name.
            SetLogContext(_config.ModelName);

            try
            {
                var options = CreateSessionOptions();

                if (_config.ExecutionProvider == ExecutionProvider.CoreML)
                {
                    LoadModelWithCoreML(modelPath, options);
                }
                else if (_config.ExecutionProvider == ExecutionProvider.CUDA)
                {
                    LoadModelWithCUDA(modelPath, options);
                }
                else
                {
                    _session = new InferenceSession(modelPath, options);
                }

                _inputNames = _session.InputMetadata.Keys.ToList();
    
                IsInitialized = true;
                QwenLog.Log($"[{_config.ModelName}] Successfully loaded model: {modelPath}");
                return _session;
            }
            catch (Exception ex)
            {
                QwenLog.LogError($"[{_config.ModelName}] Failed to load model: {ex.Message}");
                IsInitialized = false;
                throw;
            }
        }

        /// <summary>
        /// Releases the ONNX Runtime environment and the logging buffer that
        /// belongs to it, leaving both re-creatable.
        ///
        /// Only the editor needs this, and only because a domain reload
        /// destroys the managed wrappers without releasing the native objects
        /// behind them: the environment and the unmanaged buffer its sink reads
        /// would both be orphaned once per reload. <c>OrtEnv</c> is a
        /// <c>SafeHandle</c> whose release resets the singleton, so the next
        /// domain gets a fresh environment with a valid sink.
        ///
        /// **Dispose sessions first.** A live session holds the environment's
        /// logger; releasing the environment underneath one is undefined.
        /// </summary>
        internal static void ReleaseEnvironment()
        {
            try
            {
                if (OrtEnv.IsCreated)
                    OrtEnv.Instance().Dispose();
            }
            catch (Exception e)
            {
                QwenLog.LogWarning("[ORTModel] Releasing the ONNX environment: " + e.Message);
            }

            // After the environment, never before: the sink dereferences this.
            if (_loggingParam != IntPtr.Zero)
            {
                Marshal.FreeHGlobal(_loggingParam);
                _loggingParam = IntPtr.Zero;
            }
            _loggingInitialized = false;
        }

        /// <summary>
        /// Creates the ONNX Runtime environment and routes its diagnostics into
        /// Unity.
        ///
        /// The sink is a managed delegate handed to native code as a function
        /// pointer, which is only safe as long as nothing survives an editor
        /// domain reload still holding this environment: the thunk belongs to
        /// the domain, and ONNX Runtime has no API to replace an environment's
        /// sink after creation. The editor assembly therefore releases every
        /// session and then the environment itself (<see cref="ReleaseEnvironment"/>)
        /// before each reload, so nothing native outlives the delegate and
        /// editor and player share one logging path.
        ///
        /// ONNX Runtime allows one environment per process, so whichever
        /// library creates it owns the sink for everyone. <see cref="SetLogContext"/>
        /// is how another library attributes its own model to a native line.
        /// </summary>
        private static void InitializeOnnxLogging()
        {
            if (_loggingInitialized)
                return;

            if (Application.platform == RuntimePlatform.IPhonePlayer)
            {
                _loggingInitialized = true;
                return;
            }

            if (OrtEnv.IsCreated)
            {
                QwenLog.Log("[ORTModel] ONNX Runtime environment already created");
                _loggingInitialized = true;
                return;
            }

            try
            {
                // _loggingParam carries the name of whichever model is
                // currently executing, so a native line can be attributed;
                // SetLoggingParam updates it before each load and run.
                _loggingParam = Marshal.AllocHGlobal(Marshal.SizeOf(typeof(LoadingInfo)));
                var options = new EnvironmentCreationOptions
                {
                    logLevel = _ortLogLevel,
                    logId = "QwenTTS",
                    loggingFunction = UnityOnnxLoggingCallback,
                    loggingParam = _loggingParam,
                };
                OrtEnv.CreateInstanceWithOptions(ref options);
                QwenLog.Log(
                    $"[ORTModel] ONNX Runtime environment ready with Unity logging (LogLevel: {_ortLogLevel})");
                _loggingInitialized = true;
            }
            catch (Exception e)
            {
                QwenLog.LogError($"[ORTModel] Failed to initialize ONNX Runtime: {e.Message}");
                _loggingInitialized = true;
            }
        }

        /// <summary>
        /// Unity logging callback for ONNX Runtime.
        /// </summary>
        private static void UnityOnnxLoggingCallback(IntPtr param, 
                                                   OrtLoggingLevel severity, 
                                                   string category, 
                                                   string logId, 
                                                   string codeLocation, 
                                                   string message)
        {
            if (param == IntPtr.Zero || _loggingParam == IntPtr.Zero)
                return;
                
            var loadingInfo = (LoadingInfo)Marshal.PtrToStructure(param, typeof(LoadingInfo));
            string formattedMessage = FormatOnnxLogMessage(severity, category, logId, codeLocation, message, loadingInfo.ModelName);

            switch (severity)
            {
                case OrtLoggingLevel.ORT_LOGGING_LEVEL_VERBOSE:
                case OrtLoggingLevel.ORT_LOGGING_LEVEL_INFO:
                    QwenLog.Log(formattedMessage);
                    break;
                    
                case OrtLoggingLevel.ORT_LOGGING_LEVEL_WARNING:
                    QwenLog.LogWarning(formattedMessage);
                    break;
                    
                case OrtLoggingLevel.ORT_LOGGING_LEVEL_ERROR:
                case OrtLoggingLevel.ORT_LOGGING_LEVEL_FATAL:
                    QwenLog.LogError(formattedMessage);
                    break;
                    
                default:
                    QwenLog.Log(formattedMessage);
                    break;
            }
        }

        /// <summary>
        /// Formats ONNX Runtime log messages for Unity console.
        /// </summary>
        private static string FormatOnnxLogMessage(OrtLoggingLevel severity, 
                                                 string category, 
                                                 string logId, 
                                                 string codeLocation, 
                                                 string message,
                                                 string modelName)
        {
            string severityStr = severity switch
            {
                OrtLoggingLevel.ORT_LOGGING_LEVEL_VERBOSE => "VERBOSE",
                OrtLoggingLevel.ORT_LOGGING_LEVEL_INFO => "INFO", 
                OrtLoggingLevel.ORT_LOGGING_LEVEL_WARNING => "WARN",
                OrtLoggingLevel.ORT_LOGGING_LEVEL_ERROR => "ERROR",
                OrtLoggingLevel.ORT_LOGGING_LEVEL_FATAL => "FATAL",
                _ => "UNKNOWN"
            };

            string cleanCategory = !string.IsNullOrEmpty(category) ? $"[{category}]" : "";
            string cleanCodeLocation = !string.IsNullOrEmpty(codeLocation) ? $" ({codeLocation})" : "";
            
            return $"[ONNX-{severityStr}][{modelName}]{cleanCategory} {message}{cleanCodeLocation}";
        }

        #endregion

        #region Private Methods - CoreML Support


        /// <summary>
        /// Loads an ONNX model with CoreML acceleration and comprehensive error handling.
        /// This method configures CoreML provider with caching support, handles cache corruption recovery,
        /// and provides fallback mechanisms for maximum compatibility across different Apple devices.
        /// </summary>
        /// <param name="modelPath">The file path to the ONNX model</param>
        /// <param name="sessionOptions">The base session options to configure with CoreML provider</param>
        private void LoadModelWithCoreML(string modelPath, SessionOptions sessionOptions)
        {
            try
            {
                // Configure CoreML provider with caching support using dictionary API
                string cacheDirectory = GetCoreMLCacheDirectory();
                
                // Ensure cache directory exists and is writable
                EnsureCacheDirectoryExists(cacheDirectory);
                
                var coremlOptions = new Dictionary<string, string>
                {
                    ["ModelFormat"] = "MLProgram",
                    ["MLComputeUnits"] = "CPUAndGPU",
                    ["RequireStaticInputShapes"] = "0",
                    ["EnableOnSubgraphs"] = "1",
                    // Advanced options for optimization
                    // ["SpecializationStrategy"] = "FastPrediction",
                    // ["AllowLowPrecisionAccumulationOnGPU"] = "1",
                    // ["ProfileComputePlan"] = "1"
                };
                
                if (!string.IsNullOrEmpty(cacheDirectory))
                {
                    coremlOptions["ModelCacheDirectory"] = cacheDirectory;
                }
                
                sessionOptions.AppendExecutionProvider("CoreML", coremlOptions);
                QwenLog.Log($"[ModelUtils] CoreML provider configured with caching (cache: {cacheDirectory})");
                
                // Try creating the session - if it fails due to cache corruption, retry
                try
                {
                    _session = new InferenceSession(modelPath, sessionOptions);
                    QwenLog.Log($"[ModelUtils] Successfully loaded model with CoreML provider: {modelPath}");
                }
                catch (Exception sessionException)
                {
                    if (sessionException.Message.Contains("Manifest.json") || 
                        sessionException.Message.Contains("coreml_cache") ||
                        sessionException.Message.Contains("manifest does not exist"))
                    {
                        QwenLog.LogWarning($"[ModelUtils] CoreML cache corruption detected. Retrying: {sessionException.Message}");
                        _session = new InferenceSession(modelPath, sessionOptions);
                        QwenLog.Log($"[ModelUtils] Successfully loaded model with CoreML provider after retrying: {modelPath}");
                    }
                    else
                    {
                        throw; // Re-throw if it's not a cache-related issue
                    }
                }
            }
            catch (Exception e)
            {
                QwenLog.LogWarning($"[ModelUtils] CoreML provider configuration failed: {e.Message}");
                
                // Fallback to old CoreML flags approach for compatibility
                try
                {
                    var fallbackOptions = CreateSessionOptions();
                    fallbackOptions.AppendExecutionProvider_CoreML(
                        CoreMLFlags.COREML_FLAG_USE_CPU_AND_GPU | 
                        CoreMLFlags.COREML_FLAG_CREATE_MLPROGRAM |
                        CoreMLFlags.COREML_FLAG_ENABLE_ON_SUBGRAPH);
                    
                    _session = new InferenceSession(modelPath, fallbackOptions);
                    QwenLog.Log("[ModelUtils] Using fallback CoreML provider (no caching)");
                }
                catch (Exception fallbackException)
                {
                    QwenLog.LogWarning($"[ModelUtils] CoreML fallback also failed: {fallbackException.Message}. Using CPU provider.");
                }
            }
        }

        /// <summary>
        /// Gets the cache directory for CoreML compiled models.
        ///
        /// Lives under <see cref="Application.persistentDataPath"/>, which is
        /// writable everywhere this package runs. <see cref="Application.dataPath"/>
        /// would be a consumer's Assets folder in the editor — where a compiled
        /// model would be imported as an asset — or a read-only PackageCache,
        /// and is not writable at all in a player.
        /// </summary>
        /// <returns>The full path to the CoreML cache directory</returns>
        private string GetCoreMLCacheDirectory()
        {
            return Path.Combine(Application.persistentDataPath, "QwenTTS", "coreml_cache");
        }

        /// <summary>
        /// Ensures the CoreML cache directory exists and is writable with proper error handling.
        /// This method creates the cache directory structure if it doesn't exist and handles
        /// permission and filesystem errors gracefully.
        /// </summary>
        /// <param name="cacheDirectory">The cache directory path to create and validate</param>
        private void EnsureCacheDirectoryExists(string cacheDirectory)
        {
            if (string.IsNullOrEmpty(cacheDirectory))
                return;
                
            try
            {
                if (!Directory.Exists(cacheDirectory))
                {
                    Directory.CreateDirectory(cacheDirectory);
                    QwenLog.Log($"[ModelUtils] Created CoreML cache directory: {cacheDirectory}");
                }
            }
            catch (Exception e)
            {
                QwenLog.LogWarning($"[ModelUtils] Failed to create cache directory {cacheDirectory}: {e.Message}");
            }
        }
        #endregion
        
        #region Private Methods - CUDA Support


        /// <summary>
        /// Loads an ONNX model with CUDA acceleration and comprehensive error handling.
        /// This method configures CUDA provider with caching support, handles cache corruption recovery,
        /// and provides fallback mechanisms for maximum compatibility across different Apple devices.
        /// </summary>
        /// <param name="modelPath">The file path to the ONNX model</param>
        /// <param name="sessionOptions">The base session options to configure with CUDA provider</param>
        private void LoadModelWithCUDA(string modelPath, SessionOptions sessionOptions)
        {
            try
            {
                // Configure CUDA provider with caching support using dictionary API
                string cacheDirectory = GetCUDACacheDirectory();
                
                // Ensure cache directory exists and is writable
                EnsureCUDACacheDirectoryExists(cacheDirectory);
                
                
                sessionOptions.AppendExecutionProvider_CUDA(0);
                QwenLog.Log($"[ModelUtils] CUDA provider configured with caching (cache: {cacheDirectory})");
                
                // Try creating the session - if it fails due to cache corruption, retry
                try
                {
                    _session = new InferenceSession(modelPath, sessionOptions);
                    QwenLog.Log($"[ModelUtils] Successfully loaded model with CUDA provider: {modelPath}");
                }
                catch (Exception sessionException)
                {
                    if (sessionException.Message.Contains("Manifest.json") || 
                        sessionException.Message.Contains("cuda_cache") ||
                        sessionException.Message.Contains("manifest does not exist"))
                    {
                        QwenLog.LogWarning($"[ModelUtils] CUDA cache corruption detected. Retrying: {sessionException.Message}");
                        _session = new InferenceSession(modelPath, sessionOptions);
                        QwenLog.Log($"[ModelUtils] Successfully loaded model with CUDA provider after retrying: {modelPath}");
                    }
                    else
                    {
                        throw; // Re-throw if it's not a cache-related issue
                    }
                }
            }
            catch (Exception e)
            {
                QwenLog.LogWarning($"[ModelUtils] CUDA provider configuration failed: {e.Message}");
                
                // Fallback to old CUDA flags approach for compatibility
                try
                {
                    var fallbackOptions = CreateSessionOptions();
                    
                    _session = new InferenceSession(modelPath, fallbackOptions);
                    QwenLog.Log("[ModelUtils] Using fallback CUDA provider (no caching)");
                }
                catch (Exception fallbackException)
                {
                    QwenLog.LogWarning($"[ModelUtils] CUDA fallback also failed: {fallbackException.Message}. Using CPU provider.");
                }
            }
        }

        /// <summary>
        /// Gets the cache directory for CUDA compiled models with automatic path resolution.
        /// This method determines the best location for CUDA model caching based on configuration
        /// and platform-specific storage locations for optimal performance and persistence.
        /// </summary>
        /// <returns>The full path to the CUDA cache directory</returns>
        private string GetCUDACacheDirectory()
        {
            return Path.Combine(Application.dataPath, "Models", "cuda_cache");
        }

        /// <summary>
        /// Ensures the CUDA cache directory exists and is writable with proper error handling.
        /// This method creates the cache directory structure if it doesn't exist and handles
        /// permission and filesystem errors gracefully.
        /// </summary>
        /// <param name="cacheDirectory">The cache directory path to create and validate</param>
        private void EnsureCUDACacheDirectoryExists(string cacheDirectory)
        {
            if (string.IsNullOrEmpty(cacheDirectory))
                return;
                
            try
            {
                if (!Directory.Exists(cacheDirectory))
                {
                    Directory.CreateDirectory(cacheDirectory);
                    QwenLog.Log($"[ModelUtils] Created CUDA cache directory: {cacheDirectory}");
                }
            }
            catch (Exception e)
            {
                QwenLog.LogWarning($"[ModelUtils] Failed to create cache directory {cacheDirectory}: {e.Message}");
            }
        }
        #endregion

        #region Private Types

        /// <summary>
        /// Configuration for model loading and execution.
        /// </summary>
        private class ModelConfig
        {
            public string ModelName { get; set; }
            public string ModelPath { get; set; }
            public Precision Precision { get; set; } = Precision.FP32;
            public ExecutionProvider ExecutionProvider { get; set; } = ExecutionProvider.CPU;
        }

        /// <summary>
        /// Loading information structure for ONNX Runtime logging.
        /// </summary>
        [StructLayout(LayoutKind.Sequential, CharSet = CharSet.Auto)]
        private struct LoadingInfo
        {
            [MarshalAs(UnmanagedType.ByValTStr, SizeConst = 256)]
            public string ModelName;
        }

        #endregion

        #region IDisposable Implementation

        /// <summary>
        /// Disposes of the ORTModel instance.
        /// </summary>
        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        /// <summary>
        /// Disposes of the ORTModel instance.
        /// </summary>
        /// <param name="disposing">True if called from Dispose(), false if called from finalizer</param>
        protected virtual void Dispose(bool disposing)
        {
            if (!_disposed)
            {
                if (disposing)
                {
                    _session?.Dispose();
                    IsInitialized = false;
                    _loadTask?.Dispose();
                    _loadTask = null;
                    _session = null;
                    
                    QwenLog.Log($"[{_config?.ModelName ?? "ORTModel"}] Disposed successfully");
                }
                
                _disposed = true;
            }
        }

        /// <summary>
        /// Finalizer for the ORTModel class.
        /// </summary>
        ~ORTModel()
        {
            Dispose(false);
        }

        #endregion
    }
} 