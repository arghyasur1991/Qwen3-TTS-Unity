# Third-party notices

## ElBruno.QwenTTS

The ONNX inference path is derived from
[ElBruno.QwenTTS](https://github.com/elbruno/ElBruno.QwenTTS),
(C) 2026 Bruno Capuano, MIT License. Specifically:

- `Runtime/Engine/LanguageModel.cs`
- `Runtime/Engine/EmbeddingStore.cs`
- `Runtime/Engine/TextTokenizer.cs`
- `Runtime/Audio/MelSpectrogram.cs`
- `Runtime/Internal/NpyReader.cs`, `Runtime/Internal/IOUtil.cs`

`TextTokenizer` is an in-repo GPT-2 byte-level BPE standing in for
`Microsoft.ML.Tokenizers`, which is not available to this Unity package. It
follows the same HuggingFace GPT-2 / Qwen2 tokenizer algorithm.

The original license text:

    MIT License

    Copyright (c) 2026 Bruno Capuano

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING WITHOUT LIMITATION THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.

## Spark-TTS-Unity

The ONNX Runtime session infrastructure — `Runtime/Onnx/ORTModel.cs` and the
supporting helpers in `Runtime/Internal/` — comes from
[Spark-TTS-Unity](https://github.com/arghyasur1991/Spark-TTS-Unity)
(Apache-2.0), where this package's inference path was first developed. That
project's own Spark-TTS engine is not included here.

## Qwen3-TTS

Prompt construction, sampling defaults and the in-context cloning layout follow
Alibaba's [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) reference
implementation (Apache-2.0).

The exported graph layout and inference order follow
[zukky/Qwen3-TTS-ONNX-DLL](https://huggingface.co/zukky/Qwen3-TTS-ONNX-DLL)
(Apache-2.0), itself derived from Qwen3-TTS.

Model weights are Alibaba's and are covered by their own licence. This package
neither includes nor downloads them, and does not redistribute the Windows
`qwen3_tts_rust.dll` that project ships.
