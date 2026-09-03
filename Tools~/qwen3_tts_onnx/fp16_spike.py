#!/usr/bin/env python3
"""Is fp16 actually faster on the CPU execution provider?

Batch-1 autoregressive decode reads every weight once per token, so the talker
is bound by how fast its 5.27 GiB can be streamed from memory (measured
190 GB/s against 546 GB/s peak on an M4 Max) rather than by arithmetic. Halving
the weight bytes should therefore halve the step time — *if* ONNX Runtime has
native fp16 kernels for these ops on this hardware. If it does not, it inserts
Cast nodes, computes in fp32 anyway, and fp16 ends up slower while still
halving disk. That is the question, and it is not answerable from the docs.

Converts with `keep_io_types=True` on purpose: inputs and outputs stay fp32, so
the C# side needs no new buffer types and this can be a drop-in file swap. The
KV cache is a large fp32 in/out tensor under that setting, so its casts are part
of what gets measured.

Unity ships ONNX Runtime 1.21.0 and this environment has 1.21.0, so the timing
here transfers.

Defaults to the **code predictor** rather than the talker. It is 0.41 GB
against 5.27 GB, converts in seconds, and is itself 41% of synthesis wall
clock — and the question being asked is about ONNX Runtime's fp16 kernels on
this hardware, which one graph answers as well as another. Converting the
talker needs the fp32 graph, an fp16 copy and a protobuf serialisation live at
once; both the in-memory and the model-path entry points got the process killed
on a 64 GB machine, so prove the principle on the small graph first.

    conda activate sparktts
    python fp16_spike.py --model-dir ~/Downloads/Qwen3-TTS-ONNX/Qwen3-1.7B-VoiceDesign
    python fp16_spike.py --model-dir ... --graph talker    # once it is worth it
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import time

import numpy as np

np.seterr(all="ignore")




def convert(model_dir: str, graph: str) -> str:
    import onnx
    # ONNX Runtime ships a maintained fork of onnxconverter_common's converter.
    # The upstream 1.16.0 copy throws AttributeError from
    # remove_unnecessary_cast_node on these graphs.
    from onnxruntime.transformers import float16

    src = os.path.join(model_dir, f"{graph}.onnx")
    dst = os.path.join(model_dir, f"{graph}_fp16.onnx")
    if os.path.isfile(dst) and os.path.isfile(dst + ".data"):
        print(f"  {os.path.basename(dst)} already present, reusing")
        return dst

    print(f"  converting {src} (keep_io_types=True) ...")
    model = float16.convert_float_to_float16(
        onnx.load(src), keep_io_types=True, disable_shape_infer=True)

    data = dst + ".data"
    if os.path.isfile(data):
        os.remove(data)
    print(f"  saving {dst} ...")
    onnx.save_model(model, dst, save_as_external_data=True,
                    all_tensors_to_one_file=True,
                    location=os.path.basename(data))
    del model
    gc.collect()
    return dst


def quantize_int8(model_dir: str, graph: str) -> str:
    """
    Dynamic int8 weight quantization.

    Unlike fp16, ONNX Runtime's CPU provider has hand-written int8 GEMM kernels
    in MLAS, so this is the quantization that stands a chance on this hardware.
    Weights only, activations computed dynamically — no calibration set needed.
    """
    from onnxruntime.quantization import quantize_dynamic, QuantType

    src = os.path.join(model_dir, f"{graph}.onnx")
    dst = os.path.join(model_dir, f"{graph}_int8.onnx")
    if os.path.isfile(dst):
        print(f"  {os.path.basename(dst)} already present, reusing")
        return dst
    print(f"  quantizing {graph} to int8 ...")
    quantize_dynamic(src, dst, weight_type=QuantType.QInt8, per_channel=True,
                     use_external_data_format=True,
                     extra_options={"MatMulConstBOnly": True})
    return dst


def session(path):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.log_severity_level = 3
    return ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])


def talker_feeds(cfg, past, rng):
    shape = (cfg["num_hidden_layers"], 1, cfg["num_key_value_heads"], past, cfg["head_dim"])
    return {
        "inputs_embeds": rng.standard_normal((1, 1, cfg["hidden_size"])).astype(np.float32),
        "attention_mask": np.ones((1, past + 1), dtype=np.int64),
        "position_ids": np.full((3, 1, 1), past, dtype=np.int64),
        "past_keys": rng.standard_normal(shape).astype(np.float32),
        "past_values": rng.standard_normal(shape).astype(np.float32),
    }


def cp_feeds(cfg, rng, in_dim):
    """A mid-group code-predictor step: one token, a short cache."""
    past = 4
    shape = (cfg["num_hidden_layers"], 1, cfg["num_key_value_heads"], past, cfg["head_dim"])
    return {
        "inputs_embeds": rng.standard_normal((1, 1, in_dim)).astype(np.float32),
        "generation_steps": np.array([4], dtype=np.int64),
        "past_keys": rng.standard_normal(shape).astype(np.float32),
        "past_values": rng.standard_normal(shape).astype(np.float32),
    }


def bench(sess, feeds, iters):
    names = ["logits"]
    sess.run(names, feeds)  # warm
    best = 1e9
    for _ in range(iters):
        t = time.perf_counter()
        out = sess.run(names, feeds)
        best = min(best, time.perf_counter() - t)
    return best * 1000.0, out[0]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--graph", default="code_predictor",
                    choices=["code_predictor", "talker"])
    ap.add_argument("--past", type=int, default=200,
                    help="Talker KV length; a realistic mid-utterance step")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--int8", action="store_true",
                    help="Also try dynamic int8, which MLAS has real CPU kernels for")
    ap.add_argument("--skip-fp16", action="store_true",
                    help="fp16 is settled (17x slower); skip converting it")
    args = ap.parse_args()

    model_dir = os.path.expanduser(args.model_dir)
    full = json.load(open(os.path.join(model_dir, "embeddings", "config.json")))

    print("Converting ...")
    fp16_path = None if args.skip_fp16 else convert(model_dir, args.graph)

    fp32_path = os.path.join(model_dir, f"{args.graph}.onnx")
    for p in [q for q in (fp32_path, fp16_path) if q]:
        total = sum(os.path.getsize(x) for x in (p, p + ".data") if os.path.isfile(x))
        print(f"  {os.path.basename(p):<24} {total / 1e9:6.2f} GB")

    rng = np.random.default_rng(7)
    if args.graph == "talker":
        feeds = talker_feeds(full["talker"], args.past, rng)
        label = f"talker decode step, past={args.past}"
    else:
        cp = full["code_predictor"]
        feeds = cp_feeds(cp, rng, cp["hidden_size"])
        label = "code-predictor step"

    print(f"\n{label}, best of {args.iters}:")
    s32 = session(fp32_path)
    ms32, logits32 = bench(s32, feeds, args.iters)
    print(f"  fp32 {ms32:7.2f} ms")
    del s32
    gc.collect()

    ms16, logits16 = None, None
    if fp16_path:
        s16 = session(fp16_path)
        ms16, logits16 = bench(s16, feeds, args.iters)
        print(f"  fp16 {ms16:7.2f} ms   ({ms32 / ms16:.2f}x)")
        del s16
        gc.collect()

    ms8, logits8 = None, None
    if args.int8:
        i8 = quantize_int8(model_dir, args.graph)
        total = sum(os.path.getsize(x) for x in (i8, i8 + ".data", i8 + "_data")
                    if os.path.isfile(x))
        print(f"  int8 file {total / 1e9:.2f} GB")
        s8 = session(i8)
        ms8, logits8 = bench(s8, feeds, args.iters)
        print(f"  int8 {ms8:7.2f} ms   ({ms32 / ms8:.2f}x)")
        del s8
        gc.collect()

    # Absolute logit error matters less than whether the same token still wins,
    # since these go straight into top-k / top-p sampling.
    def report(name, logits):
        d = float(np.abs(logits32 - logits).max())
        rel = d / float(np.abs(logits32).max())
        agree = int(logits32.argmax()) == int(logits.argmax())
        k = 50
        a = set(np.argsort(logits32.ravel())[-k:].tolist())
        b = set(np.argsort(logits.ravel())[-k:].tolist())
        print(f"  {name}: max abs diff {d:.4e} ({rel:.2%} of peak), "
              f"argmax agrees {agree}, top-{k} overlap {len(a & b)}/{k}")

    print("\nNumerics vs fp32:")
    if logits16 is not None:
        report("fp16", logits16)
    if logits8 is not None:
        report("int8", logits8)

    def verdict(ms):
        return "FASTER" if ms < ms32 * 0.95 else ("SLOWER" if ms > ms32 * 1.05 else "NO CHANGE")

    if ms16 is not None:
        print(f"\nfp16 is {verdict(ms16)} on this build.")
    if ms8 is not None:
        print(f"int8 is {verdict(ms8)} on this build.")


if __name__ == "__main__":
    raise SystemExit(main())
