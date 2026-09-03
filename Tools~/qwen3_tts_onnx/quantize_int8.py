#!/usr/bin/env python3
"""Quantize the talker and code predictor to int8.

Batch-1 autoregressive decode reads every weight once per token, so both graphs
are limited by how fast their weights can be streamed from memory (190-218 GB/s
measured, against 546 GB/s peak on an M4 Max) rather than by arithmetic.
Quartering the weight bytes is the remaining lever.

int8 and not fp16: ONNX Runtime's CPU provider has hand-written int8 GEMM
kernels in MLAS, but no fast fp16 kernels for these ops on Apple silicon. fp16
measured **17x slower** while being numerically near-perfect — see
`fp16_spike.py`.

Two settings matter for quality and both default the careful way:

- `per_channel=True` gives each output channel its own scale instead of one
  scale for the whole tensor. A weight matrix whose rows differ in magnitude
  loses badly to a single shared scale, and this is nearly free at inference.
- `MatMulConstBOnly=True` restricts quantization to MatMuls with a constant
  right-hand side, i.e. the weights. The attention products (QK^T and PV) have
  two activation inputs; quantizing those costs accuracy and saves no weight
  traffic, because there are no weights involved.

Two sets of weights are held back in fp32:

- **The output projection**, which turns hidden states straight into the logits
  the sampler reads. ~25 MB of a 5.67 GB graph. It measurably helps the code
  predictor (8.1% -> 4.1% peak logit error) and, interestingly, not the talker,
  whose error is already accumulated by the time it gets there.
- **The first and last three decoder layers**, which is the setting that makes
  the talker usable at all. Quantized end to end its peak logit error is 15%
  and Whisper transcribes the output as "The Saner sees your ceiling" instead
  of "The scanner..." — a dropped phoneme, WER 0.125. Holding six of
  twenty-eight layers halves the error to 7.2%, restores an exact transcript,
  and still gives 1.76x instead of 2.26x.

    conda activate sparktts
    python quantize_int8.py ~/Downloads/Qwen3-TTS-ONNX/Qwen3-1.7B-VoiceDesign
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
import time

GRAPHS = ("talker", "code_predictor")


def output_projections(model_path: str) -> list[str]:
    """
    Names of the MatMuls that produce the graph's logits.

    Found by walking back from each output called `logits` rather than by
    matching on a name, so this keeps working if the exporter renames things.
    """
    import onnx

    model = onnx.load(model_path, load_external_data=False)
    producers = {out: node for node in model.graph.node for out in node.output}
    found = []
    for out in model.graph.output:
        if "logits" not in out.name:
            continue
        node = producers.get(out.name)
        if node is not None and node.op_type in ("MatMul", "Gemm"):
            found.append(node.name)
    return found


def sensitive_layer_nodes(model_path: str, layers: list[int]) -> list[str]:
    """
    MatMul names belonging to the given decoder layer indices.

    Error in an autoregressive stack compounds: what the first layers get wrong
    every later layer builds on, and the last layers write the hidden state the
    logits are read from. Holding those few in fp32 is the usual way to buy back
    accuracy for a small fraction of the weights.
    """
    import onnx
    import re

    model = onnx.load(model_path, load_external_data=False)
    wanted = set(layers)
    out = []
    for node in model.graph.node:
        if node.op_type not in ("MatMul", "Gemm"):
            continue
        m = re.search(r"layers\.(\d+)", node.name)
        if m and int(m.group(1)) in wanted:
            out.append(node.name)
    return out


# Holding a few layers back only makes sense on a deep stack. The code
# predictor has five layers, so reserving six would quantize nothing; the
# talker has twenty-eight and can spare them.
MIN_LAYERS_TO_HOLD_ENDS = 12
LAYERS_HELD_AT_EACH_END = 3


def default_held_layers(model_path: str) -> list[int]:
    """
    First and last few decoder layers, on graphs deep enough to spare them.

    This is the difference between usable and not. With every layer quantized
    the talker's peak logit error is 15% and a Whisper transcription of the
    result turns "scanner" into "Saner" — a dropped phoneme, WER 0.125. Holding
    the outermost three layers at each end halves the error to 7.2% and the
    transcript is exact again, for 1.76x instead of 2.26x.
    """
    import onnx
    import re

    model = onnx.load(model_path, load_external_data=False)
    idx = set()
    for node in model.graph.node:
        m = re.search(r"layers\.(\d+)", node.name)
        if m:
            idx.add(int(m.group(1)))
    if len(idx) < MIN_LAYERS_TO_HOLD_ENDS:
        return []
    lo, hi = min(idx), max(idx)
    n = LAYERS_HELD_AT_EACH_END
    return sorted(set(range(lo, lo + n)) | set(range(hi - n + 1, hi + 1)))


def quantize(model_dir: str, graph: str, per_channel: bool, force: bool,
             keep_head: bool = True, keep_layers: list[int] | None = None) -> str | None:
    from onnxruntime.quantization import quantize_dynamic, QuantType

    src = os.path.join(model_dir, f"{graph}.onnx")
    if not os.path.isfile(src):
        print(f"  {graph}.onnx not found, skipping")
        return None

    dst = os.path.join(model_dir, f"{graph}_int8.onnx")
    if os.path.isfile(dst) and not force:
        print(f"  {graph}_int8.onnx already present (--force to redo)")
        return dst

    # The quantizer appends to an existing external-data file rather than
    # truncating it, the same trap _onnx_util.consolidate has to work around.
    for stale in (dst, dst + ".data", dst + "_data"):
        if os.path.isfile(stale):
            os.remove(stale)

    src_bytes = os.path.getsize(src) + (
        os.path.getsize(src + ".data") if os.path.isfile(src + ".data") else 0)
    print(f"  quantizing {graph} ({src_bytes / 1e9:.2f} GB, "
          f"per_channel={per_channel}) ...")

    exclude = output_projections(src) if keep_head else []
    if exclude:
        print(f"  keeping the output projection in fp32: {', '.join(exclude)}")
    if keep_layers is None:
        keep_layers = default_held_layers(src)
    if keep_layers:
        layer_nodes = sensitive_layer_nodes(src, keep_layers)
        if layer_nodes:
            print(f"  keeping layers {keep_layers} in fp32 "
                  f"({len(layer_nodes)} matmuls)")
            exclude += layer_nodes

    t = time.perf_counter()
    quantize_dynamic(
        src, dst,
        weight_type=QuantType.QInt8,
        per_channel=per_channel,
        nodes_to_exclude=exclude,
        # reduce_range is a workaround for pre-VNNI x86 and costs a bit of
        # precision; ARM does not need it.
        reduce_range=False,
        # Required above the 2 GB protobuf limit, and harmless below it.
        use_external_data_format=True,
        extra_options={"MatMulConstBOnly": True},
    )
    gc.collect()

    out_bytes = sum(
        os.path.getsize(p) for p in (dst, dst + ".data", dst + "_data")
        if os.path.isfile(p))
    print(f"  -> {out_bytes / 1e9:.2f} GB "
          f"({src_bytes / max(out_bytes, 1):.2f}x smaller) "
          f"in {time.perf_counter() - t:.0f}s")
    return dst


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("checkpoint_dir", nargs="+")
    ap.add_argument("--graphs", nargs="*", default=list(GRAPHS), choices=GRAPHS)
    ap.add_argument("--per-tensor", action="store_true",
                    help="One scale per weight tensor. Faster to produce, "
                         "measurably worse; kept only for comparison.")
    ap.add_argument("--keep-layers", type=int, nargs="*", default=None,
                    help="Decoder layer indices to leave in fp32. Defaults to the "
                         "first and last three on stacks deep enough to spare them, "
                         "which is what makes the talker intelligible; pass an "
                         "empty list to quantize everything.")
    ap.add_argument("--quantize-head", action="store_true",
                    help="Also quantize the output projection. Saves ~25 MB and "
                         "puts error straight onto the logits; not recommended.")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    for raw in args.checkpoint_dir:
        d = os.path.expanduser(raw)
        if not os.path.isdir(d):
            print(f"{raw}: not a directory", file=sys.stderr)
            return 1
        print(f"{os.path.basename(d.rstrip('/'))}:")
        for graph in args.graphs:
            quantize(d, graph, per_channel=not args.per_tensor, force=args.force,
                     keep_head=not args.quantize_head, keep_layers=args.keep_layers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
