"""Shared ONNX helpers: collapse an export to one .onnx + one .onnx.data."""

from __future__ import annotations

import os


def consolidate(onnx_path: str, pre_export_files: set | None = None) -> None:
    """
    Rewrite the graph so every weight lives in a sibling `.onnx.data`.

    `torch.onnx.export` scatters one file per initializer past the 2 GB
    protobuf limit, which is 254 files for the talker. The runtime expects the
    single-file layout, so the graph is reloaded and re-saved pointing at one
    blob, then the scatter is deleted.

    `pre_export_files` is the directory listing from *before* the export, so
    only files this export created get cleaned up — a sibling model's data in
    the same folder is left alone.
    """
    import onnx

    onnx_dir = os.path.dirname(onnx_path)
    data_path = onnx_path + ".data"

    model = onnx.load(onnx_path)

    # onnx.save_model appends to an existing external-data file rather than
    # truncating it, and writes offsets into the part it just added. Exporting
    # twice into the same folder therefore leaves a file of double the size
    # that loads correctly and wastes exactly one copy of the weights (5.3 GB
    # for the talker). Remove the destination first.
    if os.path.isfile(data_path):
        os.remove(data_path)

    onnx.save_model(
        model,
        onnx_path,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=os.path.basename(data_path),
    )

    if pre_export_files is None:
        return
    keep = {os.path.basename(onnx_path), os.path.basename(data_path)}
    scattered = set(os.listdir(onnx_dir)) - pre_export_files - keep
    for name in scattered:
        path = os.path.join(onnx_dir, name)
        if os.path.isfile(path):
            os.remove(path)
    if scattered:
        print(f"  Cleaned up {len(scattered)} scattered external data files")
