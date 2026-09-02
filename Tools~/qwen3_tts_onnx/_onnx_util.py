"""Shared ONNX helpers: one .onnx + one .onnx.data, matching ElBruno CustomVoice layout."""

from __future__ import annotations

import os


def consolidate(onnx_path: str, pre_export_files: set | None = None) -> None:
    """Rewrite the graph so all weights live in a sibling .onnx.data file."""
    import onnx

    onnx_dir = os.path.dirname(onnx_path)
    data_path = onnx_path + ".data"
    model = onnx.load(onnx_path)
    onnx.save_model(
        model,
        onnx_path,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=os.path.basename(data_path),
    )
    if pre_export_files is None:
        return
    current = set(os.listdir(onnx_dir))
    scattered = current - pre_export_files - {
        os.path.basename(onnx_path),
        os.path.basename(data_path),
    }
    for name in scattered:
        path = os.path.join(onnx_dir, name)
        if os.path.isfile(path):
            os.remove(path)
    if scattered:
        print(f"  Cleaned up {len(scattered)} scattered external data files")
