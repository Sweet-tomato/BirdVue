# -*- coding: utf-8 -*-
"""ONNX image-encoder adapter for the delivery bundle.

Only the image tower is ONNX; text embeddings/scene router stay in the PyTorch
runtime. The wrapper preserves the `model.encode_image(tensor)` call shape.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


class OnnxImageEncoder:
    def __init__(self, model_path: Path, threads: int = 12):
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.intra_op_num_threads = max(1, int(threads))
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            str(model_path), options, providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def __call__(self, pixel_values):
        import torch

        array = pixel_values.detach().cpu().numpy().astype(np.float32)
        output = self.session.run([self.output_name], {self.input_name: array})[0]
        return torch.from_numpy(output).to(pixel_values.device)


def install_onnx_encoder(model, backend: str, onnx_dir: Path, threads: int = 12):
    name = "encode_image_fp32.onnx" if backend == "onnx_fp32" else "encode_image_int8.onnx"
    path = Path(onnx_dir) / name
    if not path.is_file():
        raise FileNotFoundError(path)
    model.encode_image = OnnxImageEncoder(path, threads=threads)
    model._encoder_backend = backend
    model._onnx_model_path = str(path)
    return model
