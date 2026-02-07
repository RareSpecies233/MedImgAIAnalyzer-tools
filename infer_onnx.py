from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnxruntime as ort

from usam_infer_utils import (
    load_config,
    load_npz,
    make_input_tensor,
    normalize_image,
    resize_image,
    resize_mask_nearest,
    resolve_config_path,
    save_npz_with_same_keys,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="U-SAM inference with ONNX")
    parser.add_argument("--onnx", required=True, help="Path to ONNX model")
    parser.add_argument("--npz", required=True, help="Path to input npz file")
    parser.add_argument("--config", default=None, help="Path to inference_params.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    npz_path = Path(args.npz)
    config_path = resolve_config_path(npz_path, args.config)
    config = load_config(config_path)

    data, image = load_npz(npz_path, config["image_key"])
    orig_h, orig_w = image.shape
    image = normalize_image(image)
    image_resized = resize_image(image, int(config["img_size"]))
    input_tensor = make_input_tensor(image_resized).cpu().numpy().astype(np.float32)

    sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    outputs = sess.run(None, {input_name: input_tensor})
    if not outputs:
        raise RuntimeError("ONNX output is empty")
    logits = outputs[0]
    pred = np.argmax(logits, axis=1)[0]

    pred_resized = resize_mask_nearest(pred, orig_h, orig_w)

    out_path = npz_path.parent / f"inference_{npz_path.stem}_onnx.npz"
    save_npz_with_same_keys(data, config["label_key"], pred_resized, out_path)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
