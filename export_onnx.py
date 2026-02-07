from __future__ import annotations

import argparse
from pathlib import Path

import torch

from usam_infer_utils import build_usam_from_checkpoint, get_device, load_config, resolve_config_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export U-SAM checkpoint to ONNX")
    parser.add_argument("--pth", required=True, help="Path to .pth checkpoint")
    parser.add_argument("--onnx", required=True, help="Path to output .onnx file")
    parser.add_argument("--config", default=None, help="Path to inference_params.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = resolve_config_path(None, args.config)
    config = load_config(config_path)
    device = get_device(config)

    model = build_usam_from_checkpoint(Path(args.pth), config, device)
    model.eval()

    img_size = int(config["img_size"])
    dummy = torch.zeros((1, 3, img_size, img_size), dtype=torch.float32, device=device)

    onnx_path = Path(args.onnx)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        dummy,
        onnx_path.as_posix(),
        input_names=["input"],
        output_names=["logits"],
        opset_version=int(config.get("onnx_opset", 17)),
        do_constant_folding=True,
    )

    print(f"Exported: {onnx_path}")


if __name__ == "__main__":
    main()
