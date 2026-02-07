from __future__ import annotations

import argparse
from pathlib import Path

import torch

from usam_infer_utils import (
    build_usam_from_checkpoint,
    get_device,
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
    parser = argparse.ArgumentParser(description="U-SAM inference with PyTorch checkpoint")
    parser.add_argument("npz", help="Path to input npz file")
    parser.add_argument(
        "--model",
        default=str(Path(__file__).resolve().parent / "best.pth"),
        help="Path to .pth checkpoint (default: repo root best.pth)",
    )
    parser.add_argument("--config", default=None, help="Path to inference_params.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    npz_path = Path(args.npz)
    config_path = resolve_config_path(npz_path, args.config)
    config = load_config(config_path)

    device = get_device(config)
    model = build_usam_from_checkpoint(Path(args.model), config, device)

    data, image = load_npz(npz_path, config["image_key"])
    orig_h, orig_w = image.shape
    image = normalize_image(image)
    image_resized = resize_image(image, int(config["img_size"]))
    input_tensor = make_input_tensor(image_resized).to(device)

    with torch.no_grad():
        logits = model(input_tensor)
        pred = torch.argmax(logits, dim=1)[0].cpu().numpy()

    pred_resized = resize_mask_nearest(pred, orig_h, orig_w)

    out_path = npz_path.parent / f"inference_{npz_path.stem}.npz"
    save_npz_with_same_keys(data, config["label_key"], pred_resized, out_path)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
