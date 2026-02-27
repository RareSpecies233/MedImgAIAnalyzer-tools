from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnxruntime as ort
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
    parser = argparse.ArgumentParser(description="Prompt mode 3 inference (ONNX/PTH)")
    parser.add_argument("--mode", choices=["onnx", "pth"], required=True, help="推理后端")
    parser.add_argument("--model", required=True, help="模型路径（.onnx 或 .pth）")
    parser.add_argument("--npz", required=True, help="输入 npz 路径")
    parser.add_argument("--config", default=None, help="inference_params.json 路径")
    parser.add_argument("--output", default=None, help="输出 npz 路径（默认 inference_<name>_mode3[_onnx].npz）")
    return parser.parse_args()


def build_prompts_from_label(label_resized: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ys, xs = np.where(label_resized > 0)
    if len(xs) == 0:
        raise ValueError("label 中没有前景像素，无法生成 mode3 提示（boxes + points）")

    x_min, x_max = int(xs.min()), int(xs.max())
    y_min, y_max = int(ys.min()), int(ys.max())
    margin = 2
    h, w = label_resized.shape
    x_min = max(0, x_min - margin)
    x_max = min(w - 1, x_max + margin)
    y_min = max(0, y_min - margin)
    y_max = min(h - 1, y_max + margin)

    boxes = np.array([[x_min, y_min, x_max, y_max]], dtype=np.float32)

    fg_points = np.stack([ys, xs], axis=1)
    if fg_points.shape[0] >= 3:
        idx = np.linspace(0, fg_points.shape[0] - 1, 3, dtype=int)
        sampled = fg_points[idx]
    else:
        sampled = np.repeat(fg_points[:1], 3, axis=0)

    point_coords = sampled.astype(np.float32)[None, :, :]
    point_labels = np.ones((1, point_coords.shape[1]), dtype=np.int64)
    return boxes, point_coords, point_labels


def infer_pth_mode3(
    model_path: Path,
    input_tensor: torch.Tensor,
    boxes: np.ndarray,
    point_coords: np.ndarray,
    point_labels: np.ndarray,
    config: dict,
) -> np.ndarray:
    device = get_device(config)
    wrapper = build_usam_from_checkpoint(model_path, config, device)
    usam = wrapper.usam
    usam.eval()

    with torch.no_grad():
        image = input_tensor.to(device)
        pixel_mean = torch.tensor(usam.pixel_mean, dtype=image.dtype, device=device).view(1, 3, 1, 1)
        pixel_std = torch.tensor(usam.pixel_std, dtype=image.dtype, device=device).view(1, 3, 1, 1)
        image = (image - pixel_mean) / pixel_std

        boxes_t = torch.from_numpy(boxes).to(device=device, dtype=torch.float32)
        coords_t = torch.from_numpy(point_coords).to(device=device, dtype=torch.float32)
        labels_t = torch.from_numpy(point_labels).to(device=device, dtype=torch.int64)

        sparse_embeddings, dense_embeddings = usam.sam.prompt_encoder(
            points=(coords_t, labels_t),
            boxes=boxes_t,
            masks=None,
        )
        bt_feature, skip_feature = usam.backbone(image)
        image_embedding = usam.sam.image_encoder(bt_feature)
        masks, _low_res_masks, _iou_predictions = usam.sam.mask_decoder(
            image_embeddings=image_embedding,
            image_pe=usam.sam.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=True,
            skip=skip_feature,
        )
        masks = usam.sam.postprocess_masks(
            masks=masks,
            input_size=masks.shape[-2:],
            original_size=[usam.img_size, usam.img_size],
        )
        pred = torch.argmax(masks, dim=1)[0].cpu().numpy()
    return pred


def infer_onnx_mode3(
    model_path: Path,
    input_tensor: np.ndarray,
    boxes: np.ndarray,
    point_coords: np.ndarray,
    point_labels: np.ndarray,
) -> np.ndarray:
    sess = ort.InferenceSession(model_path.as_posix(), providers=["CPUExecutionProvider"])
    inputs = sess.get_inputs()
    if len(inputs) < 4:
        raise RuntimeError(
            "当前 ONNX 不是 mode3 导出模型（需要 4 个输入：input/boxes/point_coords/point_labels）。"
            "请先用 export_onnx.py --prompt-mode 3 重新导出。"
        )

    feed = {
        inputs[0].name: input_tensor.astype(np.float32, copy=False),
        inputs[1].name: boxes.astype(np.float32, copy=False),
        inputs[2].name: point_coords.astype(np.float32, copy=False),
        inputs[3].name: point_labels.astype(np.int64, copy=False),
    }
    logits = sess.run(None, feed)[0]
    pred = np.argmax(logits, axis=1)[0]
    return pred


def main() -> None:
    args = parse_args()

    npz_path = Path(args.npz)
    model_path = Path(args.model)
    if not npz_path.is_file():
        raise FileNotFoundError(f"输入 npz 不存在: {npz_path}")
    if not model_path.is_file():
        raise FileNotFoundError(f"模型文件不存在: {model_path}")

    config_path = resolve_config_path(npz_path, args.config)
    config = load_config(config_path)
    data, image = load_npz(npz_path, config["image_key"])
    label_key = config["label_key"]
    if label_key not in data:
        raise KeyError(f"npz 中缺少 label 键 '{label_key}'，mode3 推理无法构造提示")

    label = data[label_key]
    if label.ndim != 2:
        raise ValueError(f"当前仅支持二维 label，收到形状: {label.shape}")

    orig_h, orig_w = image.shape
    image = normalize_image(image)
    img_size = int(config["img_size"])
    image_resized = resize_image(image, img_size)
    label_resized = resize_mask_nearest(label, img_size, img_size)

    boxes, point_coords, point_labels = build_prompts_from_label(label_resized)

    if args.mode == "pth":
        input_tensor = make_input_tensor(image_resized)
        pred = infer_pth_mode3(model_path, input_tensor, boxes, point_coords, point_labels, config)
    else:
        input_tensor = make_input_tensor(image_resized).cpu().numpy()
        pred = infer_onnx_mode3(model_path, input_tensor, boxes, point_coords, point_labels)

    pred_resized = resize_mask_nearest(pred, orig_h, orig_w)

    if args.output:
        out_path = Path(args.output)
    else:
        suffix = "_mode3_onnx" if args.mode == "onnx" else "_mode3"
        out_path = npz_path.parent / f"inference_{npz_path.stem}{suffix}.npz"

    save_npz_with_same_keys(data, label_key, pred_resized, out_path)
    print(f"Saved: {out_path}")
    print(f"Pred unique: {np.unique(pred_resized)}")


if __name__ == "__main__":
    main()
