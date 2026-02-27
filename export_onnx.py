from __future__ import annotations

import argparse
from pathlib import Path

import onnx
import torch

from usam_infer_utils import build_usam_from_checkpoint, get_device, load_config, resolve_config_path


class USamPromptMode3Wrapper(torch.nn.Module):
    def __init__(self, usam_wrapper: torch.nn.Module) -> None:
        super().__init__()
        self.usam = usam_wrapper.usam

    def forward(
        self,
        image: torch.Tensor,
        boxes: torch.Tensor,
        point_coords: torch.Tensor,
        point_labels: torch.Tensor,
    ) -> torch.Tensor:
        device = image.device
        pixel_mean = torch.tensor(self.usam.pixel_mean, dtype=image.dtype, device=device).view(1, 3, 1, 1)
        pixel_std = torch.tensor(self.usam.pixel_std, dtype=image.dtype, device=device).view(1, 3, 1, 1)
        image = (image - pixel_mean) / pixel_std

        sparse_embeddings, dense_embeddings = self.usam.sam.prompt_encoder(
            points=None,
            boxes=boxes,
            masks=None,
        )
        bt_feature, skip_feature = self.usam.backbone(image)
        image_embedding = self.usam.sam.image_encoder(bt_feature)
        masks, _low_res_masks, _iou_predictions = self.usam.sam.mask_decoder(
            image_embeddings=image_embedding,
            image_pe=self.usam.sam.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=True,
            skip=skip_feature,
        )
        masks = self.usam.sam.postprocess_masks(
            masks=masks,
            input_size=masks.shape[-2:],
            original_size=[self.usam.img_size, self.usam.img_size],
        )
        return masks


def fix_external_data_filename(onnx_path: Path) -> None:
    model = onnx.load(onnx_path.as_posix(), load_external_data=False)
    desired = f"{onnx_path.name}.data"
    current = None
    for initializer in model.graph.initializer:
        if initializer.data_location == onnx.TensorProto.EXTERNAL:
            for kv in initializer.external_data:
                if kv.key == "location":
                    current = kv.value
                    kv.value = desired

    if current is None or current == desired:
        return

    old_data = onnx_path.parent / current
    new_data = onnx_path.parent / desired
    if old_data.exists() and old_data != new_data:
        old_data.rename(new_data)

    onnx.save(model, onnx_path.as_posix())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export U-SAM checkpoint to ONNX")
    parser.add_argument("--pth", required=True, help="Path to .pth checkpoint")
    parser.add_argument("--onnx", required=True, help="Path to output .onnx file")
    parser.add_argument("--config", default=None, help="Path to inference_params.json")
    parser.add_argument(
        "--prompt-mode",
        type=int,
        choices=[0, 3],
        default=0,
        help="0: 无提示导出（默认）；3: 导出支持 boxes + points 的 mode3 推理模型",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = resolve_config_path(None, args.config)
    config = load_config(config_path)
    device = get_device(config)

    base_model = build_usam_from_checkpoint(Path(args.pth), config, device)
    base_model.eval()

    img_size = int(config["img_size"])
    dummy = torch.zeros((1, 3, img_size, img_size), dtype=torch.float32, device=device)

    onnx_path = Path(args.onnx)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)

    if args.prompt_mode == 0:
        torch.onnx.export(
            base_model,
            dummy,
            onnx_path.as_posix(),
            input_names=["input"],
            output_names=["logits"],
            opset_version=int(config.get("onnx_opset", 17)),
            do_constant_folding=True,
        )
    else:
        model = USamPromptMode3Wrapper(base_model).to(device)
        model.eval()
        dummy_boxes = torch.tensor([[0.0, 0.0, float(img_size - 1), float(img_size - 1)]], dtype=torch.float32, device=device)
        dummy_point_coords = torch.tensor([[[img_size / 2.0, img_size / 2.0], [img_size / 3.0, img_size / 3.0], [img_size / 4.0, img_size / 4.0]]], dtype=torch.float32, device=device)
        dummy_point_labels = torch.ones((1, 3), dtype=torch.int64, device=device)

        torch.onnx.export(
            model,
            (dummy, dummy_boxes, dummy_point_coords, dummy_point_labels),
            onnx_path.as_posix(),
            input_names=["input", "boxes", "point_coords", "point_labels"],
            output_names=["logits"],
            opset_version=int(config.get("onnx_opset", 17)),
            do_constant_folding=True,
        )

    fix_external_data_filename(onnx_path)

    print(f"Exported: {onnx_path}")


if __name__ == "__main__":
    main()
