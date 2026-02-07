from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F

DEFAULT_CONFIG_NAME = "inference_params.json"

DEFAULT_CONFIG: Dict[str, Any] = {
    "img_size": 224,
    "sam_num_classes": 3,
    "pixel_mean": [0.1364736, 0.1364736, 0.1364736],
    "pixel_std": [0.23238614, 0.23238614, 0.23238614],
    "image_key": "image",
    "label_key": "label",
    "device": "cuda",
    "onnx_opset": 18,
}


class USamInferenceWrapper(torch.nn.Module):
    def __init__(self, usam_model: torch.nn.Module) -> None:
        super().__init__()
        self.usam = usam_model

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        device = image.device
        pixel_mean = torch.tensor(self.usam.pixel_mean, dtype=image.dtype, device=device)
        pixel_std = torch.tensor(self.usam.pixel_std, dtype=image.dtype, device=device)
        pixel_mean = pixel_mean.view(1, 3, 1, 1)
        pixel_std = pixel_std.view(1, 3, 1, 1)
        image = (image - pixel_mean) / pixel_std

        bt_feature, skip_feature = self.usam.backbone(image)
        image_embedding = self.usam.sam.image_encoder(bt_feature)
        sparse_embeddings, dense_embeddings = self.usam.sam.prompt_encoder(
            points=None,
            boxes=None,
            masks=None,
        )
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


def resolve_config_path(npz_path: Path | None, config_path: str | None) -> Path:
    if config_path:
        return Path(config_path)
    if npz_path is not None:
        candidate = npz_path.parent / DEFAULT_CONFIG_NAME
        if candidate.exists():
            return candidate
    repo_root = Path(__file__).resolve().parent
    candidate = repo_root / DEFAULT_CONFIG_NAME
    if candidate.exists():
        return candidate
    raise FileNotFoundError(
        "Config JSON not found. Provide --config or place inference_params.json next to the npz file."
    )


def load_config(config_path: Path) -> Dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    config = dict(DEFAULT_CONFIG)
    config.update(data)
    return config


def get_device(config: Dict[str, Any]) -> torch.device:
    device = str(config.get("device", "cuda")).lower()
    if device == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device)


def normalize_image(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32, copy=False)
    max_val = float(np.max(image)) if image.size else 0.0
    if max_val > 1.0:
        image = image / 255.0
    image = np.clip(image, 0.0, 1.0)
    return image


def resize_image(image: np.ndarray, size: int) -> np.ndarray:
    tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0)
    tensor = F.interpolate(tensor, size=(size, size), mode="bilinear", align_corners=False)
    return tensor[0, 0].cpu().numpy()


def make_input_tensor(image: np.ndarray) -> torch.Tensor:
    tensor = torch.from_numpy(image).float()
    tensor = tensor.unsqueeze(0).repeat(3, 1, 1)
    return tensor.unsqueeze(0)


def resize_mask_nearest(mask: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    tensor = torch.from_numpy(mask.astype(np.float32, copy=False)).unsqueeze(0).unsqueeze(0)
    tensor = F.interpolate(tensor, size=(out_h, out_w), mode="nearest")
    out = tensor[0, 0].round().to(torch.int64).cpu().numpy()
    return out


def load_npz(npz_path: Path, image_key: str) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    npz = np.load(npz_path, allow_pickle=True)
    data = {k: npz[k] for k in npz.files}
    if image_key not in data:
        raise KeyError(f"Key '{image_key}' not found in npz: {npz_path}")
    return data, data[image_key]


def save_npz_with_same_keys(
    data: Dict[str, np.ndarray],
    label_key: str,
    pred: np.ndarray,
    out_path: Path,
) -> None:
    out_data = dict(data)
    if label_key in out_data:
        pred = pred.astype(out_data[label_key].dtype, copy=False)
    out_data[label_key] = pred
    np.savez(out_path, **out_data)


def _ensure_usam_on_path() -> Path:
    repo_root = Path(__file__).resolve().parent
    usam_dir = repo_root / "U-SAM"
    if str(usam_dir) not in sys.path:
        sys.path.insert(0, str(usam_dir))
    return usam_dir


def _load_usam_module():
    usam_dir = _ensure_usam_on_path()
    module_path = usam_dir / "u-sam.py"
    spec = importlib.util.spec_from_file_location("usam_module", module_path)
    if spec is None or spec.loader is None:
        raise ImportError("Failed to load U-SAM module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _patch_sam_registry() -> None:
    _ensure_usam_on_path()
    from segment_anything.build_sam import sam_model_registry

    original = sam_model_registry["vit_b"]

    def safe_build_sam_vit_b(num_classes, img_size, checkpoint=None):
        if checkpoint and not Path(checkpoint).exists():
            checkpoint = None
        return original(num_classes=num_classes, img_size=img_size, checkpoint=checkpoint)

    sam_model_registry["vit_b"] = safe_build_sam_vit_b


def build_usam_from_checkpoint(
    checkpoint_path: Path,
    config: Dict[str, Any],
    device: torch.device,
) -> USamInferenceWrapper:
    _patch_sam_registry()
    module = _load_usam_module()
    SAM = module.SAM
    args = SimpleNamespace(
        sam_num_classes=int(config["sam_num_classes"]),
        img_size=int(config["img_size"]),
        use_gt_box=False,
        use_gt_pts=False,
        use_psd_box=False,
        use_psd_pts=False,
        use_psd_mask=False,
        use_text=False,
    )
    model = SAM(args)
    model.pixel_mean = tuple(config["pixel_mean"])
    model.pixel_std = tuple(config["pixel_std"])
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return USamInferenceWrapper(model)
