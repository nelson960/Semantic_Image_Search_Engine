import torch
from functools import lru_cache
from pathlib import Path
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

if torch.cuda.is_available():
    device = torch.device("cuda")
elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

if device.type == "cuda":
    torch.backends.cudnn.benchmark = True

# cache model + processor
@lru_cache(maxsize=None)
def _load_model_and_processor(model_name: str):
    processor = AutoImageProcessor.from_pretrained(model_name, use_fast=True)

    dtype = torch.float16 if device.type == "cuda" else torch.float32
    model = (
        AutoModel.from_pretrained(model_name, torch_dtype=dtype)
        .to(device)
        .eval()
    )

    # compile only on CUDA
    if device.type == "cuda" and hasattr(torch, "compile"):
        model = torch.compile(model, mode="reduce-overhead")

    return processor, model

# public API 
@torch.inference_mode()
def extract_single_image_embedding(
    image_path: str,
    model_name: str = "facebook/dinov2-base",
) -> torch.Tensor:
    processor, model = _load_model_and_processor(model_name)
    image = Image.open(Path(image_path)).convert("RGB")
    px = processor(images=image, return_tensors="pt")["pixel_values"]
    px = px.to(device, dtype=model.dtype, non_blocking=(device.type == "cuda"))
    h = model(pixel_values=px, return_dict=True).last_hidden_state
    return h.mean(dim=1).float().cpu().squeeze(0)
