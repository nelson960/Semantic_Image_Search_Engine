import torch
from functools import lru_cache
from pathlib import Path
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

# ----------------------------------------------------------------------
# pick the best device (CUDA → MPS → CPU)
# ----------------------------------------------------------------------
if torch.cuda.is_available():
    device = torch.device("cuda")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

# ----------------------------------------------------------------------
# cache (model, processor) so we load weights only once per model_name
# ----------------------------------------------------------------------
@lru_cache(maxsize=None)
def _load_model_and_processor(model_name: str):
    """
    Load DINOv2 model + processor once and cache the pair.
    """
    processor = AutoImageProcessor.from_pretrained(model_name, use_fast=True)
    model = AutoModel.from_pretrained(model_name).to(device).eval()
    return processor, model

# ----------------------------------------------------------------------
# public API – unchanged signature / behaviour
# ----------------------------------------------------------------------
@torch.no_grad()
def extract_single_image_embedding(
    image_path: str,
    model_name: str = "facebook/dinov2-base",
) -> torch.Tensor:
    """
    Extract a DINOv2 feature embedding from a single image.
    Returns a tensor of shape [768] on CPU.
    """
    processor, model = _load_model_and_processor(model_name)

    # 1. load & preprocess
    image = Image.open(Path(image_path)).convert("RGB")
    pixel_values = processor(images=image, return_tensors="pt")["pixel_values"].to(device)

    # 2. forward pass
    outputs = model(pixel_values=pixel_values, return_dict=True)

    # 3. mean-pool CLS tokens → [768]
    embedding = outputs.last_hidden_state.mean(dim=1).cpu().squeeze(0)
    return embedding
