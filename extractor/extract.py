import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

# prefer CUDA, then MPS, then CPU
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

@torch.no_grad()
def extract_single_image_embedding(
    image_path: str,
    model_name: str = "facebook/dinov2-base"
) -> torch.Tensor:
    """
    Extract a DINOv2 feature embedding from a single image.
    Returns a tensor of shape [768] on CPU.
    """
    processor = AutoImageProcessor.from_pretrained(model_name, use_fast=True, device=device)
    model = AutoModel.from_pretrained(model_name)
    model.to(device).eval()

    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)

    outputs = model(pixel_values=pixel_values, return_dict=True)
    embedding = outputs.last_hidden_state.mean(dim=1)  # [1, 768]
    return embedding.squeeze(0).cpu()
