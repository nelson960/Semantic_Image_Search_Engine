# extractor/extract.py
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

# pick GPU if available
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# explicitly ask for the fast processor to avoid warnings
processor = AutoImageProcessor.from_pretrained(
    "facebook/dinov2-base"
)

# load model, move to device, eval mode
model = AutoModel.from_pretrained(
    "facebook/dinov2-base",
)
model.to(device)
model.eval()

@torch.no_grad()
def extract_single_image_embedding(image_path: str) -> torch.Tensor:
    """
    Extract a DINOv2 feature embedding from a single image.
    Returns a tensor of shape [768] on CPU.
    """
    # load + preprocess
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    
    # move to same device as model
    pixel_values = inputs["pixel_values"].to(device)
    
    # forward pass
    outputs = model(pixel_values=pixel_values, return_dict=True)
    
    # global average over the patch dimension
    last_hidden = outputs.last_hidden_state  # [1, seq_len, 768]
    embedding = last_hidden.mean(dim=1)      # [1, 768]
    
    # move back to CPU and squeeze batch dim
    return embedding.squeeze(0).cpu()
