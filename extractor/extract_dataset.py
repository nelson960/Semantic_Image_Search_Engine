"""
Key design choices
──────────────────
• **IterableDataset + DataLoader** ──  each worker decodes JPEGs in
  parallel while the main process/GPU runs the model.

• **Unzip collate_fn** ──  avoids Python unpacking cost inside the loop
  and, being a top-level function, is picklable by worker processes.

• **(path, embedding) triplets + final sort** ──  DataLoader workers
  deliver batches “first-finished”; sorting restores deterministic
  filename ↔ row alignment so downstream FAISS indexing works.

• **_load_model_and_processor() cache** ──  weights stay resident; the
  same function is used elsewhere so the public API is unchanged.
"""

from __future__ import annotations
import os, sys, argparse
from pathlib import Path
from typing import Iterable, Tuple, List

import numpy as np
import torch
from PIL import Image, ImageFile


# Pillow drops truncated images instead of crashing
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Project-local import path & cached model/processor loader
project_root = Path(__file__).parent.resolve()
sys.path.insert(0, str(project_root))
from extract import _load_model_and_processor  # noqa: E402

# Tunables via env vars (do NOT change the public API)
BATCH_SIZE  = int(os.getenv("BATCH_SIZE", 64))
NUM_WORKERS = int(os.getenv("N_WORKERS", 4))

# Helper for DataLoader: turn list of tuples → tuple of lists
# Must live at module scope so it’s picklable by worker processes
def _unzip(batch):
    """Convert [(path,img), …]  →  ([paths …], [imgs …])"""
    paths, imgs = zip(*batch)
    return list(paths), list(imgs)

# Dataset that shards work evenly across DataLoader workers
class ImageFolder(torch.utils.data.IterableDataset):
    """Stream image files under `root`, keeping lexicographic order."""
    exts = {".png", ".jpg", ".jpeg"}

    def __init__(self, root: Path):
        self.paths: List[Path] = sorted(
            p for p in root.rglob("*") if p.suffix.lower() in self.exts
        )
        if not self.paths:
            raise RuntimeError(f"No images found in {root}")

    def __iter__(self) -> Iterable[Tuple[str, Image.Image]]:
        # When num_workers > 0 PyTorch gives each worker a slice of data
        worker = torch.utils.data.get_worker_info()
        paths = self.paths[worker.id :: worker.num_workers] if worker else self.paths
        for p in paths:
            # Convert to RGB once here; avoids branch in hot loop later
            yield str(p), Image.open(p).convert("RGB")

# Main extraction routine (public signature unchanged)  
def batch_extract(input_dir: str, output_file: str, model_name: str) -> None:
    device = (
        torch.device("cuda") if torch.cuda.is_available()
        else torch.device("mps") if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
        else torch.device("cpu")
    )

    # Load HF processor + DINOv2 backbone (cached after first call)
    processor, model = _load_model_and_processor(model_name)

    # Stream files via multi-worker DataLoader
    ds = ImageFolder(Path(input_dir))
    dl = torch.utils.data.DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,            # parallel JPEG decode
        pin_memory=device.type == "cuda",   # faster H2D copies
        collate_fn=_unzip,                  # keep order per batch
    )

    triplets: List[Tuple[str, torch.Tensor]] = []   # (path, embedding)
        
    # Main loop: preprocess → move → forward → pool
    # Everything after processor() runs on the chosen device
    for paths, imgs in dl:
        with torch.inference_mode():
            # HF pre-processor: resize / crop / normalise (CPU)
            px = processor(images=list(imgs), return_tensors="pt")["pixel_values"]

            # One non-blocking copy per batch
            px = px.to(device, dtype=model.dtype, non_blocking=(device.type == "cuda"))

            # Forward pass (batched) → [B, 257, 1024] hidden states
            h = model(pixel_values=px, return_dict=True).last_hidden_state

            # Mean-pool CLS tokens, move to CPU & stash with filename
            for p, e in zip(paths, h.mean(dim=1)):
                triplets.append((p, e.float().cpu()))


    # Restore deterministic file order required by downstream FAISS code
    triplets.sort(key=lambda t: t[0])               # lexicographic path
    stacked = torch.stack([e for _, e in triplets], 0).numpy()

    # Save once (zero-copy) –> <output_file>.npy
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    np.save(output_file, stacked, allow_pickle=False)
    print(f"Saved {stacked.shape[0]} embeddings (dim={stacked.shape[1]}) → {output_file}")

    if device.type == "cuda":
        torch.cuda.empty_cache() # free VRAM early

def main():
    parser = argparse.ArgumentParser(
        description="Extract DINOv2 embeddings for all images in a folder"
    )
    parser.add_argument("input_dir",  type=Path, help="folder with images")
    parser.add_argument("output_file", type=Path, help="output .npy file")
    parser.add_argument("-m", "--model_name",
                        default="facebook/dinov2-base",
                        help="facebook/dinov2-small|base|large")
    args = parser.parse_args()
    batch_extract(str(args.input_dir), str(args.output_file), args.model_name)

if __name__ == "__main__":
    main()
