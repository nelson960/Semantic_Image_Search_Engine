#!/usr/bin/env python3
import sys
from pathlib import Path
import argparse
import numpy as np
import torch

# make sure extract.py is importable
project_root = Path(__file__).parent.resolve()
sys.path.insert(0, str(project_root))

from extract import extract_single_image_embedding

def batch_extract(input_dir: str, output_file: str, model_name: str) -> None:
    input_path = Path(input_dir)
    exts = {".png", ".jpg", ".jpeg"}
    image_paths = sorted(p for p in input_path.rglob("*")
                         if p.suffix.lower() in exts)
    if not image_paths:
        raise RuntimeError(f"No images found in {input_dir}")

    embeddings = []
    for img_path in image_paths:
        emb = extract_single_image_embedding(str(img_path), model_name)
        embeddings.append(emb)

    all_embs = torch.stack(embeddings, 0).numpy()
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    np.save(output_file, all_embs)
    print(f"Saved {all_embs.shape[0]} embeddings (dim={all_embs.shape[1]}) to {output_file}")

def main():
    parser = argparse.ArgumentParser(
        description="Extract DINOv2 embeddings for images under a folder"
    )
    parser.add_argument("input_dir",  type=Path,
                        help="folder of images")
    parser.add_argument("output_file", type=Path,
                        help="where to save embeddings (.npy)")
    parser.add_argument(
        "-m", "--model_name",
        default="facebook/dinov2-base",
        help="which DINOv2 model to use (facebook/dinov2-small|base|large)"
    )
    args = parser.parse_args()
    batch_extract(str(args.input_dir), str(args.output_file), args.model_name)

if __name__ == "__main__":
    main()
