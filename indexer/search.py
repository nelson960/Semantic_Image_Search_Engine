#!/usr/bin/env python3
import sys
from pathlib import Path

# —— insert project root so "extractor" is importable when you do python -m indexer.search
project_root = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(project_root))

import faiss
import numpy as np
from transformers import logging as hf_logging

hf_logging.set_verbosity_error()

from extractor.extract import extract_single_image_embedding

def load_filenames(image_dir: str) -> list[str]:
    base = Path(image_dir)
    exts = {".png", ".jpg", ".jpeg"}
    paths = sorted(p for p in base.rglob("*") if p.suffix.lower() in exts)
    return [str(p.relative_to(base)) for p in paths]

def search_similar_images(
    query_image_path: str,
    index_path: str,
    image_dir: str,
    top_k: int = 5
) -> list[tuple[str, float]]:
    query_emb = (
        extract_single_image_embedding(query_image_path)
        .unsqueeze(0)
        .numpy()
        .astype("float32")
    )
    index = faiss.read_index(index_path)
    distances, indices = index.search(query_emb, top_k)
    filenames = load_filenames(image_dir)

    if indices.max() >= len(filenames):
        raise RuntimeError(
            f"Index idx {indices.max()} >= number of files {len(filenames)}"
        )

    return [
        (filenames[i], float(distances[0][rank]))
        for rank, i in enumerate(indices[0])
    ]

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python -m indexer.search <query_image_path>")
        sys.exit(1)

    results = search_similar_images(
        query_image_path=sys.argv[1],
        index_path="data/index.faiss",
        image_dir="data/raw",
        top_k=5
    )
    print("\nTop 5 similar images:")
    for fn, dist in results:
        print(f"{fn} (distance: {dist:.4f})")
