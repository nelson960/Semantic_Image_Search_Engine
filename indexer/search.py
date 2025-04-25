#!/usr/bin/env python3
import sys
from pathlib import Path
import yaml
import logging

# Insert project root so "extractor" is importable
project_root = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(project_root))

# Load configuration from base config.yaml
config_path = project_root / "config.yaml"
if not config_path.exists():
    raise FileNotFoundError(f"Config file not found at {config_path}")
config = yaml.safe_load(config_path.open("r"))
PATHS = config.get("paths", {})

# Prepare logs directory
logs_dir = project_root / "logs"
logs_dir.mkdir(parents=True, exist_ok=True)

# Configure logging to console and file
log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=log_format)
logger = logging.getLogger(__name__)
file_handler = logging.FileHandler(logs_dir / "search.log")
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter(log_format))
logger.addHandler(file_handler)

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
    model_name: str,
    index_path: str = None,
    image_dir: str = None,
    top_k: int = None
) -> list[tuple[str, float]]:
    """
    Search for similar images using a FAISS index.

    model_name must be provided (e.g. from Streamlit UI).
    index_path and image_dir default to config paths, with MODEL_NAME substitution.
    """
    # Validate model_name
    if not model_name:
        logger.error("model_name must be provided")
        raise ValueError("model_name must be provided (e.g. from UI)")
    # Resolve paths, substituting MODEL_NAME placeholder
    name_key = model_name.split("/")[-1]
    index_path = index_path or PATHS.get("index", "").replace("${MODEL_NAME}", name_key)
    image_dir = image_dir or PATHS.get("images", "")
    top_k = top_k or config.get("model", {}).get("top_k", 5)

    if not index_path or not image_dir:
        logger.error("Index path or images directory not configured")
        raise ValueError("Both index_path and image_dir must be configured")

    logger.info("Loading FAISS index from %s", index_path)
    index = faiss.read_index(index_path)
    d_index = index.d

    logger.info("Extracting embedding for %s using model %s", query_image_path, model_name)
    query_emb = extract_single_image_embedding(
        query_image_path,
        model_name=model_name
    )
    query_emb = query_emb.unsqueeze(0).numpy().astype("float32")

    if query_emb.shape[1] != d_index:
        msg = f"Query embedding dim {query_emb.shape[1]} != index dim {d_index}"
        logger.error(msg)
        raise ValueError(msg)

    logger.info("Performing search for top %d results", top_k)
    distances, indices = index.search(query_emb, top_k)
    filenames = load_filenames(image_dir)

    if indices.max() >= len(filenames):
        msg = f"Index idx {indices.max()} >= number of files {len(filenames)}"
        logger.error(msg)
        raise RuntimeError(msg)

    results = [(filenames[i], float(distances[0][rank])) for rank, i in enumerate(indices[0])]
    logger.info("Search complete. Found %d results.", len(results))
    return results


if __name__ == "__main__":
    # Expect at least query_image_path and model_name
    if len(sys.argv) < 3 or len(sys.argv) > 6:
        logger.error(
            "Usage: python -m indexer.search <query_image_path> <model_name> [index_path] [image_dir] [top_k]"
        )
        sys.exit(1)

    query_path = sys.argv[1]
    model_name = sys.argv[2]
    index_path = sys.argv[3] if len(sys.argv) >= 4 else None
    image_dir = sys.argv[4] if len(sys.argv) >= 5 else None
    top_k = int(sys.argv[5]) if len(sys.argv) == 6 else None

    try:
        results = search_similar_images(
            query_image_path=query_path,
            model_name=model_name,
            index_path=index_path,
            image_dir=image_dir,
            top_k=top_k
        )
    except Exception:
        logger.exception("Search failed")
        sys.exit(1)

    logger.info("Top %d results:", len(results))
    for fn, dist in results:
        logger.info("%s (distance: %.4f)", fn, dist)
