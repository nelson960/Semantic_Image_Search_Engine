"""
Key Ideas
- One-time loads + in-process caches ➜ avoid I/O on every query.

- Automatic GPU off-load            ➜ 30-60 × faster search when CUDA
  is available, with silent CPU fallback.

- Thread-safe locks                 ➜ safe for multi-request servers.

- LRU-cached embedding extractor    ➜ repeated queries for the same
  (model, image) pair are instant.
"""

from __future__ import annotations
import os
import sys
import threading
import functools
import logging
from pathlib import Path
import yaml
import faiss
import numpy as np
from transformers import logging as hf_logging

project_root = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(project_root))

# Load YAML config (index & image dirs, model defaults …)
config_path = project_root / "config.yaml"
if not config_path.exists():
    raise FileNotFoundError(f"Config file not found at {config_path}")
config = yaml.safe_load(config_path.open("r"))
PATHS = config.get("paths", {})

# Structured logging (console + rolling file)
logs_dir = project_root / "logs"
logs_dir.mkdir(parents=True, exist_ok=True)

LOG_FMT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FMT)
logger = logging.getLogger(__name__)
fh = logging.FileHandler(logs_dir / "search.log")
fh.setLevel(logging.INFO)
fh.setFormatter(logging.Formatter(LOG_FMT))
logger.addHandler(fh)

# Silence HuggingFace info/warning spam
hf_logging.set_verbosity_error()
from extractor.extract import extract_single_image_embedding 

# Globals & thread-safe caches
_INDEX_CACHE: dict[str, faiss.Index] = {}  # path → FAISS index
_FILENAMES_CACHE: dict[str, list[str]] = {} # img_dir → sorted list
_INDEX_LOCK = threading.Lock()        # serialises first index load

_GPU_RESOURCES = None
_HAS_GPU = faiss.get_num_gpus() > 0

if not _HAS_GPU:
    # Use every CPU core for FAISS kernels
    faiss.omp_set_num_threads(os.cpu_count())

# Helper: load & cache filenames
def _load_filenames_cached(image_dir: str) -> list[str]:
    """Fast, thread-safe, cached directory scan."""
    with _INDEX_LOCK:
        cached = _FILENAMES_CACHE.get(image_dir)
        if cached is not None:
            return cached

        base = Path(image_dir)
        exts = {".png", ".jpg", ".jpeg"}
        files = sorted(p for p in base.rglob("*") if p.suffix.lower() in exts)
        rels = [str(p.relative_to(base)) for p in files]
        _FILENAMES_CACHE[image_dir] = rels
   
        return rels
    
# Helper: load & optionally GPU-offload FAISS index
def _load_index_cached(index_path: str) -> faiss.Index:
    """Load once per process; move to GPU(s) if available."""
    with _INDEX_LOCK:
        cached = _INDEX_CACHE.get(index_path)
        if cached is not None:
            return cached

        logger.info("Loading FAISS index from %s", index_path)
        idx = faiss.read_index(index_path)

        if _HAS_GPU:
            global _GPU_RESOURCES
            if _GPU_RESOURCES is None:
                _GPU_RESOURCES = faiss.StandardGpuResources()

            try:
                # Shards automatically if you have >1 GPU
                idx = faiss.index_cpu_to_all_gpus(idx)
                logger.info("Index moved to %d GPU(s)", faiss.get_num_gpus())
            except Exception as e:
                # Fall back silently (GPU OOM, etc.)
                logger.warning("Could not move index to GPU (%s); staying on CPU", e)

        _INDEX_CACHE[index_path] = idx
        return idx

# Helper: cache embeddings (exact function signature preserved)
@functools.lru_cache(maxsize=4) # small cache: (model_name, image_path) keys
def _embed_cached(model_name: str, image_path: str) -> np.ndarray:
    """
    Wrap the user-supplied extractor in an LRU cache keyed by (model, image).
    """
    emb = extract_single_image_embedding(image_path, model_name=model_name)
    return emb.unsqueeze(0).numpy().astype("float32")


# Keep legacy alias around for external callers (no speed benefit)
def load_filenames(image_dir: str) -> list[str]:         
    return _load_filenames_cached(image_dir)


def search_similar_images(
    query_image_path: str,
    model_name: str,
    index_path: str | None = None,
    image_dir: str | None = None,
    top_k: int | None = None,
) -> list[tuple[str, float]]:
    
    if not model_name:
        logger.error("model_name must be provided")
        raise ValueError("model_name must be provided")

    name_key = model_name.split("/")[-1]
    index_path = index_path or PATHS.get("index", "").replace("${MODEL_NAME}", name_key)
    image_dir = image_dir or PATHS.get("images", "")
    top_k = top_k or config.get("model", {}).get("top_k", 5)
    if not index_path or not image_dir:
        raise ValueError("Both index_path and image_dir must be configured")

# hot-path: cached loads
    index = _load_index_cached(index_path)
    d_index = index.d
    query_emb = _embed_cached(model_name, query_image_path)

# Sanity check: embedding dim must equal index dim
    if query_emb.shape[1] != d_index:
        raise ValueError(f"Query embedding dim {query_emb.shape[1]} != index dim {d_index}")

# Exact L2 search (GPU-accelerated if available)
    distances, idxs = index.search(query_emb, top_k)


    filenames = _load_filenames_cached(image_dir)

# Range check (corrupt index vs. filename list)
    if idxs.max() >= len(filenames):
        raise RuntimeError(f"Index idx {idxs.max()} >= number of files {len(filenames)}")

    return [(filenames[i], float(distances[0][rank])) for rank, i in enumerate(idxs[0])]


if __name__ == "__main__":
    if len(sys.argv) < 3 or len(sys.argv) > 6:
        logger.error(
            "Usage: python -m indexer.search <query_image_path> <model_name> "
            "[index_path] [image_dir] [top_k]"
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
            top_k=top_k,
        )
    except Exception:
        logger.exception("Search failed")
        sys.exit(1)

    logger.info("Top %d results:", len(results))
    for fn, dist in results:
        logger.info("%s (distance: %.4f)", fn, dist)
