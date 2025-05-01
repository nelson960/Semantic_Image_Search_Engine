"""
Key Ideas
────────────
- mmap load  – zero-copy file read; no RAM spike on 10-GB arrays.

- dtype short-circuit – skip astype if already float32.

- All-core CPU  – faiss.omp_set_num_threads() lets FAISS saturate
   the memory bus on NUMA machines.

- GPU fast-path – if CUDA is visible, vectors are batched into
   GpuIndexFlatL2, then moved back to CPU so downstream code still gets
   a plain IndexFlatL2 (portable across boxes).
"""

import numpy as np
import faiss
from pathlib import Path
import multiprocessing as mp


def _load_embeddings(fp: str) -> np.ndarray:
    """
    Memory-map the .npy file (no copy) and convert to float32
    *only* if the stored dtype is not already float32.
    """
    arr = np.load(fp, mmap_mode="r")  # zero-copy read
    if arr.dtype != np.float32:
        arr = arr.astype(np.float32, copy=False)
    return arr


def _add_to_gpu(emb: np.ndarray, dim: int) -> faiss.IndexFlatL2:
    """
    Build on GPU, then copy to host so the index file loads anywhere,
    even on machines without CUDA.
    """
    res = faiss.StandardGpuResources()  # one-off initialisation
    gpu_index = faiss.GpuIndexFlatL2(res, dim)

    # Tune this to your GPU RAM; 1M vectors is fine for 16 GB cards.
    batch = 1_000_000
    for i in range(0, emb.shape[0], batch):
        gpu_index.add(emb[i : i + batch])

    cpu_index = faiss.index_gpu_to_cpu(gpu_index)  # identical to IndexFlatL2
    return cpu_index


def _add_to_cpu(emb: np.ndarray, dim: int) -> faiss.IndexFlatL2:
    """
    Multi-threaded CPU build.
    """
    faiss.omp_set_num_threads(mp.cpu_count())
    index = faiss.IndexFlatL2(dim)
    index.add(emb)                       # FAISS will parallelise internally
    return index


def build_faiss_index(embedding_path: str, index_path: str):
    emb = _load_embeddings(embedding_path)
    dim = emb.shape[1]

    if faiss.get_num_gpus():             # GPU available?
        index = _add_to_gpu(emb, dim)
    else:
        index = _add_to_cpu(emb, dim)

    out = Path(index_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(out))

    print(
        f"FAISS IndexFlatL2 built with {emb.shape[0]:,} vectors "
        f"(dim={dim}) → {out}"
    )


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3:
        print("Usage: python build_index.py <embeddings.npy> <index_output_path>")
        sys.exit(1)
    build_faiss_index(sys.argv[1], sys.argv[2])
