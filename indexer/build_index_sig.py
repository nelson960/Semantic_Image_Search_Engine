import numpy as np, faiss, os

def build_faiss_index_from_memmap(vec_npy_path, paths_npy_path, out_index):
    # 1. path list  ─ small, pickles OK
    paths = np.load(paths_npy_path, allow_pickle=True)
    n     = len(paths)

    # 2. mem‑map vectors ─ huge, never fully loaded
    vecs  = np.memmap(vec_npy_path, dtype="float32", mode="r")
    dim   = vecs.size // n
    vecs  = vecs.reshape(n, dim)

    # 3. build IP index (cosine on L2‑normed vecs)
    index = faiss.IndexFlatIP(dim)
    index.add(vecs)

    faiss.write_index(index, out_index)
    print(f"✓ {out_index}: {index.ntotal} × {dim} vectors")

# call with correct arguments (note the order)
build_faiss_index_from_memmap(
    vec_npy_path="tmp/siglip_vectors.npy",
    paths_npy_path="tmp/siglip_paths.npy",
    out_index="tmp/siglip_index.faiss",
)
