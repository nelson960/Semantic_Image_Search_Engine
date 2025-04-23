import numpy as np
import faiss
from pathlib import Path

def build_faiss_index(embedding_path: str, index_path: str):
    """
    Load a NumPy .npy of shape [N, D], build an L2 FAISS index,
    and write it to disk.
    """
    embedding_path = Path(embedding_path)
    if embedding_path.suffix.lower() != ".npy":
        raise ValueError(f"Expected a .npy file, got {embedding_path.suffix}")

    # 1. load embeddings as float32
    embeddings = np.load(embedding_path)
    embeddings = embeddings.astype("float32")  # FAISS needs float32

    # 2. build a flat (L2) index
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    # 3. make sure output dir exists
    output = Path(index_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    # 4. write it out
    faiss.write_index(index, str(output))
    print(f"FAISS index built with {embeddings.shape[0]} vectors (dim={dim}), saved to {output}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python build_index.py <embeddings.npy> <index_output_path>")
        sys.exit(1)

    build_faiss_index(sys.argv[1], sys.argv[2])
