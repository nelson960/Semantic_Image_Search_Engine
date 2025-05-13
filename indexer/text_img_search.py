#!/usr/bin/env python

import numpy as np
import torch
import faiss
from pathlib import Path
from transformers import SiglipProcessor, SiglipModel
from collections import OrderedDict
import gensim
import gensim.downloader as api


def search_siglip_with_fallback(
    query: str,
    ckpt: str,
    faiss_index_path: str,
    paths_npy_path: str,
    w2v_source: str = "glove-wiki-gigaword-100",
    top_k: int = 6
) -> list[tuple[str, float]]:
    """
    Perform text-to-image search with fallback via Word2Vec expansion if match is weak.

    Args:
        query: Text prompt.
        ckpt: Hugging Face checkpoint path.
        faiss_index_path: Path to FAISS index (.faiss).
        paths_npy_path: Path to .npy file with image paths.
        w2v_source: Gensim model name or .bin/.kv file path for word vectors.
        top_k: Number of top results to return.

    Returns:
        List of (image_path, cosine_score) tuples.
    """
    device = (
        "cuda" if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else
        "cpu"
    )

    processor = SiglipProcessor.from_pretrained(ckpt, use_fast=False)
    model = SiglipModel.from_pretrained(
        ckpt,
        torch_dtype=torch.float16 if device.startswith("cuda") else torch.float32
    ).to(device).eval()

    def embed(text: str):
        tokens = processor(text=[text], return_tensors="pt").to(device)
        with torch.inference_mode():
            vec = model.get_text_features(**tokens)
            vec = vec / vec.norm(dim=-1, keepdim=True)
            return vec.cpu().numpy().astype("float32")

    def expand_with_w2v(text: str, w2v_model, topn=4):
        words = [w.lower() for w in text.split() if w.isalpha() and w in w2v_model]
        expansions = OrderedDict()
        for word in words:
            for alt, _ in w2v_model.most_similar(word, topn=topn):
                expansions[alt] = None
        return list(expansions.keys())

    # Load resources
    index = faiss.read_index(faiss_index_path)
    paths = np.load(paths_npy_path, allow_pickle=True)

    if Path(w2v_source).exists():
        w2v = gensim.models.KeyedVectors.load(w2v_source)
    else:
        w2v = api.load(w2v_source)

    # Step 1: Direct search
    query_vec = embed(query)
    D, I = index.search(query_vec, top_k)
    best_score = float(D[0, 0])
    final_query = query

    # Step 2: Fallback via W2V expansions
    if best_score < 0.40:
        expansions = expand_with_w2v(query, w2v)
        for alt in expansions:
            alt_vec = embed(alt)
            d2, i2 = index.search(alt_vec, top_k)
            if float(d2[0, 0]) > best_score:
                best_score, D, I = float(d2[0, 0]), d2, i2
                final_query += f" | fallback:{alt}"
            if best_score >= 0.40:
                break

    return [(str(paths[int(idx)]), float(score)) for idx, score in zip(I[0], D[0])]



results = search_siglip_with_fallback(
    query="red car",
    ckpt="google/siglip-so400m-patch14-384",
    faiss_index_path="tmp/siglip_index.faiss",
    paths_npy_path="tmp/siglip_paths.npy",
    w2v_source="glove-wiki-gigaword-100",  # or path to .kv/.bin file
    top_k=5
)

for i, (path, score) in enumerate(results, 1):
    print(f"{i}. {path} (cosine={score:.4f})")
