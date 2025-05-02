from __future__ import annotations
import io
from pathlib import Path
from typing import Generator
from unittest import mock
import numpy as np
import pytest
import torch
from PIL import Image
from pytest import MonkeyPatch


# constants
EMB_DIM = 1024          # DINOv2‑base hidden size
DUMMY_VEC_T = torch.ones(EMB_DIM)           # torch tensor
DUMMY_VEC_N = np.ones((1, EMB_DIM), dtype="float32")  # numpy row‑vec



# tiny 1×1 PNG – avoids Pillow errors
@pytest.fixture(scope="session")
def tiny_png_bytes() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (1, 1), (255, 0, 0)).save(buf, format="PNG")
    return buf.getvalue()



# directory with three placeholder images (unit tests need paths)
@pytest.fixture(scope="session")
def image_dir(tmp_path_factory: pytest.TempPathFactory,
              tiny_png_bytes: bytes) -> str:
    root = tmp_path_factory.mktemp("images")
    for i in range(3):
        (root / f"img_{i}.png").write_bytes(tiny_png_bytes)
    return str(root)


# FAISS index stub – a 3‑vector IndexFlatL2 kept **in memory**
def _make_dummy_faiss_index() -> "faiss.Index":
    import faiss                    # lazy import to keep startup tiny
    idx = faiss.IndexFlatL2(EMB_DIM)
    idx.add(np.random.RandomState(0).rand(3, EMB_DIM).astype("float32"))
    return idx


# autouse = applies to **every** test                                         #
@pytest.fixture(autouse=True)
def patch_heavy_stack() -> Generator[None, None, None]:
    """
    1) Replaces extractor.extract.extract_single_image_embedding
    2) Replaces indexer.search._embed_cached
    3) Replaces indexer.search._load_index_cached
    Everything else (distance sort, FastAPI routing) still runs.
    """
    import extractor.extract as _ext
    import indexer.search as _srch

    mp = MonkeyPatch()

    # extractor – always return the dummy torch vector
    mp.setattr(
        _ext,
        "extract_single_image_embedding",
        lambda *a, **k: DUMMY_VEC_T.clone(),
        raising=False,
    )

    # embedder used by search.py
    mp.setattr(
        _srch,
        "_embed_cached",
        lambda *a, **k: DUMMY_VEC_N,
        raising=False,
    )

    # FAISS index loader
    mp.setattr(
        _srch,
        "_load_index_cached",
        lambda _path: _make_dummy_faiss_index(),
        raising=False,
    )

    yield
    mp.undo()
