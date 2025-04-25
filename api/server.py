#!/usr/bin/env python3
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from tempfile import NamedTemporaryFile
from pathlib import Path
import logging
import os

# Ensure project root is importable
import sys
project_root = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(project_root))

from indexer.search import search_similar_images

# Configure logging for the server
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("api.server")

app = FastAPI()

# Allow CORS (e.g. from Streamlit UI)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Upload constraints
MAX_UPLOAD_SIZE = 5 * 1024 * 1024  # 5 MiB
ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tiff", ".webp"}

@app.post("/search/")
async def search_image(
    file: UploadFile = File(...),
    model_name: str = Query(..., description="DINOv2 model identifier e.g. facebook/dinov2-base"),
    top_k: int = Query(5, ge=1, le=100, description="Number of top results to return")
):
    """
    Accepts an uploaded image, runs semantic search with the specified DINOv2 model,
    and returns the top_k matches.
    """
    suffix = Path(file.filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file extension. Allowed: PNG, JPG, JPEG, BMP, GIF, TIFF, WEBP."
        )

    contents = await file.read()
    if len(contents) > MAX_UPLOAD_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size is {MAX_UPLOAD_SIZE // (1024*1024)} MiB."
        )

    tmp_path = None
    try:
        tmp = NamedTemporaryFile(delete=False, suffix=suffix)
        tmp_path = tmp.name
        tmp.write(contents)
        tmp.close()

        logger.info("Searching with model=%s top_k=%d", model_name, top_k)
        results = search_similar_images(
            query_image_path=tmp_path,
            model_name=model_name,
            top_k=top_k
        )

        return {
            "results": [
                {"filename": fn, "distance": dist}
                for fn, dist in results
            ]
        }

    except HTTPException:
        # pass through 4xx errors
        raise
    except Exception:
        logger.exception("Unexpected error during /search/")
        raise HTTPException(status_code=500, detail="Internal server error.")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                logger.warning("Failed to delete temp file %s", tmp_path)
