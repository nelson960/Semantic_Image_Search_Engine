from __future__ import annotations
import logging
import os
import sys
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import List
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware


project_root = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(project_root))
from indexer.search import search_similar_images 

# env-driven config 
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", 5))
MAX_UPLOAD_SIZE = MAX_UPLOAD_MB * 1024 * 1024
ALLOWED_EXT: set[str] = {
    ".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tiff", ".webp"
}
ALLOWED_ORIGINS: List[str] = os.getenv("CORS_ORIGINS", "*").split(",")

# logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("api.server")

# FastAPI & CORS
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# route behaviour
@app.post("/search/")
async def search_image(
    file: UploadFile = File(...),
    model_name: str = Query(..., description="DINOv2 model identifier, e.g. facebook/dinov2-base"),
    top_k: int = Query(5, ge=1, le=100, description="Number of top results"),
):
    # Trim model name (Swagger sometimes adds a trailing space)
    model_name = model_name.strip()

    # Extension & MIME validation
    if not file.filename:
        raise HTTPException(400, "Filename is missing in form part 'file'.")

    suffix = Path(file.filename).suffix.lower()
    if suffix not in ALLOWED_EXT:
        raise HTTPException(
            400,
            "Unsupported file extension; allowed: "
            + ", ".join(x.lstrip(".").upper() for x in sorted(ALLOWED_EXT)),
        )

    mime = file.content_type or ""
    if mime and not mime.startswith("image/"):
        logger.warning("Suspicious Content-Type %s for %s", mime, file.filename)
        raise HTTPException(400, "Content-Type must be image/*")

    # Read file
    contents = await file.read()
    if len(contents) > MAX_UPLOAD_SIZE:
        raise HTTPException(413, f"File too large; limit is {MAX_UPLOAD_MB} MiB.")

    # Write to temp file → run search in background thread
    tmp_path: str | None = None
    try:
        with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(contents)
            tmp_path = tmp.name

        logger.info("Searching with model=%s top_k=%d", model_name, top_k)
        results = await run_in_threadpool(
            search_similar_images,
            query_image_path=tmp_path,
            model_name=model_name,
            top_k=top_k,
        )

        return {"results": [{"filename": fn, "distance": dist} for fn, dist in results]}

    except HTTPException:
        raise
    except Exception:
        logger.exception("Unexpected error during /search/")
        raise HTTPException(500, "Internal server error")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                logger.warning("Failed to delete temp file %s", tmp_path)
