#!/usr/bin/env python3
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from tempfile import NamedTemporaryFile
from pathlib import Path
import logging

# Ensure project root is importable
import sys
project_root = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(project_root))

from indexer.search import search_similar_images

# Configure logging for the server
logging.basicConfig(level=logging.INFO)
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

@app.post("/search/")
async def search_image(
    file: UploadFile = File(...),
    model_name: str = Query(..., description="DINOv2 model identifier e.g. facebook/dinov2-base"),
    top_k: int = Query(5, description="Number of top results to return")
):
    """
    Accepts an uploaded image, runs semantic search with the specified DINOv2 model,
    and returns the top_k matches.
    """
    # Validate file type
    suffix = Path(file.filename).suffix.lower()
    if suffix not in {".png", ".jpg", ".jpeg"}:
        raise HTTPException(status_code=400, detail="Unsupported file type. Use PNG or JPEG.")

    # Save uploaded file to temporary path
    with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        contents = await file.read()
        tmp.write(contents)
        tmp_path = tmp.name

    try:
        # Perform search, model_name is mandatory
        logger.info("Searching for similar images using model %s", model_name)
        results = search_similar_images(
            query_image_path=tmp_path,
            model_name=model_name,
            top_k=top_k
        )
    except Exception as e:
        logger.error("Search failed: %s", e, exc_info=True)
        # Clean up temp file
        Path(tmp_path).unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=str(e))

    # Clean up temp file
    Path(tmp_path).unlink(missing_ok=True)

    # Format and return
    return {"results": [
        {"filename": fname, "distance": dist}
        for fname, dist in results
    ]}
