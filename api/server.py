# api/server.py
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from tempfile import NamedTemporaryFile
from pathlib import Path

# ensure project root is importable
import sys
project_root = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(project_root))

from indexer.search import search_similar_images

app = FastAPI()

# Optional: allow CORS from anywhere (e.g. Streamlit)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/search/")
async def search_image(file: UploadFile = File(...), top_k: int = 5):
    """
    Accepts an uploaded image, runs semantic search, and returns top_k matches.
    """
    # Validate file type
    suffix = Path(file.filename).suffix.lower()
    if suffix not in {".png", ".jpg", ".jpeg"}:
        raise HTTPException(status_code=400, detail="Unsupported file type. Use PNG or JPEG.")

    # Save uploaded image to a temp file
    with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        contents = await file.read()
        tmp.write(contents)
        tmp_path = tmp.name

    try:
        # Perform search (embeddings_path param removed)
        results = search_similar_images(
            query_image_path=tmp_path,
            index_path="data/index.faiss",
            image_dir="data/raw",
            top_k=top_k
        )
    except Exception as e:
        # Clean up temp file
        Path(tmp_path).unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=str(e))

    # Clean up temp file
    Path(tmp_path).unlink(missing_ok=True)

    return {"results": [
        {"filename": fname, "distance": dist}
        for fname, dist in results
    ]}
