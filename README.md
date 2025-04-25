# 🧠 DINOv2 Semantic Image Search Engine

A full-stack, modular image search engine powered by **DINOv2** vision transformers and **FAISS** for lightning-fast, semantic nearest-neighbor search. Includes:

- A **FastAPI** backend (`/api/server.py`)  
- A **Streamlit** frontend (`/ui/app.py`)  
- Batch embedding extraction scripts (`/extractor/`)  
- FAISS index builder & search routines (`/indexer/`)  
- Centralized **config.yaml** for all path settings  
- File-and-error-aware ZIP upload & directory clearing  
- Automatic model-index discovery in the UI  
- Comprehensive **logging** to `logs/streamlit.log` and console

---

## 📁 Project Layout

```
Semantic_Image_Search_Engine/
├── api/                        # FastAPI backend
│   └── server.py
├── data/
│   ├── raw/train/             # Extracted images (flattened)
│   ├── embeddings/            # .npy embeddings per model
│   └── index/                 # .faiss indexes per model
├── extractor/                 # DINOv2 embedding scripts
│   ├── extract.py             # single-image embedding
│   └── extract_dataset.py     # batch extraction (loads model once!)
├── indexer/                   # FAISS routines
│   ├── build_index.py         # build .faiss from .npy
│   └── search.py              # search with dimension-check & model_name
├── ui/                        # Streamlit app
│   └── app.py                 # front-end with config + logging + error handling
├── config.yaml                # path templates & API URL
├── logs/                      # streamlit & (future) server logs
├── Notebooks/                 # experiments & prototyping
├── requirements.txt
└── README.md
```

---

## ⚙️ Configuration

All runtime paths are driven by **config.yaml** in the repo root.  It uses `${MODEL_NAME}` placeholders so that one file fits all models:

```yaml
paths:
  raw:        "data/raw"
  images:     "data/raw/train"
  embeddings: "data/embeddings/${MODEL_NAME}.npy"
  index:      "data/index/${MODEL_NAME}.faiss"
  logs:       "logs"
api:
  url: "http://localhost:8000/search/"
```

- **raw**: parent of the `train/` folder holding your unzipped images.  
- **embeddings** & **index**: templates—Streamlit replaces `${MODEL_NAME}` at runtime.  
- **logs**: where Streamlit writes `streamlit.log`.  
- **api.url**: backend search endpoint.

---

## 🔧 Installation

1. **Clone & create venv**  
   ```bash
   git clone https://github.com/nelson960/Semantic_Image_Search_Engine.git
   cd Semantic_Image_Search_Engine
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Edit `config.yaml`** if you want to change data or log locations.

---

## 🚀 Backend (FastAPI)

```bash
uvicorn api.server:app --reload
```

- **Endpoint**: `POST /search/?model_name=<model>&top_k=<k>`  
- **Params**:  
  - `model_name` (e.g. `facebook/dino-v2-small`)  
  - `top_k` (integer, default 5)  
- **Body**: multipart-encoded file field named `file`

### cURL example

```bash
curl -X POST "http://localhost:8000/search/?model_name=facebook/dino-v2-small&top_k=5" \
  -F "file=@/path/to/image.jpg"
```

---

## 💻 Frontend (Streamlit)

```bash
streamlit run ui/app.py
```

### Tabs

- **🔍 Search**  
  - **Auto-detects** the one `.faiss` index in `data/index/` and infers  
    `MODEL_NAME` from its filename (e.g. `dinov2-base`).  
  - Validates “no index → error” and “no upload → info”.  
  - Sends `model_name` and `top_k` to the API, displays top-K matches.

- **🗂️ Setup Dataset**  
  1. **Upload** a single ZIP of images → extracts *all* files  
     (flattened) into `data/raw/train/`.  
  2. **Clear all data** button wipes `data/raw`, `data/embeddings`, `data/index`  
     (recreates empty folders).  
  3. **Select model** for indexing, then **Extract & Index** →  
     runs `batch_extract(...)` once-per-run, builds `.faiss`.

### Logging & Errors

- All UI events, errors, and stack traces go to `logs/streamlit.log` and stdout.  
- User-facing errors (missing ZIP, missing index, unsupported files) show via `st.error` or `st.warning`.

---

## 🧠 How It Works

1. **Setup** tab populates **`data/raw/train/`**, builds:
   - `.npy` embeddings via DINOv2
   - `.faiss` index via FAISS  
2. **Search** tab uploads a query image →  
   DINOv2 embedding → FAISS search → returns nearest image filenames + distances →  
   Streamlit displays them from `train/`.

---

## 🔍 Use Cases

- **Reverse image search**  
- **Product similarity**  
- **Semantic clustering & tagging**  

---

## 📄 License

MIT © 2025 Nelson960

---

## 🙏 Acknowledgments

- [Meta AI – DINOv2](https://github.com/facebookresearch/dinov2)  
- [FAISS](https://github.com/facebookresearch/faiss)  
- [FastAPI](https://fastapi.tiangolo.com)  
- [Streamlit](https://streamlit.io)  
```

This README reflects:

- **config.yaml**-driven paths with `${MODEL_NAME}`  
- **Auto-detection** of model/index in the UI  
- **Batch extraction** loading model & processor once  
- **Comprehensive logging**  
- **Error handling** for missing ZIPs, empty datasets, wrong model/index  
- **cURL** example for quick API testing.