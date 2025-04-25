# Semantic_Image_Search_Engine
Absolutely! Here’s the updated `README.md` with a **cURL command example** added under the API section — perfect for quick backend testing.

---

```markdown
# 🧠 DINOv2 Semantic Image Search Engine

This project is a full-stack image search engine that uses **DINOv2 Vision Transformers** for extracting semantic embeddings and **FAISS** for fast vector-based similarity search. It includes a minimal **FastAPI backend** and a **Streamlit frontend UI** for searching and managing datasets.

---

## 🚀 Features

- ✅ Semantic search using DINOv2 ViT embeddings
- 🔍 Upload image → Find top-K similar images
- 🧠 FAISS-powered approximate nearest neighbor search
- 🧾 Streamlit frontend with query + dataset setup tabs
- 🛠️ Modular, clean architecture (easy to extend or replace)

---

## 🧱 Project Structure

```
Semantic_Image_Search_Engine/
├── api/                        # FastAPI backend
│   └── server.py
├── data/
│   ├── raw/                    # Images for indexing (e.g., raw/train/*.png)
│   ├── embeddings/             # DINOv2 embeddings (e.g., .npy or .pt files)
│   └── index/                  # FAISS indexes
├── extractor/                  # DINOv2 embedding scripts
│   ├── extract.py              # Single image
│   └── extract_dataset.py      # Batch extraction
├── indexer/                    # FAISS indexing & search
│   ├── build_index.py
│   └── search.py
├── ui/                         # Streamlit frontend
│   └── app.py
├── utils/                      # Helper functions
│   └── (optional utilities)
├── Notebooks/                  # For experiments, testing, visualizations
├── requirements.txt
└── README.md

```

---

## ⚙️ Setup

### 1. Clone & Create Environment

```bash
git clone https://github.com/yourusername/semantic-search-engine
cd semantic-search-engine

python -m venv venv
source venv/bin/activate  # or .\venv\Scripts\activate on Windows
pip install -r requirements.txt
```

---

## 🚀 Run the Backend API

```bash
uvicorn api.server:app --reload
```

> API endpoint: `http://localhost:8000/search/`  
> Swagger UI is **disabled** by default.

### 🔁 Test the API via cURL

```bash
curl -X POST "http://localhost:8000/search/?top_k=5" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@path/to/query_image.jpg"
```

Replace `path/to/query_image.jpg` with the image file you want to use for testing.

---

## 💻 Streamlit Frontend

```bash
streamlit run ui/app.py
```

### Tabs:
- **🔍 Search**: Upload a query image and find the most similar images from your dataset.
- **🗂️ Setup Dataset**:
  - Upload new images to populate the dataset.
  - Run DINOv2 embedding extraction and FAISS indexing in one click.
  - Reset the dataset at any time.

---

## 🧠 How It Works

```
images → DINOv2-ViT → feature vectors → FAISS index
        ↑                                      ↓
   query image → embedding → nearest neighbors → show results
```

- DINOv2 extracts semantic features from images.
- FAISS handles fast nearest neighbor search.
- FastAPI exposes a simple `/search/` endpoint.
- Streamlit provides an intuitive UI.

---

## 🖼️ Use Cases

- Reverse image search
- Visual product recommendation
- Duplicate image detection
- Semantic clustering & tagging

---

## 📝 License

MIT License. ©️ Your Name 2025

---

## 🙌 Acknowledgments

- [Meta AI - DINOv2](https://github.com/facebookresearch/dinov2)
- [FAISS](https://github.com/facebookresearch/faiss)
- [Streamlit](https://streamlit.io)
