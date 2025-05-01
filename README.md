# 🧠 DINOv2 Semantic Image Search Engine

A modular semantic image search engine powered by DINOv2 Vision Transformers for self-supervised feature extraction and FAISS for fast, scalable nearest-neighbor search. Supports any image resolution — from thumbnails to ultra-high-res.
Built to demonstrate full-stack computer vision engineering: feature learning, vector indexing, REST API, and UI.
Fully reproducible with clean MLOps integration.

## 🚀 Project Features

- **Human-level semantics** – Leverages Meta’s DINOv2 (small/base/large) to embed images of any resolution into a rich semantic space.
- **Sub-second retrieval at scale** – FAISS IVF-Flat index enables top‑K semantic search over millions of images in ~50 ms on a laptop.
- **Scalable architecture** – FastAPI-powered backend supports batch embedding extraction, dynamic indexing, and automatic model selection.
- **Modular evaluation** – Includes precision/recall benchmarking, t-SNE visualizations, and pseudo-labeling for performance insight.
- **Production-ready stack** – Async FastAPI + Pydantic behind Docker Compose; all requests validated, unit-tested, and logged.
- **No-code frontend** – Streamlit UI for uploading images, building indexes, and interactively exploring results.

## Datasets for Fine-Tuning & Evaluation
Fine-tuning:
Used the **unlabeled STL-10** dataset (100k images, originally 96×96, resized to 224×224) for efficient domain adaptation via **LoRA adapters** and **multi-crop self-distillation** on a ViT-Base backbone using just 16 GB of GPU memory.

**Testing & Evaluation:**

- **CIFAR-10:** 60k low-res (32×32) images across 10 classes.

- **Custom MS-COCO Split:** 81 classes with 2k high-res (640×425) images per class.

- **Misc. high-res sets:** Natural images for qualitative checks.

**Evaluation Methods:**

- Generate DINOv2 embeddings across all sets.

- Use **K-Means pseudo-labeling** for unsupervised grouping.

- Assess retrieval quality with **Precision@5, Recall@5,** and cluster consistency.

## 📈 Fine-tuning
fine-tunes a pre-trained DINOv2 ViT-base on the STL-10 data using DINO-style self-distillation, but does so with a series of memory-savvy tricks so it can run comfortably on a 16 GB
**MacBook Pro M2 (GPU = mps):**

 - **Hardware constraints first:** the script automatically chooses mps (Apple-Silicon GPU) and caps the batch-size at 8, a setting you note is “safe for 16 GB”; all other choices—from pin-memory to modest data-loader workers—are tuned to avoid out-of-memory errors while still delivering GPU acceleration.

 - **Parameter-efficient fine-tuning:** every ViT weight is frozen and LoRA adapters are injected only into the QKV and MLP layers; together with a three-layer projection head, this keeps the trainable set tiny and slashes memory/compute versus full fine-tuning.

 - **Multi-crop training à la DINO:** each sample produces 2 global (224 × 224) and several local (96 × 96) crops; the local crops are concatenated into a single tensor to maximize throughput on the limited GPU RAM.

 - **Student–teacher setup with EMA:** the teacher backbone/head are updated by exponential moving average after every step, so the extra forward pass adds almost no memory overhead (done under torch.no_grad()).

 - **Stability aids for small hardware:** gradients are clipped at 3.0, AdamW uses a modest 6 e-4 LR, and all non-LoRA parameters (LayerNorms, embeddings, etc.) are explicitly frozen to save RAM and avoid accidental updates.

## 📊 DINOv2 Model Comparison (Evaluation Report)

| Model         | Precision@5 (%) | Recall@5 (%) |
|---------------|-----------------|--------------|
| dinov2-small  | 89.8             | 4.4          |
| dinov2-base   | 93.8             | 4.6          |
| dinov2-large  | 95.8             | 4.8          |

### 🔍 Observations
- **Precision** improves consistently as model size increases.
- **Recall** remains stable due to clustering approximation.
- **Tradeoff**: `dinov2-large` provides best semantic retrieval at the cost of inference speed.

![Model Comparison](reports/model_comparison.png)


## 🔧 Local Development

```bash
# Clone the repository
git clone https://github.com/nelson960/Semantic_Image_Search_Engine.git
cd Semantic_Image_Search_Engine

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
pip install -r requirements.txt # Install dependencies
uvicorn api.server:app --reload #API at :8000
streamlit run ui/app.py          # UI at :8501
```
**Example cURL Command:**

```bash
curl -X POST "http://localhost:8000/search/?top_k=5" \
  -F "file=@/path/to/query_image.jpg"
```
---
## 💻 Frontend (Streamlit)

- Upload ZIP images
- Select DINOv2 model for embedding
- Extract and index in one click
- Upload and search images
- Visualize results in the UI

---

## 🛠️ How It Works

1. Upload images (ZIP) via the Streamlit UI.
2. Extract embeddings using the selected DINOv2 model.
3. Build a FAISS index from the extracted embeddings.
4. Upload a query image.
5. The system searches the FAISS index and retrieves top-K semantically similar images.

![System Diagram](reports/system_diagram.png)
---


## 🔍 Use Cases

- **Reverse Image Search**
- **Product Recommendation Systems**
- **Content-Based Image Retrieval**
- **Semantic Clustering and Tagging**
- **Copyright‑infringement detection**
- **Deduplication & clustering of large photo archives**


---

## 📄 License

MIT License © 2025 Nelson960

---

## 🙏 Acknowledgments

- [Meta AI - DINOv2](https://github.com/facebookresearch/dinov2)
- [FAISS by Facebook AI](https://github.com/facebookresearch/faiss)
- [FastAPI](https://fastapi.tiangolo.com)
- [Streamlit](https://streamlit.io)

