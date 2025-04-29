# 🧠 DINOv2 Semantic Image Search Engine

A modular, vision-centric semantic image search engine that uses **DINOv2** transformers for feature extraction and FAISS for high-speed semantic nearest-neighbor retrieval, handling images of any resolution from thumbnails to ultra-high-res.

## 🚀 Project Features

- Batch embedding extraction using DINOv2 (small, base, large models)
- FAISS index building for efficient semantic search
- Model comparison based on Precision@5 and Recall@5
- Interactive Streamlit-based frontend for uploading images and retrieving similar matches
- FastAPI backend for scalable, modular API architecture
- Automatic model selection, dynamic indexing
- Full evaluation pipeline with pseudo-labeling and t-SNE visualization

## 🗂 Dataset

TThis project relies on a curated slice of CIFAR-10 and a custom COCO split for demos and evaluation. CIFAR-10 contributes 60 000 color images at 32 × 32 px across 10 classes, while the custom COCO portion supplies 1000 images per class for 81 classes at 640 × 425 px. We use these datasets to:

- Generate semantic embeddings with DINOv2.  
- Form pseudo-labels through unsupervised K-Means clustering to emulate semantic groupings.  
- Gauge retrieval quality by verifying cluster consistency in the returned search results.

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


## 🔧 Installation

```bash
# Clone the repository
git clone https://github.com/nelson960/Semantic_Image_Search_Engine.git
cd Semantic_Image_Search_Engine

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
.venv\Scripts\activate    # Windows

# Install dependencies
pip install -r requirements.txt
```

---


## 🛠️ How It Works

1. Upload images (ZIP) via the Streamlit UI.
2. Extract embeddings using the selected DINOv2 model.
3. Build a FAISS index from the extracted embeddings.
4. Upload a query image.
5. The system searches the FAISS index and retrieves top-K semantically similar images.

![System Diagram](reports/system_diagram.png)
---

## 🔥 API Usage (FastAPI)

```bash
uvicorn api.server:app --reload
```

**Example cURL Command:**

```bash
curl -X POST "http://localhost:8000/search/?top_k=5" \
  -F "file=@/path/to/query_image.jpg"
```

---

## 💻 Frontend (Streamlit)

```bash
streamlit run ui/app.py
```

- Upload ZIP images
- Select DINOv2 model for embedding
- Extract and index in one click
- Upload and search images
- Visualize results in the UI

---

## 🔍 Use Cases

- **Reverse Image Search**
- **Product Recommendation Systems**
- **Content-Based Image Retrieval**
- **Semantic Clustering and Tagging**

---

## 📄 License

MIT License © 2025 Nelson960

---

## 🙏 Acknowledgments

- [Meta AI - DINOv2](https://github.com/facebookresearch/dinov2)
- [FAISS by Facebook AI](https://github.com/facebookresearch/faiss)
- [FastAPI](https://fastapi.tiangolo.com)
- [Streamlit](https://streamlit.io)

