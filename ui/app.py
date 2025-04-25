#!/usr/bin/env python3
# ui/app.py

import sys
import logging
import yaml
from pathlib import Path
import streamlit as st
import requests
import shutil
import zipfile
import io

# ——— PYTHONPATH for local imports ———
project_root = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(project_root))

# ——— Load config.yaml ———
cfg = yaml.safe_load(open(project_root / "config.yaml"))
paths = cfg["paths"]
API_URL = cfg["api"]["url"]

# ——— Resolve dirs/templates ———
index_tmpl      = paths["index"]      # e.g. "data/index/${MODEL_NAME}.faiss"
emb_tmpl        = paths["embeddings"] # e.g. "data/embeddings/${MODEL_NAME}.npy"
train_dir       = project_root / paths["images"]
raw_dir         = train_dir.parent
logs_dir        = project_root / paths["logs"]

# ——— Logging setup ———
logs_dir.mkdir(parents=True, exist_ok=True)
log_file = logs_dir / "streamlit.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# ——— Local imports ———
from extractor.extract_dataset import batch_extract
from indexer.build_index      import build_faiss_index

# ——— Ensure base dirs ———
for d in (raw_dir, train_dir):
    d.mkdir(parents=True, exist_ok=True)

st.set_page_config(page_title="Semantic Image Search", layout="centered")
st.title("🔍 DINOv2-Powered Semantic Image Search")

search_tab, setup_tab = st.tabs(["🔍 Search", "🗂️ Setup Dataset"])

# --- SEARCH TAB (auto‐detect model/index) ---
with search_tab:
    st.subheader("Upload a query image")

    # 1) Find all .faiss files in index directory
    idx_dir = project_root / Path(index_tmpl).parent
    faiss_files = list(idx_dir.glob("*.faiss"))

    if not faiss_files:
        st.error("No FAISS index found. Please build one in the Setup tab first.")
    else:
        if len(faiss_files) > 1:
            st.warning(
                "Multiple indexes detected. "
                f"Using '{faiss_files[0].name}'."
            )

        idx_path     = faiss_files[0]
        model_key    = idx_path.stem             # e.g. "dinov2-small"
        model_name   = f"facebook/{model_key}"
        st.info(f"Using index for model: **{model_key}**")

        uploaded = st.file_uploader("", type=["jpg", "jpeg", "png"])
        if not uploaded:
            st.info("Please upload an image to search.")
        else:
            # only when uploaded is present do we show the image, slider & button
            st.image(uploaded, caption="Query Image", use_container_width=True)
            top_k = st.slider("Number of results", 1, 10, 5)

            if st.button("Search"):
                logger.info("Search: model=%s top_k=%d", model_name, top_k)
                files  = {"file": (uploaded.name, uploaded.getvalue())}
                params = {"model_name": model_name, "top_k": top_k}
                try:
                    resp = requests.post(API_URL, files=files, params=params, timeout=10)
                    resp.raise_for_status()
                    results = resp.json().get("results", [])
                    logger.info("Received %d results", len(results))

                    if not results:
                        st.warning("No similar images found.")
                    else:
                        cols = st.columns(len(results))
                        for col, item in zip(cols, results):
                            img_path = train_dir / item["filename"]
                            if img_path.exists():
                                col.image(
                                    str(img_path),
                                    caption=f"{item['filename']}\nDist={item['distance']:.2f}"
                                )
                            else:
                                col.write(f"{item['filename']} not found.")
                except Exception as e:
                    logger.error("Search failed", exc_info=True)
                    st.error(f"Search failed: {e}")

# --- SETUP TAB (unchanged) ---
with setup_tab:
    st.subheader("Upload a ZIP of images and build index")
    uploaded_zip = st.file_uploader("", type=["zip"])

    model_choice = st.selectbox(
        "Select DINOv2 model for indexing",
        ["facebook/dinov2-small", "facebook/dinov2-base", "facebook/dinov2-large"]
    )
    model_key = model_choice.split("/")[-1]
    emb_path   = project_root / emb_tmpl.replace("${MODEL_NAME}", model_key)
    idx_path   = project_root / index_tmpl.replace("${MODEL_NAME}", model_key)

    if st.button("Clear all data"):
        logger.info("Clearing all data")
        shutil.rmtree(raw_dir, ignore_errors=True)
        shutil.rmtree(emb_path.parent, ignore_errors=True)
        shutil.rmtree(idx_path.parent, ignore_errors=True)
        for d in (raw_dir, train_dir, emb_path.parent, idx_path.parent):
            d.mkdir(parents=True, exist_ok=True)
        st.success("Cleared raw, embeddings, and index.")

    if uploaded_zip:
        logger.info("Extracting ZIP to %s", train_dir)
        shutil.rmtree(train_dir, ignore_errors=True)
        train_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(io.BytesIO(uploaded_zip.getvalue())) as z:
            for member in z.infolist():
                if not member.is_dir():
                    dest = train_dir / Path(member.filename).name
                    dest.write_bytes(z.read(member.filename))
        st.success("Extracted images to train directory.")

    if st.button("Extract & Index"):
        images = list(train_dir.glob("*"))
        if not images:
            st.error("No images to index—please upload and extract a ZIP first.")
        else:
            logger.info("Extract & Index (model=%s)", model_choice)
            emb_path.parent.mkdir(parents=True, exist_ok=True)
            idx_path.parent.mkdir(parents=True, exist_ok=True)
            with st.spinner("Extracting embeddings..."):
                batch_extract(str(train_dir), str(emb_path), model_name=model_choice)
            with st.spinner("Building FAISS index..."):
                build_faiss_index(str(emb_path), str(idx_path))
            logger.info("Done Extract & Index: %s, %s", emb_path, idx_path)
            st.success(f"Dataset ready for search with model '{model_key}'!")
