#!/usr/bin/env python3
# ui/app.py
import sys
from pathlib import Path
import streamlit as st
import requests
import os
import shutil

# Ensure project root is importable for local modules
project_root = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(project_root))

# Direct imports instead of subprocess
from extractor.extract_dataset import batch_extract
from indexer.build_index import build_faiss_index

# Constants
API_URL = "http://localhost:8000/search/"
DATA_DIR = project_root / "data" / "raw"
EMB_PATH = project_root / "data" / "embeddings" / "embeddings.npy"
IDX_PATH = project_root / "data" / "index" / "index.faiss"

st.set_page_config(page_title="Semantic Image Search", layout="centered")
st.title("🔍 DINOv2-Powered Semantic Image Search")

# Tabs: Search | Setup
search_tab, setup_tab = st.tabs(["🔍 Search", "🗂️ Setup Dataset"])

# --- Search Tab ---
with search_tab:
    st.subheader("Upload a query image")
    uploaded = st.file_uploader("", type=["jpg", "jpeg", "png"])

    if uploaded:
        st.image(uploaded, caption="Query Image", use_container_width=True)
        top_k = st.slider("Number of results", 1, 10, 5)

        if st.button("Search"):
            files = {"file": (uploaded.name, uploaded.getvalue())}
            try:
                resp = requests.post(API_URL, files=files, params={"top_k": top_k}, timeout=10)
                resp.raise_for_status()
                results = resp.json().get("results", [])
                if not results:
                    st.warning("No similar images found.")
                else:
                    cols = st.columns(len(results))
                    for col, item in zip(cols, results):
                        img_path = DATA_DIR / item["filename"]
                        if img_path.exists():
                            col.image(str(img_path), caption=f"{item['filename']}\nDist={item['distance']:.2f}")
                        else:
                            col.write(f"{item['filename']} not found.")
            except Exception as e:
                st.error(f"Search failed: {e}")

# --- Setup Tab ---
with setup_tab:
    st.subheader("Upload images and build index")
    uploaded_files = st.file_uploader("", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

    if st.button("Clear dataset"):
        shutil.rmtree(DATA_DIR, ignore_errors=True)
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        st.success("Dataset cleared.")

    if uploaded_files:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        for file in uploaded_files:
            dest = DATA_DIR / file.name
            with open(dest, "wb") as f:
                f.write(file.getbuffer())
        st.success(f"Saved {len(uploaded_files)} images.")

    if st.button("Extract & Index"):
        with st.spinner("Extracting embeddings..."):
            # Directly call batch_extract
            batch_extract(str(DATA_DIR), str(EMB_PATH))
        with st.spinner("Building FAISS index..."):
            build_faiss_index(str(EMB_PATH), str(IDX_PATH))
        st.success("Dataset ready for search!")
