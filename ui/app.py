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

# ——— Load config.yaml —————
cfg        = yaml.safe_load(open(project_root / "config.yaml"))
paths      = cfg["paths"]
API_URL    = cfg["api"]["url"]
index_tmpl = paths["index"]
emb_tmpl   = paths["embeddings"]
train_dir  = project_root / paths["images"]
raw_dir    = train_dir.parent
logs_dir   = project_root / paths["logs"]

# ——— Page config + Tailwind injection —————
st.set_page_config(page_title="DINOv2 Semantic Search", layout="wide")
st.markdown(
    """
    <!-- Tailwind CSS -->
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
      /* stretch the main container */
      .css-18e3th9 { max-width: 95% !important; }
      /* tweak default padding */
      .css-1d391kg { padding-top:1rem !important; padding-bottom:1rem !important; }
    </style>
    """,
    unsafe_allow_html=True
)

# ——— Logging setup —————
logs_dir.mkdir(parents=True, exist_ok=True)
log_file = logs_dir / "streamlit.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# ——— Local imports —————
from extractor.extract_dataset import batch_extract
from indexer.build_index      import build_faiss_index

# ——— Ensure base dirs —————
for d in (raw_dir, train_dir):
    d.mkdir(parents=True, exist_ok=True)

# ——— Header —————
st.title("🔍 DINOv2 Semantic Image Search")
st.markdown(
    """
    <div class="bg-blue-50 border-l-4 border-blue-400 p-4 mb-6">
      <p class="font-medium">
        This demo uses <strong>DINOv2</strong> for embeddings and <strong>FAISS</strong> for search.
        Use the tabs below to <em>search</em> or <em>build</em> your dataset.
      </p>
    </div>
    """,
    unsafe_allow_html=True
)

search_tab, setup_tab = st.tabs(["🔍 Search", "🗂️ Setup Dataset"])

# --- SEARCH TAB 
with search_tab:
    st.markdown('<div class="bg-white shadow-md rounded-lg p-6">', unsafe_allow_html=True)
    st.subheader("Query Image")
    st.markdown("Uploads your query image, Detects similar images")

    # ——— locate FAISS index ———
    idx_dir     = project_root / Path(index_tmpl).parent
    faiss_files = list(idx_dir.glob("*.faiss"))

    if not faiss_files:
        st.error("⚠️ No FAISS index found. Build one in **Setup Dataset** first.")
    else:
        if len(faiss_files) > 1:
            st.warning(f"Multiple indexes detected—using `{faiss_files[0].name}`.")
        idx_path   = faiss_files[0]
        model_key  = idx_path.stem
        model_name = f"facebook/{model_key}"
        st.info(f"**{model_key}**")

        uploaded = st.file_uploader("📂 Select an image", type=["jpg", "jpeg", "png"])
        if uploaded:

            # ── preview + controls (single row)
            #     ratio 1 : 1 + small gap  → slider moves closer to preview
            ctrl_cols = st.columns([1, 1], gap="small")
            with ctrl_cols[0]:
                st.image(uploaded, width=300, caption="Query preview")   # bigger preview
            with ctrl_cols[1]:
                top_k = st.select_slider("Number of results", options=list(range(1,21)), value=10)
                do_search = st.button("Search")

            # ── run search
            if do_search:
                logger.info("Search: model=%s, top_k=%d", model_name, top_k)
                try:
                    resp = requests.post(
                        API_URL,
                        files={"file": (uploaded.name, uploaded.getvalue())},
                        params={"model_name": model_name, "top_k": top_k},
                        timeout=10
                    )
                    resp.raise_for_status()
                    results = resp.json().get("results", [])
                    logger.info("Received %d results", len(results))

                    if not results:
                        st.warning("No similar images found.")
                    else:
                        st.subheader("🔎 Results")

                        # layout constants  (preview > thumb size)
                        THUMBS_PER_ROW = 4      # thumbnails per row
                        THUMB_WIDTH    = 350    # px  (smaller than preview)
                        ROW_GAP_HTML   = "<div style='height:0.75rem'></div>"

                        # grid without preview duplication
                        for start in range(0, len(results), THUMBS_PER_ROW):
                            row_cols = st.columns(THUMBS_PER_ROW, gap="medium")
                            for col, item in zip(row_cols, results[start:start + THUMBS_PER_ROW]):
                                img_path = train_dir / item["filename"]
                                if img_path.exists():
                                    col.image(str(img_path), width=THUMB_WIDTH)
                                else:
                                    col.write("❌ Not found")
                            st.markdown(ROW_GAP_HTML, unsafe_allow_html=True)

                # ——— unchanged error handling ———
                except requests.HTTPError as http_err:
                    st.error(f"Search failed: {http_err}")
                    try:
                        detail = http_err.response.json().get("detail", "")
                    except Exception:
                        detail = http_err.response.text
                    st.error(f"Search failed: {detail or http_err}")
                except Exception as e:
                    logger.error("Search failed", exc_info=True)
                    st.error(f"Search failed: {e}")

    st.markdown('</div>', unsafe_allow_html=True)



# --- SETUP TAB ---
with setup_tab:
    st.markdown('<div class="bg-white shadow-md rounded-lg p-6">', unsafe_allow_html=True)
    st.subheader("Upload Image Dataset")

    col1, col2 = st.columns([3,1], gap="large")
    with col1:
        uploaded_zip = st.file_uploader("ZIP file", type=["zip"])
    with col2:
        st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)
        if st.button("Clear all data"):
            logger.info("Clearing raw, embeddings, index")
            shutil.rmtree(raw_dir, ignore_errors=True)
            shutil.rmtree((project_root / emb_tmpl.replace("${MODEL_NAME}", "")).parent, ignore_errors=True)
            shutil.rmtree((project_root / index_tmpl.replace("${MODEL_NAME}", "")).parent, ignore_errors=True)
            for d in (raw_dir, train_dir):
                d.mkdir(parents=True, exist_ok=True)
            st.success("✔️ All data cleared")

    if uploaded_zip:
        logger.info("Extracting ZIP to %s", train_dir)
        shutil.rmtree(train_dir, ignore_errors=True)
        train_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(io.BytesIO(uploaded_zip.getvalue())) as z:
            for m in z.infolist():
                if not m.is_dir():
                    dest = train_dir / Path(m.filename).name
                    dest.write_bytes(z.read(m.filename))
        count = len(list(train_dir.glob("*")))
        st.success(f"✔️ Extracted {count} images")

    st.markdown("### Build & Index", unsafe_allow_html=True)
    col3, col4 = st.columns(2, gap="large")
    with col3:
        model_choice = st.selectbox(
            "Model",
            ["facebook/dinov2-small", "facebook/dinov2-base", "facebook/dinov2-large"]
        )
    with col4:
        if st.button("Extract & Index"):
            images = list(train_dir.glob("*"))
            if not images:
                st.error("🚨 No images—upload a ZIP first.")
            else:
                model_key = model_choice.split("/")[-1]
                emb_path  = project_root / emb_tmpl.replace("${MODEL_NAME}", model_key)
                idx_path  = project_root / index_tmpl.replace("${MODEL_NAME}", model_key)

                logger.info("Extract & Index: model=%s", model_choice)
                emb_path.parent.mkdir(parents=True, exist_ok=True)
                idx_path.parent.mkdir(parents=True, exist_ok=True)
                with st.spinner("Extracting embeddings..."):
                    batch_extract(str(train_dir), str(emb_path), model_name=model_choice)
                with st.spinner("Building FAISS index..."):
                    build_faiss_index(str(emb_path), str(idx_path))

                st.success(f"✔️ Indexed {len(images)} images with **{model_key}**")
                logger.info("Done: %s, %s", emb_path, idx_path)
    st.markdown('</div>', unsafe_allow_html=True)
