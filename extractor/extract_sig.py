
import pathlib, concurrent.futures, cv2, warnings, time
import numpy as np, torch, faiss
from tqdm import tqdm
from transformers import SiglipProcessor, SiglipModel, AutoConfig

def encode_images_to_memmap(
    image_dir: str,
    ckpt: str,
    batch_size: int,
    num_threads: int,
    vec_npy_path: str,
    paths_npy_path: str,
    target_res: int = 256,    # pre‑resize long side
):
    # ── device & dtype ──────────────────────────────────────────────
    if torch.backends.mps.is_available():
        device, dtype = "mps", torch.float16
    elif torch.cuda.is_available():
        device, dtype = "cuda", torch.float16
    else:
        device, dtype = "cpu",  torch.float32
    print(f"▶ device={device}  dtype={dtype}")

    # ── model & processor ──────────────────────────────────────────
    processor = SiglipProcessor.from_pretrained(ckpt, use_fast=False)
    model = SiglipModel.from_pretrained(ckpt, torch_dtype=dtype).to(device).eval()

    # ── gather images ──────────────────────────────────────────────
    paths = sorted(
        p for p in pathlib.Path(image_dir).rglob("*")
        if p.suffix.lower() in {".jpg",".jpeg",".png"})
    if not paths: raise RuntimeError("no images")

    emb_dim = model.config.text_config.hidden_size
    vecs = np.memmap(vec_npy_path, dtype="float32", mode="w+", shape=(len(paths), emb_dim))
    np.save(paths_npy_path, np.asarray([str(p) for p in paths], dtype="object"))

    # ── threaded JPEG loader with resize ───────────────────────────
    def load(path):
        im = cv2.imread(str(path))
        if im is None: return None
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        h, w = im.shape[:2]
        scale = target_res / max(h, w)
        if scale < 1.0:
            im = cv2.resize(im, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
        return im

    pool = concurrent.futures.ThreadPoolExecutor(max_workers=num_threads)

    cur_bs = batch_size
    i = 0
    pbar = tqdm(total=len(paths), desc="encode", ncols=80)
    while i < len(paths):
        batch_paths = paths[i:i+cur_bs]
        imgs = list(pool.map(load, batch_paths))
        valid = [(p, im) for p, im in zip(batch_paths, imgs) if im is not None]
        if not valid:
            i += cur_bs
            pbar.update(cur_bs)
            continue
        _, imgs = zip(*valid)

        inputs = processor(images=list(imgs), return_tensors="pt", padding=True)
        inputs = {k: v.to(device, dtype=dtype) for k, v in inputs.items()}

        try:
            with torch.no_grad():
                feat = model.get_image_features(**inputs)
                feat = feat / feat.norm(dim=-1, keepdim=True)
            vecs[i:i+len(feat)] = feat.cpu().numpy().astype("float32")
            i += cur_bs
            pbar.update(cur_bs)
            torch.mps.empty_cache() if device=="mps" else None
        except RuntimeError as e:
            if "out of memory" in str(e) and cur_bs > 1:
                torch.mps.empty_cache() if device=="mps" else torch.cuda.empty_cache()
                cur_bs //= 2
                warnings.warn(f"OOM – reducing batch to {cur_bs}")
                time.sleep(0.5)
            else:
                raise e
    pbar.close()
    vecs.flush()
    print("✓ finished, stored", len(paths), "vectors")

# Example
if __name__ == "__main__":
    encode_images_to_memmap(
        image_dir="data/raw/train",
        ckpt="google/siglip-so400m-patch14-384",
        batch_size=32,             # will auto‑shrink on OOM
        num_threads=4,
        vec_npy_path="tmp/siglip_vectors.npy",
        paths_npy_path="tmp/siglip_paths.npy",
    )
