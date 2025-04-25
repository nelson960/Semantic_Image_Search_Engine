# utils/generate_pseudo_labels.py
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from pathlib import Path
import argparse

def generate_pseudo_labels(embedding_path: str, output_csv_path: str, num_clusters: int = 10):
    """
    Cluster embeddings into pseudo-labels using KMeans.

    Args:
        embedding_path (str): Path to the .npy embedding file.
        output_csv_path (str): Path where pseudo-label CSV will be saved.
        num_clusters (int): Number of clusters to form (default=10).
    """
    # Load embeddings
    print(f"Loading embeddings from {embedding_path}")
    embeddings = np.load(embedding_path)

    # Cluster
    print(f"Clustering into {num_clusters} clusters with KMeans...")
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    cluster_ids = kmeans.fit_predict(embeddings)

    # Assume filenames are in the same order
    image_dir = Path("data/raw/train")  # ⚡ adjust if different
    filenames = sorted([p.name for p in image_dir.glob("*.png")])

    if len(filenames) != len(cluster_ids):
        raise ValueError(f"Mismatch: {len(filenames)} images vs {len(cluster_ids)} embeddings")

    # Save to CSV
    df = pd.DataFrame({
        "filename": filenames,
        "cluster_id": cluster_ids
    })
    df.to_csv(output_csv_path, index=False)
    print(f"Saved pseudo-labels to {output_csv_path}")

    
generate_pseudo_labels()
