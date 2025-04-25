# utils/evaluate_model.py
import torch
import faiss
import numpy as np
import pandas as pd
import random
import json
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from pathlib import Path

def load_faiss_index(index_path):
    return faiss.read_index(index_path)

def search_top_k(index, query_vec, k=5):
    query_vec = np.expand_dims(query_vec, axis=0).astype('float32')
    distances, indices = index.search(query_vec, k)
    return distances[0], indices[0]

def precision_at_k(retrieved_clusters, query_cluster, k):
    correct = sum(1 for cluster in retrieved_clusters[:k] if cluster == query_cluster)
    return correct / k

def recall_at_k(retrieved_clusters, query_cluster, all_clusters, k):
    total_relevant = sum(1 for c in all_clusters if c == query_cluster)
    retrieved_relevant = sum(1 for cluster in retrieved_clusters[:k] if cluster == query_cluster)
    return retrieved_relevant / total_relevant if total_relevant else 0

def plot_distance_histogram(distances, correct_flags, save_path):
    plt.figure(figsize=(8,6))
    correct = [d for d, correct in zip(distances, correct_flags) if correct]
    incorrect = [d for d, correct in zip(distances, correct_flags) if not correct]
    plt.hist(correct, bins=30, alpha=0.6, label='Correct')
    plt.hist(incorrect, bins=30, alpha=0.6, label='Incorrect')
    plt.xlabel("Distance")
    plt.ylabel("Frequency")
    plt.title("Distance Distribution for Top-K Results")
    plt.legend()
    plt.savefig(save_path)
    plt.close()

def plot_tsne(embeddings, cluster_ids, save_path):
    tsne = TSNE(n_components=2, random_state=42)
    reduced = tsne.fit_transform(embeddings)
    plt.figure(figsize=(10,8))
    scatter = plt.scatter(reduced[:,0], reduced[:,1], c=cluster_ids, cmap='tab10', alpha=0.6)
    plt.colorbar(scatter)
    plt.title("t-SNE Projection of Embeddings Colored by Cluster")
    plt.savefig(save_path)
    plt.close()

def evaluate_model(embedding_path, index_path, label_path, k=5, num_queries=100, output_dir="reports"):
    # Load everything
    embeddings = np.load(embedding_path)
    index = load_faiss_index(index_path)
    labels_df = pd.read_csv(label_path)
    labels_dict = dict(zip(labels_df['filename'], labels_df['cluster_id']))

    # Assume filenames are sorted same as embeddings
    image_dir = Path("data/raw/train")
    filenames = sorted([p.name for p in image_dir.glob("*.png")])

    queries = random.sample(list(zip(filenames, embeddings)), num_queries)

    precisions = []
    recalls = []
    all_distances = []
    all_correct_flags = []

    for filename, query_vec in queries:
        query_cluster = labels_dict[filename]
        dists, idxs = search_top_k(index, query_vec, k)
        retrieved_filenames = [filenames[i] for i in idxs]
        retrieved_clusters = [labels_dict[f] for f in retrieved_filenames]

        # Precision and Recall
        p = precision_at_k(retrieved_clusters, query_cluster, k)
        r = recall_at_k(retrieved_clusters, query_cluster, list(labels_dict.values()), k)

        precisions.append(p)
        recalls.append(r)

        # For distance plotting
        correct_flags = [1 if c == query_cluster else 0 for c in retrieved_clusters]
        all_distances.extend(dists.tolist())
        all_correct_flags.extend(correct_flags)

    avg_precision = np.mean(precisions)
    avg_recall = np.mean(recalls)

    # Save metrics
    Path(output_dir).mkdir(exist_ok=True, parents=True)
    report_path = Path(output_dir) / f"{Path(embedding_path).stem}_results.json"
    with open(report_path, "w") as f:
        json.dump({
            f"Precision@{k}": avg_precision,
            f"Recall@{k}": avg_recall,
            "Num Queries": num_queries
        }, f, indent=4)

    print(f"Saved evaluation report to {report_path}")

    # Plot and save figures
    plot_distance_histogram(all_distances, all_correct_flags, save_path=Path(output_dir)/f"{Path(embedding_path).stem}_distance_hist.png")
    plot_tsne(embeddings, list(labels_dict.values()), save_path=Path(output_dir)/f"{Path(embedding_path).stem}_tsne.png")



evaluate_model()