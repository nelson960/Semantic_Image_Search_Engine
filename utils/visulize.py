import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np

# Load a smaller subset of embeddings
embeddings = np.load("data/embeddings.npy")[:200]  # Only first 200

# Simulate labels (e.g., 10 classes × 20 images each)
labels = np.repeat(np.arange(10), 20)

# Run t-SNE
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
embeddings_2d = tsne.fit_transform(embeddings)

# Plot
plt.figure(figsize=(10, 8))
scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=labels, cmap='tab10', alpha=0.7)
plt.title("t-SNE of DINOv2 Embeddings (200 CIFAR-10 Images)")
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.colorbar(scatter, ticks=range(10), label="Simulated Class")
plt.grid(True)
plt.tight_layout()
plt.show()