import torchvision.datasets as datasets
from pathlib import Path

def save_cifar10_dataset(target_dir='data/raw', split='train', num_images=1000):
    # Determine whether we want the train or test split
    is_train = split.lower() == 'train'
    # Download / load the dataset (returns PIL.Image objects by default)
    dataset = datasets.CIFAR10(root='./data', train=is_train, download=True)

    # Create an output subfolder per split, e.g. data/raw/train or data/raw/test
    output_dir = Path(target_dir) / split
    output_dir.mkdir(parents=True, exist_ok=True)

    for i, (img, label) in enumerate(dataset):
        if i >= num_images:
            break
        # img is already a PIL.Image, so just save it
        img.save(output_dir / f"{split}_{i:04d}.png")

if __name__ == "__main__":
    save_cifar10_dataset()
