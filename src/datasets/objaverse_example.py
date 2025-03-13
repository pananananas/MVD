import os
import torch
import pytorch_lightning as pl
from objaverse_dataset import ObjaverseDataModule
import matplotlib.pyplot as plt

def visualize_sample(sample):
    """Visualize a sample from the dataset."""
    # Get source and target images
    src_img = sample['source']['image'][0].permute(1, 2, 0).numpy()
    tgt_img = sample['target']['image'][0].permute(1, 2, 0).numpy()
    
    # Display the images
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.imshow(src_img)
    ax1.set_title("Source View")
    ax1.axis('off')
    
    ax2.imshow(tgt_img)
    ax2.set_title("Target View")
    ax2.axis('off')
    
    # Add prompt as figure title
    fig.suptitle(f"Prompt: {sample['prompt']}", fontsize=14)
    plt.tight_layout()
    plt.savefig("objaverse_example.png")
    
    # Print camera matrices
    print("Source Camera Matrix:")
    print(sample['source']['camera'][0])
    print("\nTarget Camera Matrix:")
    print(sample['target']['camera'][0])


def main():
    print("Starting Objaverse example...")
    # Path to the Objaverse dataset
    data_root = "/net/pr2/projects/plgrid/plggtattooai/MeshDatasets/objaverse"
    
    # Create data module with sample limit
    data_module = ObjaverseDataModule(
        data_root=data_root,
        batch_size=4,
        num_workers=4,
        target_size=(1024, 1024),
        max_views_per_object=4,
        max_samples=100,
    )
    
    # Setup the data module
    data_module.setup()
    
    # Get a sample batch from the training dataloader
    train_loader = data_module.train_dataloader()
    batch = next(iter(train_loader))
    
    # Visualize the first sample in the batch
    visualize_sample(batch)
    
    # Print some stats
    print(f"Number of training samples: {len(data_module.train_dataset)}")
    print(f"Number of validation samples: {len(data_module.val_dataset)}")
    print(f"Number of test samples: {len(data_module.test_dataset)}")


if __name__ == "__main__":
    main() 