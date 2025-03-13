from torch.utils.data import Dataset, DataLoader
from typing import Dict, Tuple, Optional, Any
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from pathlib import Path
from PIL import Image
import numpy as np
import logging
import zipfile
import random
import torch
import glob
import time
import io


logger = logging.getLogger(__name__)


class ObjaverseDataset(Dataset):
    """
    Dataset for loading multi-view renderings from Objaverse dataset.
    
    Each item consists of a source and target view pair, where both have
    an image, camera matrix, and the object's text prompt.
    """
    
    def __init__(
        self,
        data_root: str,
        split: str = "train",
        split_ratio: Tuple[float, float, float] = (0.8, 0.1, 0.1),
        transform=None,
        target_size: Tuple[int, int] = (256, 256),
        max_views_per_object: int = 4,
        seed: int = 42,
        max_samples: Optional[int] = None,
    ):
        """
        Initialize the Objaverse dataset.
        
        Args:
            data_root: Path to the directory containing Objaverse zip files
            split: One of "train", "val", or "test"
            split_ratio: Ratio for train/val/test splits
            transform: Optional transforms to apply to images
            target_size: Target image size (height, width)
            max_views_per_object: Maximum number of views to use per object
            seed: Random seed for reproducibility
            max_samples: Maximum number of samples to use (if None, use all available)
        """
        super().__init__()
        self.data_root = Path(data_root)
        self.split = split
        self.transform = transform
        self.target_size = target_size
        self.max_views_per_object = max_views_per_object
        self.max_samples = max_samples
        self.rng = random.Random(seed)
        
        # Find all zip files in the renders directory
        render_dir = self.data_root / "renders"
        self.zip_files = sorted(glob.glob(str(render_dir / "*.zip")))
        logger.info(f"Found {len(self.zip_files)} object zip files in {render_dir}")
        
        # Split the dataset
        self._split_dataset(split_ratio)
        
        # Prepare data structure that will hold information about view pairs
        self.view_pairs = []
        
        # Index the dataset to build view pairs
        self._build_view_pairs()
        
        # Limit the number of samples if requested
        if self.max_samples is not None and len(self.view_pairs) > self.max_samples:
            logger.info(f"Limiting dataset to {self.max_samples} samples (out of {len(self.view_pairs)} available)")
            # Use random sampling to get a subset
            self.view_pairs = self.rng.sample(self.view_pairs, self.max_samples)
    
    def _split_dataset(self, split_ratio: Tuple[float, float, float]) -> None:
        """Split the dataset into train/val/test based on the provided ratio."""
        assert sum(split_ratio) == 1.0, "Split ratios must sum to 1.0"
        
        # Shuffle zip files with fixed seed for reproducibility
        all_zips = self.zip_files.copy()
        self.rng.shuffle(all_zips)
        
        # Calculate split indices
        train_end = int(len(all_zips) * split_ratio[0])
        val_end = train_end + int(len(all_zips) * split_ratio[1])
        
        # Assign zips based on requested split
        if self.split == "train":
            self.zip_files = all_zips[:train_end]
        elif self.split == "val":
            self.zip_files = all_zips[train_end:val_end]
        elif self.split == "test":
            self.zip_files = all_zips[val_end:]
        else:
            raise ValueError(f"Unknown split: {self.split}")
        
        logger.info(f"Using {len(self.zip_files)} objects for {self.split} split")
    
    def _build_view_pairs(self) -> None:
        """Build pairs of source and target views from each object."""
        for zip_path in self.zip_files:
            object_uid = Path(zip_path).stem
            
            try:
                # Check zip file contents without extracting everything
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    file_list = zip_ref.namelist()
                    
                    # Get all PNG and NPY files
                    png_files = sorted([f for f in file_list if f.endswith('.png')])
                    npy_files = sorted([f for f in file_list if f.endswith('.npy')])
                    
                    # Check if prompt file exists
                    prompt_files = [f for f in file_list if f == f"{object_uid}/prompt.txt"]
                    has_prompt = len(prompt_files) > 0
                    
                    # Ensure we have matching images and camera parameters
                    if len(png_files) != len(npy_files) or len(png_files) == 0:
                        logger.warning(f"Skipping {object_uid}: mismatched files ({len(png_files)} images, {len(npy_files)} camera params)")
                        continue
                    
                    if not has_prompt:
                        # Create a placeholder prompt if missing
                        prompt = "3D object"
                    else:
                        # Read prompt from zip file
                        with zip_ref.open(prompt_files[0]) as f:
                            prompt = f.read().decode('utf-8').strip()
                    
                    # Determine the number of views to use (at least 2 for source/target)
                    num_views = min(len(png_files), self.max_views_per_object)
                    if num_views < 2:
                        logger.warning(f"Skipping {object_uid}: not enough views ({num_views})")
                        continue
                    
                    # Select random indices for these views
                    view_indices = self.rng.sample(range(len(png_files)), num_views)
                    
                    # Create all possible pairs of views (cartesian product)
                    for i, src_idx in enumerate(view_indices):
                        for tgt_idx in view_indices[i+1:]:  # Avoid pairing with self
                            src_png = png_files[src_idx]
                            src_npy = npy_files[src_idx]
                            tgt_png = png_files[tgt_idx]
                            tgt_npy = npy_files[tgt_idx]
                            
                            self.view_pairs.append({
                                'zip_path': zip_path,
                                'object_uid': object_uid,
                                'prompt': prompt,
                                'source': {'image': src_png, 'camera': src_npy},
                                'target': {'image': tgt_png, 'camera': tgt_npy}
                            })
            
            except (zipfile.BadZipFile, KeyError, Exception) as e:
                logger.error(f"Error processing {zip_path}: {str(e)}")
                continue
        
        logger.info(f"Created {len(self.view_pairs)} view pairs for {self.split} split")
    
    def __len__(self) -> int:
        """Return the number of view pairs in the dataset."""
        return len(self.view_pairs)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single source-target pair by index."""
        pair_info = self.view_pairs[idx]
        zip_path = pair_info['zip_path']
        object_uid = pair_info['object_uid']
        
        # Get the source and target info
        source_info = pair_info['source']
        target_info = pair_info['target']
        
        # Load the data from the zip file
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Load source data
            source_img = self._load_image(zip_ref, source_info['image'])
            source_cam = self._load_camera(zip_ref, source_info['camera'])
            
            # Load target data
            target_img = self._load_image(zip_ref, target_info['image'])
            target_cam = self._load_camera(zip_ref, target_info['camera'])
            
        # Apply transformations if specified
        if self.transform:
            source_img = self.transform(source_img)
            target_img = self.transform(target_img)
        
        # Convert to tensors
        source_img_tensor = torch.from_numpy(np.array(source_img)).permute(2, 0, 1).float() / 255.0
        target_img_tensor = torch.from_numpy(np.array(target_img)).permute(2, 0, 1).float() / 255.0
        source_cam_tensor = torch.from_numpy(source_cam).float()
        target_cam_tensor = torch.from_numpy(target_cam).float()
        
        return {
            'object_uid': object_uid,
            'prompt': pair_info['prompt'],
            'source': {
                'image': source_img_tensor,
                'camera': source_cam_tensor,
            },
            'target': {
                'image': target_img_tensor,
                'camera': target_cam_tensor,
            },
        }
    
    def _load_image(self, zip_ref: zipfile.ZipFile, image_path: str) -> Image.Image:
        """Load and resize an image from the zip file."""
        with zip_ref.open(image_path) as f:
            img_data = f.read()
            img = Image.open(io.BytesIO(img_data))
            
            # White background for transparent images
            if img.mode == 'RGBA':
                white_bg = Image.new('RGBA', img.size, (255, 255, 255, 255))
                img = Image.alpha_composite(white_bg, img)
            
            # Convert to RGB
            img = img.convert('RGB')
            
            # Resize if needed
            if img.size != self.target_size:
                img = img.resize(self.target_size, Image.Resampling.LANCZOS)
            
            return img
    
    def _load_camera(self, zip_ref: zipfile.ZipFile, camera_path: str) -> np.ndarray:
        """Load camera parameters from the zip file."""
        with zip_ref.open(camera_path) as f:
            # Load the RT matrix (4x4 rotation-translation matrix)
            camera_data = np.load(io.BytesIO(f.read()))
            return camera_data


class ObjaverseDataModule(pl.LightningDataModule):
    """
    Lightning Data Module for Objaverse dataset.
    
    Handles train/val/test splits and creates appropriate dataloaders.
    """
    
    def __init__(
        self,
        data_root: str,
        batch_size: int = 32,
        num_workers: int = 4,
        split_ratio: Tuple[float, float, float] = (0.8, 0.1, 0.1),
        target_size: Tuple[int, int] = (256, 256),
        max_views_per_object: int = 4,
        seed: int = 42,
        max_samples: Optional[int] = None,
    ):
        """
        Initialize the Objaverse Data Module.
        
        Args:
            data_root: Path to the directory containing Objaverse zip files
            batch_size: Batch size for training and evaluation
            num_workers: Number of workers for data loading
            split_ratio: Ratio for train/val/test splits
            target_size: Target image size (height, width)
            max_views_per_object: Maximum number of views to use per object
            seed: Random seed for reproducibility
            max_samples: Maximum number of samples to use per split (if None, use all available)
        """
        super().__init__()
        self.data_root = data_root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.split_ratio = split_ratio
        self.target_size = target_size
        self.max_views_per_object = max_views_per_object
        self.seed = seed
        self.max_samples = max_samples
        
        self.transform = None  # Define transforms if needed
    
    def setup(self, stage: Optional[str] = None):
        """
        Setup train, validation, and test datasets.
        
        Args:
            stage: Current stage (fit, validate, test, predict)
        """
        if stage == 'fit' or stage is None:
            self.train_dataset = ObjaverseDataset(
                data_root=self.data_root,
                split="train",
                split_ratio=self.split_ratio,
                transform=self.transform,
                target_size=self.target_size,
                max_views_per_object=self.max_views_per_object,
                seed=self.seed,
                max_samples=self.max_samples,
            )
            
            self.val_dataset = ObjaverseDataset(
                data_root=self.data_root,
                split="val",
                split_ratio=self.split_ratio,
                transform=self.transform,
                target_size=self.target_size,
                max_views_per_object=self.max_views_per_object,
                seed=self.seed,
                max_samples=self.max_samples,
            )
        
        if stage == 'test' or stage is None:
            self.test_dataset = ObjaverseDataset(
                data_root=self.data_root,
                split="test",
                split_ratio=self.split_ratio,
                transform=self.transform,
                target_size=self.target_size,
                max_views_per_object=self.max_views_per_object,
                seed=self.seed,
                max_samples=self.max_samples,
            )
    
    def train_dataloader(self):
        """Create the training dataloader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )
    
    def val_dataloader(self):
        """Create the validation dataloader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
    
    def test_dataloader(self):
        """Create the test dataloader."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        ) 

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
    fig.suptitle(f"Prompt: {sample['prompt'][0]}", fontsize=14)
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
    # data_root = "/net/pr2/projects/plgrid/plggtattooai/MeshDatasets/objaverse"
    data_root = "/Users/ewojcik/Code/pwr/MVD"
    start_time = time.time()
    # Create data module with sample limit
    data_module = ObjaverseDataModule(
        data_root=data_root,
        batch_size=64,
        num_workers=4,
        target_size=(1024, 1024),
        max_views_per_object=4,
        max_samples=1,
    )
    # Setup the data module
    data_module.setup()
    
    # Get a sample batch from the training dataloader
    train_loader = data_module.train_dataloader()
    batch = next(iter(train_loader))
    end_time = time.time()
    print(f"Time taken to create data module: {end_time - start_time} seconds")
    
    print(train_loader)
    # Visualize the first sample in the batch
    visualize_sample(batch)
    
    # Print some stats
    print(f"Number of training samples: {len(data_module.train_dataset)}")
    print(f"Number of validation samples: {len(data_module.val_dataset)}")
    print(f"Number of test samples: {len(data_module.test_dataset)}")


if __name__ == "__main__":
    main() 