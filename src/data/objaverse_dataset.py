import glob
import io
import json
import logging
import os
import random
import time
import zipfile
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
from icecream import ic
from PIL import Image
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


class ObjaverseDataset(Dataset):
    def __init__(
        self,
        data_root: str,
        split: str = "train",
        split_ratio: Tuple[float, float, float] = (0.8, 0.1, 0.1),
        transform=None,
        target_size: Tuple[int, int] = (256, 256),
        max_views_per_object: int = 4,
        seed: int = 42,
        dataset_samples: Optional[int] = None,
    ):
        super().__init__()
        self.data_root = Path(data_root)
        self.split = split
        self._split_ratio_arg = split_ratio  # Store for load_state_dict
        self.transform = transform
        self.target_size = target_size
        self.max_views_per_object = max_views_per_object
        self.dataset_samples = dataset_samples
        self.seed = seed
        self.rng = random.Random(self.seed)

        render_dir = self.data_root / "renders_final"
        # Fetch all zips first, don't assign to self.zip_files yet
        # This list should remain constant unless underlying file system changes.
        self._initial_all_zip_files_on_disk = sorted(glob.glob(str(render_dir / "*.zip")))
        logger.info(f"Found {len(self._initial_all_zip_files_on_disk)} total object zip files in {render_dir}")

        # _split_dataset will populate self.zip_files and self.zip_files_hash
        # It uses self.rng, which is now initialized.
        self._split_dataset(self._initial_all_zip_files_on_disk, self._split_ratio_arg)

        self.view_pairs = []
        # _build_view_pairs uses self.rng and self.zip_files (set by _split_dataset)
        self._build_view_pairs()
        ic(f"ObjaverseDataset ({split}): Initialization complete. {len(self.view_pairs)=}")

    def _split_dataset(self, all_zips_input_list: list, split_ratio: Tuple[float, float, float]) -> None:
        """Split the dataset into train/val/test based on the provided ratio."""
        ic(f"ObjaverseDataset ({self.split}): Running _split_dataset.")
        assert sum(split_ratio) == 1.0, "Split ratios must sum to 1.0"

        # Work on a copy of the input list for shuffling
        all_zips_shuffled = list(all_zips_input_list) # Make a copy to shuffle
        self.rng.shuffle(all_zips_shuffled) # Shuffle the copy using self.rng

        train_end = int(len(all_zips_shuffled) * split_ratio[0])
        val_end = train_end + int(len(all_zips_shuffled) * split_ratio[1])

        if self.split == "train":
            self.zip_files = all_zips_shuffled[:train_end]
        elif self.split == "val":
            self.zip_files = all_zips_shuffled[train_end:val_end]
        elif self.split == "test":
            self.zip_files = all_zips_shuffled[val_end:]
        else:
            raise ValueError(f"Unknown split: {self.split}")

        # Generate a unique hash for the *split-specific* list of zip files
        # This hash is used for caching view_pairs for this specific split and dataset configuration
        zip_list_str = "".join(sorted(self.zip_files))
        try:
            import hashlib
            self.zip_files_hash = hashlib.md5(zip_list_str.encode()).hexdigest()
        except ImportError:
            self.zip_files_hash = str(hash(zip_list_str))

        logger.info(f"Using {len(self.zip_files)} objects for {self.split} split (hash: {self.zip_files_hash})")
        ic(f"ObjaverseDataset ({self.split}): _split_dataset complete. {len(self.zip_files)=}, hash: {self.zip_files_hash}")

    def _build_view_pairs(self) -> None:
        """Build pairs of source and target views from each object, using a cache if possible."""
        ic(f"ObjaverseDataset ({self.split}): Running _build_view_pairs. {self.dataset_samples=}")

        dataset_samples_suffix = (
            f"_max{self.dataset_samples}"
            if self.dataset_samples is not None
            else "_all"
        )
        cache_filename = f"objaverse_{self.split}_pairs_cache_{self.zip_files_hash}{dataset_samples_suffix}.json"
        cache_path = self.data_root / cache_filename

        if cache_path.exists():
            logger.info(f"Attempting to load view pairs from cache: {cache_path}")
            try:
                with open(cache_path, "r") as f:
                    self.view_pairs = json.load(f)
                logger.info(
                    f"Successfully loaded {len(self.view_pairs)} view pairs from cache."
                )
                return  # Skip building pairs if loaded from cache
            except (json.JSONDecodeError, Exception) as e:
                logger.warning(
                    f"Failed to load or parse cache file {cache_path}: {e}. Rebuilding..."
                )
                # If cache loading fails, proceed to rebuild
        else:
            logger.info(f"Cache file not found at {cache_path}. Building view pairs...")

        # If we reach here, cache wasn't used or failed.
        start_build_time = time.time()

        for zip_path in self.zip_files:
            object_uid = Path(zip_path).stem

            try:
                with zipfile.ZipFile(zip_path, "r") as zip_ref:
                    file_list = zip_ref.namelist()

                    png_files = sorted([f for f in file_list if f.endswith(".png")])
                    npy_files = sorted([f for f in file_list if f.endswith(".npy")])

                    prompt_files = [
                        f for f in file_list if f == f"{object_uid}/prompt.txt"
                    ]
                    has_prompt = len(prompt_files) > 0

                    if len(png_files) != len(npy_files) or len(png_files) == 0:
                        logger.warning(
                            f"Skipping {object_uid}: mismatched files ({len(png_files)} images, {len(npy_files)} camera params)"
                        )
                        continue

                    if not has_prompt:
                        prompt = "3D object"
                    else:
                        with zip_ref.open(prompt_files[0]) as f:
                            prompt = f.read().decode("utf-8").strip()

                    num_views = min(len(png_files), self.max_views_per_object)
                    if num_views < 2:
                        logger.warning(
                            f"Skipping {object_uid}: not enough views ({num_views})"
                        )
                        continue

                    view_indices = self.rng.sample(range(len(png_files)), num_views)

                    for i, src_idx in enumerate(view_indices):
                        for tgt_idx in view_indices[i + 1 :]:  # avoid pairing with self
                            src_png = png_files[src_idx]
                            src_npy = npy_files[src_idx]
                            tgt_png = png_files[tgt_idx]
                            tgt_npy = npy_files[tgt_idx]

                            self.view_pairs.append(
                                {
                                    "zip_path": zip_path,
                                    "object_uid": object_uid,
                                    "prompt": prompt,
                                    "source_image": src_png,
                                    "source_camera": src_npy,
                                    "target_image": tgt_png,
                                    "target_camera": tgt_npy,
                                }
                            )

                    if (
                        self.dataset_samples is not None
                        and len(self.view_pairs) >= self.dataset_samples
                    ):
                        break

            except (zipfile.BadZipFile, KeyError, Exception) as e:
                logger.error(f"Error processing {zip_path}: {str(e)}")
                continue

            if (
                self.dataset_samples is not None
                and len(self.view_pairs) >= self.dataset_samples
            ):
                break

        try:
            with open(cache_path, "w") as f:
                json.dump(self.view_pairs, f)  # Save the potentially limited list
        except Exception as e:
            logger.error(f"Failed to save cache file {cache_path}: {e}")

        end_build_time = time.time()
        logger.info(
            f"Created {len(self.view_pairs)} view pairs for {self.split} split from zips."
        )
        ic(
            f"Total time taken to build view pairs from zips: {end_build_time - start_build_time:.2f} seconds"
        )

    def __len__(self) -> int:
        """Return the number of view pairs in the dataset."""
        return len(self.view_pairs)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single source-target pair by index."""
        max_retries = 3
        last_exception = None

        for retry in range(max_retries):
            try:
                pair_info = self.view_pairs[idx]
                zip_path = pair_info["zip_path"]

                if not os.path.exists(zip_path):
                    self.view_pairs.pop(idx)
                    idx = (idx + 1) % len(self.view_pairs)
                    continue

                object_uid = pair_info["object_uid"]
                source_image = pair_info["source_image"]
                source_camera = pair_info["source_camera"]
                target_image = pair_info["target_image"]
                target_camera = pair_info["target_camera"]

                with zipfile.ZipFile(zip_path, "r") as zip_ref:
                    source_img = self._load_image(zip_ref, source_image)
                    source_cam = self._load_camera(zip_ref, source_camera)

                    target_img = self._load_image(zip_ref, target_image)
                    target_cam = self._load_camera(zip_ref, target_camera)

                break

            except (FileNotFoundError, zipfile.BadZipFile, KeyError) as e:
                last_exception = e
                self.view_pairs.pop(idx)
                idx = (idx + 1) % len(self.view_pairs)
                continue
        else:
            raise RuntimeError(
                f"Failed to load valid sample after {max_retries} attempts. Last error: {last_exception}"
            )

        if self.transform:
            source_img = self.transform(source_img)
            target_img = self.transform(target_img)

        source_img_tensor = (
            torch.from_numpy(np.array(source_img)).permute(2, 0, 1).float() / 127.5
            - 1.0
        )
        target_img_tensor = (
            torch.from_numpy(np.array(target_img)).permute(2, 0, 1).float() / 127.5
            - 1.0
        )
        source_cam_tensor = torch.from_numpy(source_cam).float()
        target_cam_tensor = torch.from_numpy(target_cam).float()

        return {
            "object_uid": object_uid,
            "prompt": pair_info["prompt"],
            "source_image": source_img_tensor,
            "target_image": target_img_tensor,
            "source_camera": source_cam_tensor,
            "target_camera": target_cam_tensor,
        }

    def _load_image(self, zip_ref: zipfile.ZipFile, image_path: str) -> Image.Image:
        """Load and resize an image from the zip file."""
        with zip_ref.open(image_path) as f:
            img_data = f.read()
            img = Image.open(io.BytesIO(img_data))

            if img.mode == "RGBA":
                white_bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
                img = Image.alpha_composite(white_bg, img)

            img = img.convert("RGB")

            if img.size != self.target_size:
                img = img.resize(self.target_size, Image.Resampling.LANCZOS)

            return img

    def _load_camera(self, zip_ref: zipfile.ZipFile, camera_path: str) -> np.ndarray:
        """Load camera parameters from the zip file."""
        with zip_ref.open(camera_path) as f:
            camera_data = np.load(io.BytesIO(f.read()))
            return camera_data

    def state_dict(self):
        ic(f"ObjaverseDataset ({self.split}): Saving state...")
        state = {"rng_state": self.rng.getstate()}
        ic(f"ObjaverseDataset ({self.split}): RNG state saved.")
        return state

    def load_state_dict(self, state_dict):
        ic(f"ObjaverseDataset ({self.split}): Loading state...")
        self.rng.setstate(state_dict["rng_state"])
        ic(f"ObjaverseDataset ({self.split}): RNG state restored.")

        # Re-run dataset construction logic that depends on RNG
        ic(f"ObjaverseDataset ({self.split}): Re-initializing split and view pairs post RNG state load.")
        
        # It's crucial that _initial_all_zip_files_on_disk reflects the true state of discoverable files
        # If files changed on disk, this list would be different. For typical SLURM restarts, it should be the same.
        # Re-fetch it to be safe, though it was stored in __init__
        render_dir = self.data_root / "renders_final"
        current_all_zip_files_on_disk = sorted(glob.glob(str(render_dir / "*.zip")))
        if len(current_all_zip_files_on_disk) != len(self._initial_all_zip_files_on_disk):
            logger.warning(f"ObjaverseDataset ({self.split}): Number of zip files on disk changed between initial setup and load_state_dict. "
                           f"Initial: {len(self._initial_all_zip_files_on_disk)}, Current: {len(current_all_zip_files_on_disk)}")
            # Potentially update self._initial_all_zip_files_on_disk if this is desired behavior,
            # but for strict resumability, using the original list might be preferred if it's guaranteed to exist.
            # For now, use the current state from disk for re-splitting.
            self._initial_all_zip_files_on_disk = current_all_zip_files_on_disk


        # Re-apply the split logic using the restored RNG and the (potentially updated) full list of zips
        self._split_dataset(self._initial_all_zip_files_on_disk, self._split_ratio_arg)
        
        # Re-build view pairs, which also uses self.rng and the now correctly set self.zip_files
        self.view_pairs = [] # Clear any old/partially built pairs
        self._build_view_pairs() 
        
        ic(f"ObjaverseDataset ({self.split}): Re-initialization complete after loading state. {len(self.view_pairs)=}")


class ObjaverseDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_root: str,
        batch_size: int = 32,
        num_workers: int = 4,
        split_ratio: Tuple[float, float, float] = (0.8, 0.1, 0.1),
        target_size: Tuple[int, int] = (256, 256),
        max_views_per_object: int = 4,
        seed: int = 42,
        dataset_samples: Optional[int] = None,
    ):
        super().__init__()
        self.data_root = data_root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.split_ratio = split_ratio
        self.target_size = target_size
        self.max_views_per_object = max_views_per_object
        self.seed = seed
        self.dataset_samples = dataset_samples

        self.transform = None
        ic(f"ObjaverseDataModule: Initialized with data_root={data_root}, batch_size={batch_size}, dataset_samples={dataset_samples}")

    def setup(self, stage: Optional[str] = None):
        ic(f"ObjaverseDataModule: setup(stage='{stage}') called.")
        if stage == "fit" or stage is None:
            self.train_dataset = ObjaverseDataset(
                data_root=self.data_root,
                split="train",
                split_ratio=self.split_ratio,
                transform=self.transform,
                target_size=self.target_size,
                max_views_per_object=self.max_views_per_object,
                seed=self.seed,
                dataset_samples=self.dataset_samples,
            )

            self.val_dataset = ObjaverseDataset(
                data_root=self.data_root,
                split="val",
                split_ratio=self.split_ratio,
                transform=self.transform,
                target_size=self.target_size,
                max_views_per_object=self.max_views_per_object,
                seed=self.seed,
                dataset_samples=self.dataset_samples,
            )

        if stage == "test" or stage is None:
            self.test_dataset = ObjaverseDataset(
                data_root=self.data_root,
                split="test",
                split_ratio=self.split_ratio,
                transform=self.transform,
                target_size=self.target_size,
                max_views_per_object=self.max_views_per_object,
                seed=self.seed,
                dataset_samples=self.dataset_samples,
            )
        ic(f"ObjaverseDataModule: setup complete. Train: {hasattr(self, 'train_dataset')}, Val: {hasattr(self, 'val_dataset')}, Test: {hasattr(self, 'test_dataset')}")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
        )

    def state_dict(self):
        ic("ObjaverseDataModule: Saving state...")
        state = {}
        if hasattr(self, 'train_dataset') and self.train_dataset is not None:
            state['train_dataset_state'] = self.train_dataset.state_dict()
            ic("ObjaverseDataModule: train_dataset state included.")
        if hasattr(self, 'val_dataset') and self.val_dataset is not None:
            state['val_dataset_state'] = self.val_dataset.state_dict()
            ic("ObjaverseDataModule: val_dataset state included.")
        if hasattr(self, 'test_dataset') and self.test_dataset is not None:
            state['test_dataset_state'] = self.test_dataset.state_dict()
            ic("ObjaverseDataModule: test_dataset state included.")
        
        if not state:
            ic("ObjaverseDataModule: No dataset states to save (datasets might not be setup).")
        else:
            ic(f"ObjaverseDataModule: State saved with keys: {list(state.keys())}")
        return state

    def load_state_dict(self, state_dict):
        ic("ObjaverseDataModule: Loading state...")
        loaded_any = False
        if hasattr(self, 'train_dataset') and self.train_dataset is not None and 'train_dataset_state' in state_dict:
            ic("ObjaverseDataModule: Loading state for train_dataset.")
            self.train_dataset.load_state_dict(state_dict['train_dataset_state'])
            loaded_any = True
        elif 'train_dataset_state' in state_dict:
            ic("ObjaverseDataModule: train_dataset_state found in state_dict, but train_dataset not initialized in module.")
        
        if hasattr(self, 'val_dataset') and self.val_dataset is not None and 'val_dataset_state' in state_dict:
            ic("ObjaverseDataModule: Loading state for val_dataset.")
            self.val_dataset.load_state_dict(state_dict['val_dataset_state'])
            loaded_any = True
        elif 'val_dataset_state' in state_dict:
            ic("ObjaverseDataModule: val_dataset_state found in state_dict, but val_dataset not initialized in module.")

        if hasattr(self, 'test_dataset') and self.test_dataset is not None and 'test_dataset_state' in state_dict:
            ic("ObjaverseDataModule: Loading state for test_dataset.")
            self.test_dataset.load_state_dict(state_dict['test_dataset_state'])
            loaded_any = True
        elif 'test_dataset_state' in state_dict:
            ic("ObjaverseDataModule: test_dataset_state found in state_dict, but test_dataset not initialized in module.")
            
        if not loaded_any:
            ic("ObjaverseDataModule: No dataset states were loaded (or datasets not setup).")
        ic("ObjaverseDataModule: State loading process complete.")


def visualize_sample(sample):
    src_img = sample["source_image"][0].permute(1, 2, 0).numpy()
    tgt_img = sample["target_image"][0].permute(1, 2, 0).numpy()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.imshow(src_img)
    ax1.set_title("Source View")
    ax1.axis("off")

    ax2.imshow(tgt_img)
    ax2.set_title("Target View")
    ax2.axis("off")

    fig.suptitle(f"Prompt: {sample['prompt'][0]}", fontsize=14)
    plt.tight_layout()
    plt.savefig("objaverse_example.png")

    print("Source Camera Matrix:")
    print(sample["source_camera"][0])
    print("\nTarget Camera Matrix:")
    print(sample["target_camera"][0])


def main():
    print("Starting Objaverse example...")

    # data_root = "/net/pr2/projects/plgrid/plggtattooai/MeshDatasets/objaverse"
    data_root = "/Users/ewojcik/Code/pwr/MVD"
    start_time = time.time()

    data_module = ObjaverseDataModule(
        data_root=data_root,
        batch_size=64,
        num_workers=4,
        target_size=(1024, 1024),
        max_views_per_object=4,
        dataset_samples=1,
    )
    data_module.setup()

    train_loader = data_module.train_dataloader()
    batch = next(iter(train_loader))
    end_time = time.time()
    print(f"Time taken to create data module: {end_time - start_time} seconds")

    print(train_loader)
    visualize_sample(batch)

    print(f"Number of training samples: {len(data_module.train_dataset)}")
    print(f"Number of validation samples: {len(data_module.val_dataset)}")
    print(f"Number of test samples: {len(data_module.test_dataset)}")


if __name__ == "__main__":
    main()
