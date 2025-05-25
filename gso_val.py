import argparse
import csv
import logging
import os
import time
from pathlib import Path, PosixPath
from typing import Dict, List

import lpips
import numpy as np
import torch
import yaml
# from icecream import ic
from PIL import Image
from pytorch_msssim import SSIM
from torch.utils.data import DataLoader
from torchmetrics.image import FrechetInceptionDistance, PeakSignalNoiseRatio
from torchmetrics.multimodal import CLIPScore
from tqdm import tqdm

torch.serialization.add_safe_globals([PosixPath])

from src.data.objaverse_dataset import ObjaverseDataset
from src.models.mvd_unet import create_mvd_pipeline
from src.training.losses import PerceptualLoss

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def validate_tensor_ranges(
    tensor: torch.Tensor, name: str, expected_range: str = "[-1, 1]"
):
    """Validate tensor value ranges and log statistics."""
    min_val = tensor.min().item()
    max_val = tensor.max().item()
    mean_val = tensor.mean().item()
    std_val = tensor.std().item()

    logger.info(
        f"{name} stats: min={min_val:.4f}, max={max_val:.4f}, mean={mean_val:.4f}, std={std_val:.4f}"
    )

    if expected_range == "[-1, 1]":
        if min_val < -1.1 or max_val > 1.1:
            logger.warning(
                f"⚠ {name} values outside expected range [-1, 1]: [{min_val:.4f}, {max_val:.4f}]"
            )
    elif expected_range == "[0, 1]":
        if min_val < -0.1 or max_val > 1.1:
            logger.warning(
                f"⚠ {name} values outside expected range [0, 1]: [{min_val:.4f}, {max_val:.4f}]"
            )

    # Check for NaN/Inf
    if torch.isnan(tensor).any():
        logger.error(f"✗ {name} contains NaN values!")
        return False
    if torch.isinf(tensor).any():
        logger.error(f"✗ {name} contains Inf values!")
        return False

    return True


class ValidationMetrics:
    """Class to handle all validation metrics calculation."""

    def __init__(
        self, device: str, clip_model_name: str = "openai/clip-vit-large-patch14"
    ):
        self.device = device
        logger.info(f"Initializing ValidationMetrics on device: {device}")

        # Initialize metrics with error handling
        try:
            self.ssim = SSIM(data_range=2.0, size_average=True).to(device)
            logger.info("✓ SSIM metric initialized")
        except Exception as e:
            logger.error(f"✗ Failed to initialize SSIM: {e}")
            self.ssim = None

        try:
            self.psnr = PeakSignalNoiseRatio(data_range=2.0).to(device)
            logger.info("✓ PSNR metric initialized")
        except Exception as e:
            logger.error(f"✗ Failed to initialize PSNR: {e}")
            self.psnr = None

        try:
            self.perceptual_loss = PerceptualLoss(device=device)
            logger.info("✓ Perceptual loss initialized")
        except Exception as e:
            logger.error(f"✗ Failed to initialize Perceptual loss: {e}")
            self.perceptual_loss = None

        # Initialize LPIPS
        try:
            self.lpips_metric = lpips.LPIPS(net="alex").to(device)
            logger.info("✓ LPIPS metric initialized")
        except Exception as e:
            logger.error(f"✗ Failed to initialize LPIPS: {e}")
            self.lpips_metric = None

        # Initialize CLIP Score
        try:
            self.clip_score_metric = CLIPScore(model_name_or_path=clip_model_name).to(
                device
            )
            logger.info("✓ CLIP Score metric initialized")
        except Exception as e:
            logger.warning(f"⚠ Failed to initialize CLIPScore: {e}")
            self.clip_score_metric = None

        # Initialize FID
        try:
            self.fid_metric = FrechetInceptionDistance(normalize=True).to(device)
            logger.info("✓ FID metric initialized")
        except Exception as e:
            logger.warning(f"⚠ Failed to initialize FID: {e}")
            self.fid_metric = None

    def calculate_metrics(
        self,
        generated_images: torch.Tensor,
        target_images: torch.Tensor,
        prompts: List[str] = None,
    ) -> Dict[str, float]:
        """Calculate all metrics for a batch of images."""
        logger.debug("Calculating metrics for batch...")

        # Validate input tensors
        if not validate_tensor_ranges(generated_images, "Generated images", "[-1, 1]"):
            logger.warning("Generated images tensor validation failed")
        if not validate_tensor_ranges(target_images, "Target images", "[-1, 1]"):
            logger.warning("Target images tensor validation failed")

        metrics = {}

        with torch.no_grad():
            # Ensure images are in the correct range [-1, 1]
            gen_img = generated_images.float()
            tgt_img = target_images.float()

            # PSNR
            if self.psnr is not None:
                try:
                    psnr_val = self.psnr(gen_img, tgt_img).item()
                    metrics["psnr"] = psnr_val
                    logger.debug(f"PSNR calculated: {psnr_val:.4f}")
                except Exception as e:
                    logger.warning(f"PSNR calculation failed: {e}")
                    metrics["psnr"] = 0.0
            else:
                metrics["psnr"] = 0.0

            # SSIM
            if self.ssim is not None:
                try:
                    ssim_val = self.ssim(gen_img, tgt_img).item()
                    metrics["ssim"] = ssim_val
                    logger.debug(f"SSIM calculated: {ssim_val:.4f}")
                except Exception as e:
                    logger.warning(f"SSIM calculation failed: {e}")
                    metrics["ssim"] = 0.0
            else:
                metrics["ssim"] = 0.0

            # Perceptual Loss (VGG-based)
            if self.perceptual_loss is not None:
                try:
                    perceptual_val = self.perceptual_loss(gen_img, tgt_img).item()
                    metrics["perceptual_loss"] = perceptual_val
                    logger.debug(f"Perceptual loss calculated: {perceptual_val:.4f}")
                except Exception as e:
                    logger.warning(f"Perceptual loss calculation failed: {e}")
                    metrics["perceptual_loss"] = 0.0
            else:
                metrics["perceptual_loss"] = 0.0

            # LPIPS
            if self.lpips_metric is not None:
                try:
                    lpips_val = self.lpips_metric(gen_img, tgt_img).mean().item()
                    metrics["lpips"] = lpips_val
                    logger.debug(f"LPIPS calculated: {lpips_val:.4f}")
                except Exception as e:
                    logger.warning(f"LPIPS calculation failed: {e}")
                    metrics["lpips"] = 0.0
            else:
                metrics["lpips"] = 0.0

            # CLIP Score (if prompts are provided)
            if self.clip_score_metric is not None and prompts is not None:
                try:
                    # Convert to uint8 for CLIP
                    gen_uint8 = ((gen_img.clamp(-1, 1) + 1) / 2.0 * 255).to(torch.uint8)
                    validate_tensor_ranges(gen_uint8.float(), "CLIP input", "[0, 255]")
                    clip_val = self.clip_score_metric(gen_uint8, prompts).item()
                    metrics["clip_score"] = clip_val
                    logger.debug(f"CLIP score calculated: {clip_val:.4f}")
                except Exception as e:
                    logger.warning(f"CLIP score calculation failed: {e}")
                    metrics["clip_score"] = 0.0
            else:
                metrics["clip_score"] = 0.0

            # FID (accumulate for batch processing)
            if self.fid_metric is not None:
                try:
                    # Convert to [0, 1] range for FID
                    gen_fid = (gen_img.clamp(-1, 1) + 1) / 2.0
                    tgt_fid = (tgt_img.clamp(-1, 1) + 1) / 2.0

                    validate_tensor_ranges(gen_fid, "FID generated input", "[0, 1]")
                    validate_tensor_ranges(tgt_fid, "FID target input", "[0, 1]")

                    self.fid_metric.update(gen_fid, real=False)
                    self.fid_metric.update(tgt_fid, real=True)
                    logger.debug("FID metric updated")
                except Exception as e:
                    logger.warning(f"FID update failed: {e}")

        return metrics

    def compute_fid(self) -> float:
        """Compute final FID score."""
        if self.fid_metric is not None:
            try:
                fid_val = self.fid_metric.compute().item()
                logger.info(f"Final FID computed: {fid_val:.4f}")
                return fid_val
            except Exception as e:
                logger.warning(f"FID computation failed: {e}")
                return 0.0
        return 0.0

    def reset_fid(self):
        """Reset FID metric for new evaluation."""
        if self.fid_metric is not None:
            self.fid_metric.reset()
            logger.debug("FID metric reset")


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file."""
    logger.info(f"Loading configuration from: {config_path}")

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Log key configuration parameters
    logger.info("=== CONFIGURATION ===")
    logger.info(f"Architecture: {config.get('architecture', 'Not specified')}")
    logger.info(f"Image size: {config.get('image_size', 'Not specified')}")
    logger.info(f"Batch size: {config.get('batch_size', 'Not specified')}")
    logger.info(
        f"Use camera conditioning: {config.get('use_camera_conditioning', 'Not specified')}"
    )
    logger.info(
        f"Use image conditioning: {config.get('use_image_conditioning', 'Not specified')}"
    )
    logger.info("=== CONFIGURATION END ===")

    return config


def setup_pipeline(config: Dict, checkpoint_path: str, device: str):
    """Setup the MVD pipeline with trained weights."""
    logger.info(f"Setting up pipeline on device: {device}")
    logger.info(f"Creating pipeline from {config['architecture']}")

    # Validate checkpoint exists
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    cache_dir = None
    if device == "cuda":
        torch.set_float32_matmul_precision("high")
        SCRATCH = os.getenv("SCRATCH", "/net/tscratch/people/plgewoj")
        HUGGINGFACE_CACHE = os.path.join(SCRATCH, "huggingface_cache")
        os.makedirs(HUGGINGFACE_CACHE, exist_ok=True)
        os.environ["HF_HOME"] = HUGGINGFACE_CACHE
        cache_dir = HUGGINGFACE_CACHE
        logger.info(f"Using HuggingFace cache: {cache_dir}")

    try:
        pipeline = create_mvd_pipeline(
            pretrained_model_name_or_path=config["architecture"],
            use_memory_efficient_attention=config["use_memory_efficient_attention"],
            enable_gradient_checkpointing=config["enable_gradient_checkpointing"],
            dtype=getattr(torch, config["torch_dtype"]),
            use_camera_conditioning=config["use_camera_conditioning"],
            use_image_conditioning=config["use_image_conditioning"],
            img_ref_scale=config["img_ref_scale"],
            cam_modulation_strength=config["cam_modulation_strength"],
            cache_dir=cache_dir,
            scheduler_config=config.get("scheduler_config", {}),
        )
        logger.info("✓ Pipeline created successfully")
    except Exception as e:
        logger.error(f"✗ Failed to create pipeline: {e}")
        raise

    logger.info(f"Loading checkpoint from {checkpoint_path}")
    try:
        state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)["state_dict"]
        logger.info(f"✓ Checkpoint loaded with {len(state_dict)} keys")

        unet_keys = [k for k in state_dict.keys() if k.startswith("unet.")]
        logger.info(f"Found {len(unet_keys)} UNet keys in checkpoint")

        unet_state_dict = {
            k.replace("unet.", ""): v for k, v in state_dict.items() if k in unet_keys
        }
        missing_keys, unexpected_keys = pipeline.unet.load_state_dict(
            unet_state_dict, strict=False
        )

        if missing_keys:
            logger.warning(
                f"Missing keys in checkpoint: {missing_keys[:5]}..."
            )  # Show first 5
        if unexpected_keys:
            logger.warning(
                f"Unexpected keys in checkpoint: {unexpected_keys[:5]}..."
            )  # Show first 5

        logger.info("✓ Checkpoint loaded into UNet")
    except Exception as e:
        logger.error(f"✗ Failed to load checkpoint: {e}")
        raise

    logger.info("Moving pipeline to device and setting to eval mode")
    pipeline = pipeline.to(device)
    pipeline.unet.eval()
    pipeline.vae.eval()
    pipeline.text_encoder.eval()

    if (
        hasattr(pipeline.unet, "image_encoder")
        and pipeline.unet.image_encoder is not None
    ):
        pipeline.unet.image_encoder.eval()
        logger.info("✓ Image encoder set to eval mode")
    if (
        hasattr(pipeline.unet, "camera_encoder")
        and pipeline.unet.camera_encoder is not None
    ):
        pipeline.unet.camera_encoder.eval()
        logger.info("✓ Camera encoder set to eval mode")

    logger.info("✓ Pipeline setup complete")
    return pipeline


def run_validation(
    pipeline, dataset: ObjaverseDataset, config: Dict, device: str, output_dir: Path
):
    """Run validation on the dataset and calculate metrics."""

    # Setup metrics
    metrics_calculator = ValidationMetrics(
        device, config.get("clip_model_name", "openai/clip-vit-large-patch14")
    )

    # Setup dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=True,
    )

    # Results storage
    all_results = []
    batch_metrics = []

    logger.info(f"Starting validation on {len(dataset)} samples")

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Validating")):
            try:
                # Move batch to device
                source_images = batch["source_image"].to(device)
                target_images = batch["target_image"].to(device)
                source_camera = batch.get("source_camera", None)
                target_camera = batch.get("target_camera", None)
                prompts = batch["prompt"]
                object_uids = batch["object_uid"]

                if source_camera is not None:
                    source_camera = source_camera.to(device)
                if target_camera is not None:
                    target_camera = target_camera.to(device)

                # Validate input tensors
                validate_tensor_ranges(
                    source_images, f"Batch {batch_idx} source images"
                )
                validate_tensor_ranges(
                    target_images, f"Batch {batch_idx} target images"
                )

                # Run inference
                logger.debug(f"Running inference for batch {batch_idx}")
                pipeline_output = pipeline(
                    prompt=prompts,
                    num_inference_steps=config["num_inference_steps"],
                    source_camera=source_camera,
                    target_camera=target_camera,
                    source_images=source_images,
                    guidance_scale=config["guidance_scale"],
                    ref_scale=config["ref_scale"],
                    output_type="pt",
                    use_camera_embeddings=config["use_camera_conditioning"],
                    use_image_conditioning=config["use_image_conditioning"],
                )

                generated_images = pipeline_output["images"]
                # Convert from [0, 1] to [-1, 1] to match target_images
                generated_images = (generated_images * 2.0 - 1.0).to(device)

                # Validate generated images
                validate_tensor_ranges(
                    generated_images, f"Batch {batch_idx} generated images"
                )

                # Calculate metrics for this batch
                batch_metrics_dict = metrics_calculator.calculate_metrics(
                    generated_images, target_images, prompts
                )

                # Store individual sample results
                for i in range(len(object_uids)):
                    sample_metrics = {}
                    for metric_name, metric_value in batch_metrics_dict.items():
                        if metric_name != "clip_score":  # CLIP score is per-batch
                            # For per-sample metrics, we need to calculate individually
                            if (
                                metric_name == "psnr"
                                and metrics_calculator.psnr is not None
                            ):
                                sample_metrics[metric_name] = metrics_calculator.psnr(
                                    generated_images[i : i + 1],
                                    target_images[i : i + 1],
                                ).item()
                            elif (
                                metric_name == "ssim"
                                and metrics_calculator.ssim is not None
                            ):
                                sample_metrics[metric_name] = metrics_calculator.ssim(
                                    generated_images[i : i + 1],
                                    target_images[i : i + 1],
                                ).item()
                            elif (
                                metric_name == "lpips"
                                and metrics_calculator.lpips_metric is not None
                            ):
                                sample_metrics[metric_name] = (
                                    metrics_calculator.lpips_metric(
                                        generated_images[i : i + 1],
                                        target_images[i : i + 1],
                                    ).item()
                                )
                            elif (
                                metric_name == "perceptual_loss"
                                and metrics_calculator.perceptual_loss is not None
                            ):
                                sample_metrics[metric_name] = (
                                    metrics_calculator.perceptual_loss(
                                        generated_images[i : i + 1],
                                        target_images[i : i + 1],
                                    ).item()
                                )
                            else:
                                sample_metrics[metric_name] = 0.0
                        else:
                            sample_metrics[metric_name] = (
                                metric_value  # Use batch value for CLIP
                            )

                    all_results.append(
                        {
                            "object_uid": object_uids[i],
                            "prompt": prompts[i],
                            **sample_metrics,
                        }
                    )

                batch_metrics.append(batch_metrics_dict)

                # Save sample images periodically
                if batch_idx % 10 == 0:
                    save_sample_images(
                        source_images,
                        target_images,
                        generated_images,
                        object_uids,
                        output_dir,
                        batch_idx,
                    )

                # Log progress
                if batch_idx % 5 == 0:
                    logger.info(f"Processed batch {batch_idx}/{len(dataloader)}")

            except Exception as e:
                logger.error(f"Error processing batch {batch_idx}: {e}")
                continue

    # Compute final FID
    final_fid = metrics_calculator.compute_fid()
    logger.info(f"Final FID score: {final_fid}")

    # Calculate overall statistics
    overall_metrics = calculate_overall_metrics(batch_metrics, final_fid)

    # Save results
    save_results(all_results, overall_metrics, output_dir)

    return all_results, overall_metrics


def save_sample_images(
    source_images, target_images, generated_images, object_uids, output_dir, batch_idx
):
    """Save sample images for visual inspection."""
    samples_dir = output_dir / "samples"
    samples_dir.mkdir(exist_ok=True, parents=True)

    # Convert tensors to PIL images
    def tensor_to_pil(tensor):
        # Convert from [-1, 1] to [0, 1] then to [0, 255]
        img = ((tensor.cpu().permute(1, 2, 0) + 1) / 2 * 255).numpy().astype("uint8")
        return Image.fromarray(img)

    # Save first image from batch
    if len(source_images) > 0:
        try:
            source_pil = tensor_to_pil(source_images[0])
            target_pil = tensor_to_pil(target_images[0])
            generated_pil = tensor_to_pil(generated_images[0])

            # Create comparison image
            total_width = source_pil.width + target_pil.width + generated_pil.width
            max_height = max(source_pil.height, target_pil.height, generated_pil.height)
            comparison = Image.new("RGB", (total_width, max_height))

            comparison.paste(source_pil, (0, 0))
            comparison.paste(target_pil, (source_pil.width, 0))
            comparison.paste(generated_pil, (source_pil.width + target_pil.width, 0))

            comparison.save(samples_dir / f"batch_{batch_idx:04d}_{object_uids[0]}.png")
            logger.debug(f"Saved sample images for batch {batch_idx}")
        except Exception as e:
            logger.warning(f"Failed to save sample images for batch {batch_idx}: {e}")


def calculate_overall_metrics(
    batch_metrics: List[Dict], final_fid: float
) -> Dict[str, float]:
    """Calculate overall statistics from batch metrics."""
    if not batch_metrics:
        return {}

    # Aggregate metrics
    aggregated = {}
    for metric_name in batch_metrics[0].keys():
        if metric_name != "clip_score":  # Skip CLIP score for now
            values = [
                batch[metric_name]
                for batch in batch_metrics
                if metric_name in batch and batch[metric_name] > 0
            ]
            if values:
                aggregated[f"{metric_name}_mean"] = np.mean(values)
                aggregated[f"{metric_name}_std"] = np.std(values)
                aggregated[f"{metric_name}_min"] = np.min(values)
                aggregated[f"{metric_name}_max"] = np.max(values)

    # Add FID
    aggregated["fid"] = final_fid

    # Add CLIP score if available
    clip_scores = [batch.get("clip_score", 0) for batch in batch_metrics]
    valid_clip_scores = [score for score in clip_scores if score > 0]
    if valid_clip_scores:
        aggregated["clip_score_mean"] = np.mean(valid_clip_scores)
        aggregated["clip_score_std"] = np.std(valid_clip_scores)

    return aggregated


def save_results(all_results: List[Dict], overall_metrics: Dict, output_dir: Path):
    """Save validation results to CSV files."""

    # Save individual results
    results_file = output_dir / "validation_results.csv"
    if all_results:
        fieldnames = all_results[0].keys()
        with open(results_file, "w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_results)
        logger.info(f"Individual results saved to {results_file}")

    # Save overall metrics
    overall_file = output_dir / "overall_metrics.csv"
    with open(overall_file, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Metric", "Value"])
        for metric, value in overall_metrics.items():
            writer.writerow([metric, value])
    logger.info(f"Overall metrics saved to {overall_file}")

    # Print summary
    logger.info("=== VALIDATION SUMMARY ===")
    for metric, value in overall_metrics.items():
        logger.info(f"{metric}: {value:.4f}")


def main():
    parser = argparse.ArgumentParser(description="GSO Dataset Validation Script")
    parser.add_argument(
        "--ckpt", type=str, required=True, help="Path to trained checkpoint"
    )
    parser.add_argument(
        "--dataset-path", type=str, required=True, help="Path to GSO dataset"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/infer_config.yaml",
        help="Path to inference configuration file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/gso_validation",
        help="Output directory for results",
    )

    args = parser.parse_args()

    # Load configuration
    try:
        config = load_config(args.config)
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        return 1

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    logger.info(f"Output directory: {output_dir}")

    # Setup pipeline
    try:
        pipeline = setup_pipeline(config, args.ckpt, device)
    except Exception as e:
        logger.error(f"Failed to setup pipeline: {e}")
        return 1

    # Setup dataset
    logger.info(f"Loading dataset from {args.dataset_path}")
    try:
        dataset = ObjaverseDataset(
            data_root=args.dataset_path,
            split="test",  # Use test split for validation
            target_size=(config["image_size"][0], config["image_size"][1]),
            max_views_per_object=config["max_views_per_object"],
        )
        logger.info(f"Dataset loaded with {len(dataset)} samples")
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        return 1

    # Run validation
    try:
        start_time = time.time()
        all_results, overall_metrics = run_validation(
            pipeline, dataset, config, device, output_dir
        )
        end_time = time.time()

        logger.info(f"Validation completed in {end_time - start_time:.2f} seconds")
        logger.info(f"Results saved to {output_dir}")
        return 0
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())


# uv run gso_val.py --ckpt /net/pr2/projects/plgrid/plggtattooai/code/eryk/MVD/outputs/2025-05-23_13-39-27/checkpoints/last.ckpt --dataset-path /net/pr2/projects/plgrid/plggtattooai/MeshDatasets/gso_rendered_output/
