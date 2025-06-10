import argparse
import csv
import datetime
import logging
import os
from pathlib import Path, PosixPath
from typing import Dict, List
import time

import lovely_tensors as lt
import lpips
import numpy as np
import torch
import yaml

# from icecream import ic
from PIL import Image, ImageDraw
from pytorch_msssim import SSIM
from torch.utils.data import DataLoader
from torchmetrics.image import FrechetInceptionDistance, PeakSignalNoiseRatio
from torchmetrics.multimodal import CLIPScore
from tqdm import tqdm

from src.data.objaverse_dataset import ObjaverseDataset
from src.models.mvd_unet import create_mvd_pipeline
from src.training.losses import PerceptualLoss

lt.monkey_patch()
torch.serialization.add_safe_globals([PosixPath])

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def validate_tensor_ranges(
    tensor: torch.Tensor, name: str, expected_range: str = "[-1, 1]"
):
    min_val = tensor.min().item()
    max_val = tensor.max().item()

    if expected_range == "[-1, 1]":
        if min_val < -1.1 or max_val > 1.1:
            logger.warning(f"{name} values of tensor {name}: {tensor}]")
    elif expected_range == "[0, 1]":
        if min_val < -0.1 or max_val > 1.1:
            logger.warning(f"{name} values of tensor {name}: {tensor}]")

    if torch.isnan(tensor).any():
        logger.error(f"{name} contains NaN values!")
        return False
    if torch.isinf(tensor).any():
        logger.error(f"{name} contains Inf values!")
        return False

    return True


class ValidationMetrics:
    """Class to handle all validation metrics calculation."""

    def __init__(
        self, device: str, clip_model_name: str = "openai/clip-vit-large-patch14"
    ):
        self.device = device

        try:
            self.ssim = SSIM(data_range=2.0, size_average=True).to(device)
        except Exception as e:
            logger.error(f"Failed to initialize SSIM: {e}")
            self.ssim = None

        try:
            self.psnr = PeakSignalNoiseRatio(data_range=2.0).to(device)
        except Exception as e:
            logger.error(f"Failed to initialize PSNR: {e}")
            self.psnr = None

        try:
            self.perceptual_loss = PerceptualLoss(device=device)
        except Exception as e:
            logger.error(f"Failed to initialize Perceptual loss: {e}")
            self.perceptual_loss = None

        try:
            self.lpips_metric = lpips.LPIPS(net="alex").to(device)
        except Exception as e:
            logger.error(f"Failed to initialize LPIPS: {e}")
            self.lpips_metric = None

        try:
            self.clip_score_metric = CLIPScore(model_name_or_path=clip_model_name).to(
                device
            )
        except Exception as e:
            logger.warning(f"Failed to initialize CLIPScore: {e}")
            self.clip_score_metric = None

        try:
            self.fid_metric = FrechetInceptionDistance(normalize=True).to(device)
        except Exception as e:
            logger.warning(f"Failed to initialize FID: {e}")
            self.fid_metric = None

    def calculate_metrics(
        self,
        generated_images: torch.Tensor,
        target_images: torch.Tensor,
        prompts: List[str] = None,
    ) -> Dict[str, float]:
        metrics = {}

        with torch.no_grad():
            gen_img = generated_images.float()
            tgt_img = target_images.float()

            if self.psnr is not None:
                try:
                    psnr_val = self.psnr(gen_img, tgt_img).item()
                    metrics["psnr"] = psnr_val
                except Exception as e:
                    logger.warning(f"PSNR calculation failed: {e}")
                    metrics["psnr"] = 0.0
            else:
                metrics["psnr"] = 0.0

            if self.ssim is not None:
                try:
                    ssim_val = self.ssim(gen_img, tgt_img).item()
                    metrics["ssim"] = ssim_val
                except Exception as e:
                    logger.warning(f"SSIM calculation failed: {e}")
                    metrics["ssim"] = 0.0
            else:
                metrics["ssim"] = 0.0

            if self.perceptual_loss is not None:
                try:
                    perceptual_val = self.perceptual_loss(gen_img, tgt_img).item()
                    metrics["perceptual_loss"] = perceptual_val
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
                except Exception as e:
                    logger.warning(f"LPIPS calculation failed: {e}")
                    metrics["lpips"] = 0.0
            else:
                metrics["lpips"] = 0.0

            if self.clip_score_metric is not None and prompts is not None:
                try:
                    gen_uint8 = ((gen_img.clamp(-1, 1) + 1) / 2.0 * 255).to(torch.uint8)
                    validate_tensor_ranges(gen_uint8.float(), "CLIP input", "[0, 255]")
                    clip_val = self.clip_score_metric(gen_uint8, prompts).item()
                    metrics["clip_score"] = clip_val
                except Exception as e:
                    logger.warning(f"CLIP score calculation failed: {e}")
                    metrics["clip_score"] = 0.0
            else:
                metrics["clip_score"] = 0.0

            if self.fid_metric is not None:
                try:
                    gen_fid = (gen_img.clamp(-1, 1) + 1) / 2.0
                    tgt_fid = (tgt_img.clamp(-1, 1) + 1) / 2.0
                    self.fid_metric.update(gen_fid, real=False)
                    self.fid_metric.update(tgt_fid, real=True)

                except Exception as e:
                    logger.warning(f"FID update failed: {e}")

        return metrics

    def compute_fid(self) -> float:
        if self.fid_metric is not None:
            try:
                fid_val = self.fid_metric.compute().item()
                return fid_val
            except Exception as e:
                logger.warning(f"FID computation failed: {e}")
                return 0.0
        return 0.0

    def reset_fid(self):
        if self.fid_metric is not None:
            self.fid_metric.reset()


def load_config(config_path: str) -> Dict:
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return config


def setup_pipeline(config: Dict, checkpoint_path: str, device: str):
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
            cam_output_dim=config["cam_output_dim"],
            cam_hidden_dim=config["cam_hidden_dim"],
            simple_cam_encoder=config["simple_encoder"],
            cache_dir=cache_dir,
            scheduler_config=config.get("scheduler_config", {}),
        )
    except Exception as e:
        logger.error(f"Failed to create pipeline: {e}")
        raise

    try:
        state_dict = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )["state_dict"]

        unet_checkpoint_keys = {
            k: v for k, v in state_dict.items() if k.startswith("unet.")
        }

        state_dict_for_mv_unet = {
            k.replace("unet.", "", 1): v for k, v in unet_checkpoint_keys.items()
        }

        fixed_state_dict = {}
        for key, value in state_dict_for_mv_unet.items():
            new_key = key
            if key.startswith("image_encoder.") and not key.startswith(
                "image_encoder.unet."
            ):
                parts = key.split(".", 1)
                if len(parts) == 2 and parts[0] == "image_encoder":
                    new_key = f"image_encoder.unet.{parts[1]}"

            fixed_state_dict[new_key] = value

        missing_keys, unexpected_keys = pipeline.unet.load_state_dict(
            fixed_state_dict, strict=False
        )

        if missing_keys:
            logger.warning(f"Missing keys in checkpoint: {missing_keys[:5]}...")
        if unexpected_keys:
            logger.warning(f"Unexpected keys in checkpoint: {unexpected_keys[:5]}...")

    except Exception as e:
        logger.error(f"Failed to load checkpoint: {e}")
        raise

    pipeline = pipeline.to(device)
    pipeline.unet.eval()
    pipeline.vae.eval()
    pipeline.text_encoder.eval()

    if (
        hasattr(pipeline.unet, "image_encoder")
        and pipeline.unet.image_encoder is not None
    ):
        pipeline.unet.image_encoder.eval()
    if (
        hasattr(pipeline.unet, "camera_encoder")
        and pipeline.unet.camera_encoder is not None
    ):
        pipeline.unet.camera_encoder.eval()

    return pipeline


def run_validation(
    pipeline, dataset: ObjaverseDataset, config: Dict, device: str, output_dir: Path
):
    metrics_calculator = ValidationMetrics(
        device, config.get("clip_model_name", "openai/clip-vit-large-patch14")
    )

    dataloader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=True,
    )

    all_results = []
    batch_metrics = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Validating")):
            try:
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

                inference_start_time = time.perf_counter()
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
                inference_end_time = time.perf_counter()
                current_batch_inference_time = inference_end_time - inference_start_time
                num_samples_in_batch = len(object_uids)
                logger.info(f"Batch {batch_idx} inference time: {current_batch_inference_time:.4f} seconds for {num_samples_in_batch} samples.")

                generated_images = pipeline_output["images"]
                generated_images = (generated_images * 2.0 - 1.0).to(device)
                
                batch_metrics_dict = metrics_calculator.calculate_metrics(
                    generated_images, target_images, prompts
                )
                batch_metrics_dict['batch_inference_time_seconds'] = current_batch_inference_time
                batch_metrics_dict['num_samples_in_batch'] = num_samples_in_batch


                per_sample_inference_time = current_batch_inference_time / num_samples_in_batch if num_samples_in_batch > 0 else 0

                for i in range(len(object_uids)):
                    sample_metrics = {}
                    sample_metrics['inference_time_seconds'] = per_sample_inference_time
                    for metric_name, metric_value in batch_metrics_dict.items():
                        if metric_name not in ["clip_score", 'batch_inference_time_seconds', 'num_samples_in_batch']:
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
                            sample_metrics[metric_name] = metric_value

                    all_results.append(
                        {
                            "object_uid": object_uids[i],
                            "prompt": prompts[i],
                            **sample_metrics,
                        }
                    )

                save_sample_images(
                    source_image_tensor=source_images[i],
                    target_image_tensor=target_images[i],
                    generated_image_tensor=generated_images[i],
                    object_uid=object_uids[i],
                    metrics=sample_metrics,
                    output_dir=output_dir,
                    batch_idx=batch_idx,
                )

                batch_metrics.append(batch_metrics_dict)

            except Exception as e:
                logger.error(f"Error processing batch {batch_idx}: {e}")
                continue

    final_fid = metrics_calculator.compute_fid()

    overall_metrics = calculate_overall_metrics(batch_metrics, final_fid)

    save_results(all_results, overall_metrics, output_dir)

    return all_results, overall_metrics


def save_sample_images(
    source_image_tensor: torch.Tensor,
    target_image_tensor: torch.Tensor,
    generated_image_tensor: torch.Tensor,
    object_uid: str,
    metrics: Dict[str, float],
    output_dir: Path,
    batch_idx: int,
):
    samples_dir = output_dir / "samples"
    samples_dir.mkdir(exist_ok=True, parents=True)

    def tensor_to_pil(tensor):
        img = ((tensor.cpu().permute(1, 2, 0) + 1) / 2 * 255).numpy().astype("uint8")
        return Image.fromarray(img)

    try:
        source_pil = tensor_to_pil(source_image_tensor)
        target_pil = tensor_to_pil(target_image_tensor)
        generated_pil = tensor_to_pil(generated_image_tensor)

        total_width = source_pil.width + target_pil.width + generated_pil.width
        max_height = source_pil.height

        lpips_val = metrics.get("lpips", 0.0)
        psnr_val = metrics.get("psnr", 0.0)
        ssim_val = metrics.get("ssim", 0.0)
        
        header_text = f"UID: {object_uid} | LPIPS: {lpips_val:.3f} | PSNR: {psnr_val:.2f} | SSIM: {ssim_val:.3f}"
        
        header_height = 30
        final_image_height = max_height + header_height

        comparison_img = Image.new("RGB", (total_width, final_image_height), (255, 255, 255))
        draw = ImageDraw.Draw(comparison_img)

        try:
            draw.text((10, 5), header_text, fill="black")
        except Exception as font_e:
            logger.warning(f"Failed to draw text on image for UID {object_uid} (batch {batch_idx}): {font_e}. Text was: '{header_text}'")
            draw.text((10,5), f"UID: {object_uid} (Metrics text render error)", fill="black")

        # Paste the images below the header
        comparison_img.paste(source_pil, (0, header_height))
        comparison_img.paste(target_pil, (source_pil.width, header_height))
        comparison_img.paste(generated_pil, (source_pil.width + target_pil.width, header_height))

        filename = f"comparison_batch{batch_idx:04d}_uid_{object_uid}.png"
        comparison_img.save(samples_dir / filename)

    except Exception as e:
        logger.warning(f"Failed to save sample comparison image for UID {object_uid}, batch {batch_idx}: {e}")


def calculate_overall_metrics(
    batch_metrics: List[Dict], final_fid: float
) -> Dict[str, float]:
    if not batch_metrics:
        return {}

    aggregated = {}
    for metric_name in batch_metrics[0].keys():
        if metric_name not in ["clip_score", 'batch_inference_time_seconds', 'num_samples_in_batch']:
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

    aggregated["fid"] = final_fid

    clip_scores = [batch.get("clip_score", 0) for batch in batch_metrics]
    valid_clip_scores = [score for score in clip_scores if score > 0]
    if valid_clip_scores:
        aggregated["clip_score_mean"] = np.mean(valid_clip_scores)
        aggregated["clip_score_std"] = np.std(valid_clip_scores)

    all_batch_inference_times = [b['batch_inference_time_seconds'] for b in batch_metrics if 'batch_inference_time_seconds' in b]
    num_samples_per_batch_list = [b['num_samples_in_batch'] for b in batch_metrics if 'num_samples_in_batch' in b]

    if all_batch_inference_times:
        aggregated['mean_batch_inference_time_seconds'] = np.mean(all_batch_inference_times)
        aggregated['std_batch_inference_time_seconds'] = np.std(all_batch_inference_times)
        aggregated['min_batch_inference_time_seconds'] = np.min(all_batch_inference_times)
        aggregated['max_batch_inference_time_seconds'] = np.max(all_batch_inference_times)
        
        total_inference_duration = np.sum(all_batch_inference_times)
        aggregated['total_inference_duration_seconds'] = total_inference_duration
        
        total_samples_processed = np.sum(num_samples_per_batch_list)
        aggregated['total_samples_processed_in_inference'] = total_samples_processed
        
        if total_samples_processed > 0:
            aggregated['avg_per_sample_inference_time_seconds'] = total_inference_duration / total_samples_processed
        else:
            aggregated['avg_per_sample_inference_time_seconds'] = 0.0
    else:
        aggregated['mean_batch_inference_time_seconds'] = 0.0
        aggregated['total_inference_duration_seconds'] = 0.0
        aggregated['total_samples_processed_in_inference'] = 0.0
        aggregated['avg_per_sample_inference_time_seconds'] = 0.0


    return aggregated


def save_results(all_results: List[Dict], overall_metrics: Dict, output_dir: Path):
    results_file = output_dir / "validation_results.csv"
    if all_results:
        fieldnames = all_results[0].keys()
        with open(results_file, "w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_results)

    overall_file = output_dir / "overall_metrics.csv"
    with open(overall_file, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Metric", "Value"])
        for metric, value in overall_metrics.items():
            writer.writerow([metric, value])


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

    try:
        config = load_config(args.config)
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        return 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = Path(args.output_dir) / timestamp
    output_dir.mkdir(exist_ok=True, parents=True)

    try:
        pipeline = setup_pipeline(config, args.ckpt, device)
    except Exception as e:
        logger.error(f"Failed to setup pipeline: {e}")
        return 1

    try:
        dataset = ObjaverseDataset(
            data_root=args.dataset_path,
            split="test",
            target_size=(config["image_size"][0], config["image_size"][1]),
            max_views_per_object=config["max_views_per_object"],
        )
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        return 1

    try:
        all_results, overall_metrics = run_validation(
            pipeline, dataset, config, device, output_dir
        )
        logger.info(f"Calculated metrics: {overall_metrics}")
        logger.info(f"All results: {all_results}")

        return 0
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())


# uv run val.py --ckpt /net/pr2/projects/plgrid/plggtattooai/code/eryk/MVD/outputs/2025-06-05_00-20-32/checkpoints/last.ckpt --dataset-path /net/pr2/projects/plgrid/plggtattooai/MeshDatasets/objaverse/
