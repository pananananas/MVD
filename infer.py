from src.utils import create_output_dirs, load_image, create_camera_matrix
from src.models.mvd_unet import create_mvd_pipeline
from pathlib import Path, PosixPath
from PIL import Image
import argparse
import logging
import torch
import os

torch.serialization.add_safe_globals([PosixPath])
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main(args):
    dirs = create_output_dirs("outputs/infer")

    cache_dir = None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda":
        torch.set_float32_matmul_precision("high")
        SCRATCH = os.getenv("SCRATCH", "/net/tscratch/people/plgewoj")
        HUGGINGFACE_CACHE = os.path.join(SCRATCH, "huggingface_cache")
        os.makedirs(HUGGINGFACE_CACHE, exist_ok=True)
        os.environ["HF_HOME"] = HUGGINGFACE_CACHE
        cache_dir = HUGGINGFACE_CACHE

    num_generated_images = args.num_generated_images

    pipeline = create_mvd_pipeline(
        pretrained_model_name_or_path=args.base_model,
        use_memory_efficient_attention=True,
        enable_gradient_checkpointing=False,
        dtype=torch.float32,
        use_camera_conditioning=True,
        use_image_conditioning=True,
        # cam_output_dim=2048,
        # cam_hidden_dim=1024,
        simple_cam_encoder=False,
        cache_dir=cache_dir,
    )

    state_dict = torch.load(args.checkpoint, map_location="cpu")["state_dict"]

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
        logger.warning(f"Missing keys in UNet state_dict: {missing_keys[:5]}...")
    if unexpected_keys:
        logger.warning(f"Unexpected keys in UNet state_dict: {unexpected_keys[:5]}...")

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

    source_image = load_image(
        args.source_image, target_size=(args.image_size, args.image_size)
    )
    source_image = source_image.to(device)

    source_pos = [0, 0, 2.0]
    target_pos = [1.5, 0, 1.5]
    source_camera = create_camera_matrix(source_pos, [0, 0, 0])
    target_camera = create_camera_matrix(target_pos, [0, 0, 0])

    source_camera = source_camera.unsqueeze(0).to(device)
    target_camera = target_camera.unsqueeze(0).to(device)

    logger.info("Running inference...")
    logger.info(f"Prompt: {args.prompt}")
    logger.info(f"Source camera shape: {source_camera.shape}")
    logger.info(f"Target camera shape: {target_camera.shape}")
    logger.info(f"Source image shape: {source_image.shape}")

    for i in range(num_generated_images):
        with torch.no_grad():
            output = pipeline(
                prompt=args.prompt,
                num_inference_steps=args.steps,
                source_camera=source_camera,
                target_camera=target_camera,
                source_images=source_image,
                guidance_scale=args.guidance_scale,
                ref_scale=args.ref_scale,
                output_type="pt",
            )

        images = output["images"]

        images = (images + 1.0) / 2.0
        images = images.clamp(0, 1)

        images = (images * 255).round().cpu().numpy().astype("uint8")
        images = images.squeeze(0).transpose(1, 2, 0)
        output_image = Image.fromarray(images)

        source_img_pil = (
            ((source_image[0].cpu().permute(1, 2, 0) + 1) / 2 * 255)
            .numpy()
            .astype("uint8")
        )
        source_img_pil = Image.fromarray(source_img_pil)

        output_dir = dirs["samples"] / f"inference_{Path(args.source_image).stem}"
        output_dir.mkdir(exist_ok=True, parents=True)

        source_img_pil.save(output_dir / "source.png")
        output_image.save(output_dir / f"generated_{i}.png")

        combined = Image.new(
            "RGB",
            (
                source_img_pil.width + output_image.width,
                max(source_img_pil.height, output_image.height),
            ),
        )
        combined.paste(source_img_pil, (0, 0))
        combined.paste(output_image, (source_img_pil.width, 0))
        combined.save(output_dir / f"comparison_{i}.png")

    logger.info(f"Results saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MVD Inference")
    parser.add_argument(
        "--base-model",
        type=str,
        default="stabilityai/stable-diffusion-2-1",
        help="Base model path",
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to trained checkpoint"
    )
    parser.add_argument(
        "--source-image", type=str, default="imgs/004.png", help="Path to source image"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Create an abstract sculpture or figurine with a smooth, matte finish in a pale yellow or beige color. The object should have a simplified human profile with a contemplative expression, featuring a slightly elongated head and shoulders, and a slightly tilted head. The overall shape should be elongated, with the head being the most prominent feature.",
        help="Text prompt",
    )
    parser.add_argument(
        "--steps", type=int, default=20, help="Number of inference steps"
    )
    parser.add_argument(
        "--guidance-scale", type=float, default=1.0, help="Guidance scale"
    )
    parser.add_argument("--ref-scale", type=float, default=1.0, help="Reference scale")
    parser.add_argument("--image-size", type=int, default=768, help="Image size")
    parser.add_argument(
        "--num-generated-images", type=int, default=8, help="Number of generated images"
    )
    args = parser.parse_args()
    main(args)
