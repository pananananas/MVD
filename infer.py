from src.utils import create_output_dirs, load_image, create_camera_matrix
from src.models.mvd_unet import create_mvd_pipeline
from pathlib import Path, PosixPath
from PIL import Image
import argparse
import logging
import torch
import os

torch.serialization.add_safe_globals([PosixPath])
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main(args):
    dirs = create_output_dirs("outputs/infer")

    cache_dir = None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda":
        torch.set_float32_matmul_precision('high')
        SCRATCH = os.getenv('SCRATCH', '/net/tscratch/people/plgewoj')
        HUGGINGFACE_CACHE = os.path.join(SCRATCH, 'huggingface_cache')
        os.makedirs(HUGGINGFACE_CACHE, exist_ok=True)
        os.environ['HF_HOME'] = HUGGINGFACE_CACHE
        cache_dir = HUGGINGFACE_CACHE

    logger.info(f"Creating pipeline from {args.base_model}")
    pipeline = create_mvd_pipeline(
        pretrained_model_name_or_path=args.base_model,
        use_memory_efficient_attention=True,
        enable_gradient_checkpointing=False,
        dtype=torch.float32,
        use_camera_conditioning=True,
        use_image_conditioning=True,
        cache_dir=cache_dir,
    )
    
    logger.info(f"Loading checkpoint from {args.checkpoint}")
    state_dict = torch.load(args.checkpoint, map_location="cpu")["state_dict"]    
    logger.info(f"Successfully loaded checkpoint with {len(state_dict)} keys")
    
    unet_keys = [k for k in state_dict.keys() if k.startswith("unet.")]
    unet_state_dict = {k.replace("unet.", ""): v for k, v in state_dict.items() if k in unet_keys}
    pipeline.unet.load_state_dict(unet_state_dict, strict=False)
    
    logger.info("Moving pipeline to GPU")
    pipeline = pipeline.to(device)
    
    source_image = load_image(args.source_image, target_size=(args.image_size, args.image_size))
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
    
    with torch.no_grad():
        output = pipeline(
            prompt=args.prompt,
            num_inference_steps=args.steps,
            source_camera=source_camera,
            target_camera=target_camera,
            source_images=source_image,
            guidance_scale=args.guidance_scale,
            ref_scale=args.ref_scale,
            output_type="pt"
        )
    
    images = output["images"]
    logger.info(f"Output tensor shape: {images.shape}")
    logger.info(f"Output tensor stats - min: {images.min().item()}, max: {images.max().item()}, mean: {images.mean().item()}")
    logger.info(f"Contains NaN: {torch.isnan(images).any().item()}")
    logger.info(f"Contains Inf: {torch.isinf(images).any().item()}")
    
    images = (images + 1.0) / 2.0
    images = images.clamp(0, 1)
    
    images = (images * 255).round().cpu().numpy().astype("uint8")
    images = images.squeeze(0).transpose(1, 2, 0)
    output_image = Image.fromarray(images)
    
    source_img_pil = ((source_image[0].cpu().permute(1, 2, 0) + 1) / 2 * 255).numpy().astype('uint8')
    source_img_pil = Image.fromarray(source_img_pil)
    
    output_dir = dirs['samples'] / f"inference_{Path(args.source_image).stem}"
    output_dir.mkdir(exist_ok=True, parents=True)
    
    source_img_pil.save(output_dir / "source.png")
    output_image.save(output_dir / "generated.png")
    
    combined = Image.new('RGB', (source_img_pil.width + output_image.width, max(source_img_pil.height, output_image.height)))
    combined.paste(source_img_pil, (0, 0))
    combined.paste(output_image, (source_img_pil.width, 0))
    combined.save(output_dir / "comparison.png")
    
    logger.info(f"Results saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MVD Inference")
    parser.add_argument(
        "--base-model", 
        type=str, 
        default="stabilityai/stable-diffusion-2-1", 
        help="Base model path"
    )
    parser.add_argument(
        "--checkpoint", 
        type=str, 
        required=True, 
        help="Path to trained checkpoint"
    )
    parser.add_argument(
        "--source-image", 
        type=str, 
        default="imgs/snail.png",
        help="Path to source image"
    )
    parser.add_argument(
        "--prompt", 
        type=str, 
        default="create a detailed image of a snail", 
        help="Text prompt"
    )
    parser.add_argument(
        "--steps", 
        type=int, 
        default=20, 
        help="Number of inference steps"
    )
    parser.add_argument(
        "--guidance-scale", 
        type=float, 
        default=1.0, 
        help="Guidance scale"
    )
    parser.add_argument(
        "--ref-scale", 
        type=float, 
        default=0.1, 
        help="Reference scale"
    )
    parser.add_argument(
        "--image-size", 
        type=int, 
        default=768, 
        help="Image size"
    )
    
    args = parser.parse_args()
    main(args)