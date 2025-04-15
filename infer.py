from src.models.mvd_unet import create_mvd_pipeline
from src.utils import create_output_dirs
from pathlib import Path
from PIL import Image
import numpy as np
import argparse
import logging
import torch
import PIL

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_image(image_path, target_size=(768, 768)):
    image = Image.open(image_path).convert("RGB")
    image = image.resize(target_size, PIL.Image.Resampling.LANCZOS)
    image_np = np.array(image).astype(np.float32) / 127.5 - 1.0
    image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0)
    return image_tensor

def create_camera_matrix(position, target, up=None):
    if up is None:
        up = np.array([0.0, 1.0, 0.0])
    
    position = np.array(position)
    target = np.array(target)
    up = np.array(up)
    
    forward = target - position
    forward = forward / np.linalg.norm(forward)
    
    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)
    
    new_up = np.cross(right, forward)
    
    rotation = np.zeros((3, 3))
    rotation[:, 0] = right
    rotation[:, 1] = new_up
    rotation[:, 2] = -forward
    
    camera_matrix = np.zeros((3, 4))
    camera_matrix[:3, :3] = rotation
    camera_matrix[:3, 3]  = position
    
    return torch.from_numpy(camera_matrix).float()


def main(args):
    dirs = create_output_dirs("inference_outputs")
    
    logger.info(f"Creating pipeline from {args.base_model}")
    pipeline = create_mvd_pipeline(
        pretrained_model_name_or_path=args.base_model,
        use_memory_efficient_attention=True,
        enable_gradient_checkpointing=False,
        dtype=torch.float32,
        use_camera_embeddings=True,
        use_image_conditioning=True,
    )
    
    logger.info(f"Loading checkpoint from {args.checkpoint}")
    state_dict = torch.load(args.checkpoint, map_location="cpu")["state_dict"]
    
    unet_keys = [k for k in state_dict.keys() if k.startswith("unet.")]
    unet_state_dict = {k.replace("unet.", ""): v for k, v in state_dict.items() if k in unet_keys}
    pipeline.unet.load_state_dict(unet_state_dict, strict=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipeline = pipeline.to(device)
    
    source_image = load_image(args.source_image, target_size=(args.image_size, args.image_size))
    source_image = source_image.to(device)
    
    source_pos = [0, 0, 2.0]  # Front view
    target_pos = [0, 0, 0]    # Looking at origin
    source_camera = create_camera_matrix(source_pos, target_pos)
    
    # For target camera, we create a different viewpoint (e.g., 45° to the side)
    # This matches the relative camera transform used in training
    target_pos = [1.5, 0, 1.5]  # Side view at 45°
    target_camera = create_camera_matrix(target_pos, [0, 0, 0])
    
    # Expand dimensions for batch
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
            output_type="pil"
        )
    
    # Save results
    source_img_pil = ((source_image[0].cpu().permute(1, 2, 0) + 1) / 2 * 255).numpy().astype('uint8')
    source_img_pil = Image.fromarray(source_img_pil)
    
    output_dir = dirs['samples'] / f"inference_{Path(args.source_image).stem}"
    output_dir.mkdir(exist_ok=True, parents=True)
    
    source_img_pil.save(output_dir / "source.png")
    output["images"][0].save(output_dir / "generated.png")
    
    # Save side by side comparison
    combined = Image.new('RGB', (source_img_pil.width + output["images"][0].width, max(source_img_pil.height, output["images"][0].height)))
    combined.paste(source_img_pil, (0, 0))
    combined.paste(output["images"][0], (source_img_pil.width, 0))
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