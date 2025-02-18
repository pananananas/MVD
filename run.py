from src.load_co3d import analyze_dataset, CO3DDatasetLoader
from diffusers import StableDiffusionPipeline
from src.dataset import create_dataloaders, get_image_path
from src.vis_co3d import CO3DVisualizer
from src.mvd import create_mvd_pipeline
import torch.multiprocessing as mp
import torch.nn.functional as F
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
from icecream import ic
from tqdm import tqdm
from PIL import Image
import numpy as np
import random
import wandb
import torch
import os
# data_path = "/Users/ewojcik/Code/datasets/co3d/laptop"
# model_id = wandb.config.architecture
# sequence_id = "62_4317_10724"

# analyze_dataset(data_path)

# loader = CO3DDatasetLoader(data_path)
# visualizer = CO3DVisualizer(loader)


# ic(loader.frame_annotations[0].keys())
# ic(loader.sequence_annotations[0].keys())

# visualizer.visualize_cameras_and_pointcloud(sequence_id)

SCRATCH           = os.getenv('SCRATCH', '/net/tscratch/people/plgewoj')
HUGGINGFACE_CACHE = os.path.join(SCRATCH, 'huggingface_cache')
os.makedirs(HUGGINGFACE_CACHE, exist_ok=True)
os.environ['HF_HOME'] = HUGGINGFACE_CACHE

# dataset_path = "/Users/ewojcik/Code/datasets/co3d/laptop"
dataset_path = "/net/pr2/projects/plgrid/plggtattooai/MeshDatasets/co3d/motorcycle"


def train_step(batch, pipeline, optimizer, device, batch_idx=0, accumulation_steps=4, prompt="a photo of a laptop"):
    # Scale loss by accumulation steps
    loss = train_step_inner(batch, pipeline, device, prompt) / accumulation_steps
    loss.backward()
    
    # Only update weights after accumulating gradients
    if (batch_idx + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
    
    return loss.item()

def train_step_inner(batch, pipeline, device, prompt):
    # Move all batch tensors to the device
    source_images = batch['source_image'].to(device)  # [B, 3, H, W]
    target_images = batch['target_image'].to(device)
    source_camera = {k: v.to(device) for k, v in batch['source_camera'].items()}
    target_camera = {k: v.to(device) for k, v in batch['target_camera'].items()}
    points_3d = [points.to(device) for points in batch['points_3d']]
    
    # Print shapes for debugging
    # ic(source_images.shape[0])
    
    # Tokenize and encode the prompt
    prompts = [prompt] * source_images.shape[0]
    
    # Tokenize with proper batch size
    text_input = pipeline.tokenizer(
        prompts,
        padding="max_length",
        max_length=pipeline.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt"
    ).to(device)
    
    # Encode prompt
    text_embeddings = pipeline.text_encoder(text_input.input_ids)[0]
    # ic(text_embeddings.shape)
    
    # Forward pass
    noise = torch.randn_like(source_images)
    timesteps = torch.randint(0, pipeline.scheduler.config.num_train_timesteps, 
                            (source_images.shape[0],), device=device)
    noisy_images = pipeline.scheduler.add_noise(source_images, noise, timesteps)
    
    # Add extra channel for noise prediction (alpha channel)
    noisy_images = torch.cat([noisy_images, torch.zeros_like(noisy_images[:, :1])], dim=1)  # [B, 4, H, W]
    
    # Predict noise
    noise_pred = pipeline.unet(
        sample=noisy_images,
        timestep=timesteps,
        encoder_hidden_states=text_embeddings,
        source_camera=source_camera,
        target_camera=target_camera,
        points_3d=points_3d
    ).sample
    
    # Multiple loss components
    noise_loss = F.mse_loss(noise_pred[:, :3], noise)
    
    # Denoise the image (simplified version)
    alpha_t = pipeline.scheduler.alphas_cumprod[timesteps]
    alpha_t = alpha_t.view(-1, 1, 1, 1)
    
    denoised_images = (noisy_images - (1 - alpha_t).sqrt() * noise_pred) / alpha_t.sqrt()
    
    # Image reconstruction loss
    recon_loss = F.l1_loss(denoised_images[:, :3], target_images)
    
    # Combine losses
    total_loss = noise_loss + recon_loss
    
    # Log individual losses
    wandb.log({
        "noise_loss": noise_loss.item(),
        "recon_loss": recon_loss.item(),
        "total_loss": total_loss.item()
    })
    
    return total_loss

def visualize_batch(batch, generated_images, output_dir, batch_idx):
    """
    Creates a visualization grid for a batch of images.
    Args:
        batch: Dictionary containing source_image and target_image
        generated_images: List of generated PIL images
        output_dir: Directory to save the visualization
        batch_idx: Current batch index
    """
    batch_size = len(batch['source_image'])
    fig, axes = plt.subplots(batch_size, 3, figsize=(15, 5*batch_size))
    
    # If batch_size is 1, axes will not be 2D, so we need to handle this case
    if batch_size == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(batch_size):
        # Helper function to denormalize and convert to numpy
        def denorm(x):
            # x is in [-1, 1], convert to [0, 1]
            x = (x + 1) / 2
            x = x.clamp(0, 1)
            return x.cpu().permute(1, 2, 0).numpy()
        
        # Plot source image
        source = denorm(batch['source_image'][i])
        axes[i, 0].imshow(source)
        axes[i, 0].set_title('Source')
        axes[i, 0].axis('off')
        
        # Plot target image
        target = denorm(batch['target_image'][i])
        axes[i, 1].imshow(target)
        axes[i, 1].set_title('Target')
        axes[i, 1].axis('off')
        
        # Plot generated image
        generated = np.array(generated_images[i])
        axes[i, 2].imshow(generated)
        axes[i, 2].set_title('Generated')
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'comparison_batch_{batch_idx}.png'))
    plt.close()

def save_generated_images(pipeline, batch, output_dir, batch_idx, device, prompt):
    # Create output directory if it doesn't exist
    output_dir = Path(output_dir) / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        # Generate images
        batch_size = len(batch['source_image'])
        images = pipeline(
            prompt=[prompt] * batch_size,  # Ensure we have one prompt per image
            num_inference_steps=20,  # Lower for faster generation during training
            source_camera=batch['source_camera'],
            target_camera=batch['target_camera'],
            points_3d=batch['points_3d'],
            num_images_per_prompt=1,
            output_type="np"  # Get numpy arrays directly
        ).images
        
        # Create visualization
        visualize_batch(batch, images, output_dir, batch_idx)
        
        # Save individual images
        for i, (source, target, generated) in enumerate(zip(batch['source_image'], batch['target_image'], images)):
            # Convert tensors to PIL images (properly denormalized)
            source_np = ((source.cpu().permute(1, 2, 0) + 1) / 2 * 255).numpy().astype('uint8')
            target_np = ((target.cpu().permute(1, 2, 0) + 1) / 2 * 255).numpy().astype('uint8')
            
            # Save images
            Image.fromarray(source_np).save(output_dir / f'batch_{batch_idx}_sample_{i}_source.png')
            Image.fromarray(target_np).save(output_dir / f'batch_{batch_idx}_sample_{i}_target.png')
            Image.fromarray((generated * 255).astype('uint8')).save(output_dir / f'batch_{batch_idx}_sample_{i}_generated.png')

if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    ic(device)
    

    wandb.init(
        project="mvd",
        config={
            # Model parameters
            "architecture": "runwayml/stable-diffusion-v1-5",
            "torch_dtype": "float32",
            "use_memory_efficient_attention": True,
            "enable_gradient_checkpointing": True,
            "gradient_accumulation_steps": 4,
            
            # Training parameters
            "learning_rate": 0.0002,
            "epochs": 10,
            "batch_size": 2,
            
            # Dataset parameters
            "dataset": "CO3D-motor",
            "image_size": (256, 256),
            "max_angle_diff": 45.0,
            "min_angle_diff": 15.0,
            "max_pairs_per_sequence": 10,
            
            # Loss parameters
            "loss_fn": "L2",
            
            # Prompt parameters
            "prompt": "a photo of a motorcycle",

            # Debug parameters
            "debug": True,
            "debug_num_sequences": 2,
            "debug_max_pairs": 5,
        }
    )

    # Create dataloaders with debug mode support
    train_loader, val_loader = create_dataloaders(
        data_path=dataset_path,
        batch_size=wandb.config.batch_size,
        image_size=wandb.config.image_size,
        max_angle_diff=wandb.config.max_angle_diff,
        min_angle_diff=wandb.config.min_angle_diff,
        max_pairs_per_sequence=wandb.config.max_pairs_per_sequence if not wandb.config.debug else wandb.config.debug_max_pairs,
        debug_mode=wandb.config.debug,
        debug_num_sequences=wandb.config.debug_num_sequences
    )

    pipeline = create_mvd_pipeline(
        pretrained_model_name_or_path=wandb.config.architecture,
        dtype=torch.float32,                                        # TODO: use the dtype from the config
        use_memory_efficient_attention=wandb.config.use_memory_efficient_attention,
        enable_gradient_checkpointing=wandb.config.enable_gradient_checkpointing,
        cache_dir=HUGGINGFACE_CACHE,
    )
    pipeline.to(device)
    optimizer = torch.optim.Adam(pipeline.unet.parameters(), lr=wandb.config.learning_rate)

    output_dir = "outputs"
    
    # Training loop
    for batch_idx, batch in enumerate(tqdm(train_loader, desc="Training")):
        # ic(batch.keys())

        # ic(batch['source_image'].shape)
        # ic(batch['target_image'].shape)
        # ic(batch['source_camera'])
        # ic(batch['target_camera'])

        # ic(batch['source_camera']['R'].shape)
        # ic(batch['source_camera']['T'].shape)
        # ic(batch['target_camera']['R'].shape)
        # ic(batch['target_camera']['T'].shape)

        # ic([points.shape for points in batch['points_3d']])
        # break
        loss = train_step(
            batch, 
            pipeline, 
            optimizer, 
            device,
            batch_idx=batch_idx,
            accumulation_steps=wandb.config.gradient_accumulation_steps,
            prompt=wandb.config.prompt
        )
        # ic(loss)
        
        # Save generated images every N batches
        if batch_idx % 10 == 0:
            save_generated_images(
                pipeline, 
                batch, 
                output_dir, 
                batch_idx, 
                device, 
                wandb.config.prompt
            )
        
        # if batch_idx >= 100:  # Early stop for testing
        #     break
    
    wandb.finish()

# pipe = StableDiffusionPipeline.from_pretrained(model_id)
# unet = pipe.unet
# ic(unet)
 



# simulate training
# epochs = wandb.config.epochs
# offset = random.random() / 5
# for epoch in range(2, epochs):
#     acc = 1 - 2 ** -epoch - random.random() / epoch - offset
#     loss = 2 ** -epoch + random.random() / epoch + offset

#     # log metrics to wandb
#     wandb.log({"acc": acc, "loss": loss})

# # [optional] finish the wandb run, necessary in notebooks
# wandb.finish()