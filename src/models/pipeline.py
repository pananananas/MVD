from typing import Any, Callable, Dict, List, Optional, Union
from diffusers import StableDiffusionPipeline
import torch

class MVDPipeline(StableDiffusionPipeline):
    """
    Custom pipeline for Multi-View Diffusion models.
    
    This pipeline extends StableDiffusionPipeline to properly handle camera parameters
    and pass them to the UNet during the diffusion process.
    """
    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        source_camera: Optional[torch.Tensor] = None,  # added camera parameters
        target_camera: Optional[torch.Tensor] = None,  # added camera parameters
        source_images: Optional[torch.Tensor] = None,  # added source images
    ):
        # Process prompts and embeddings as in the original pipeline
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]
            
        if prompt_embeds is None:
            # Process prompts
            prompt = prompt if prompt is not None else ""
            text_inputs = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            prompt_embeds = self.text_encoder(text_input_ids.to(self.device))[0]
            
        # Process negative prompt if needed
        if negative_prompt_embeds is None and negative_prompt is not None:
            uncond_input = self.tokenizer(
                negative_prompt,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]
        elif negative_prompt_embeds is not None:
            uncond_embeddings = negative_prompt_embeds
        else:
            uncond_embeddings = None
            
        # Create classifier-free guidance embeddings
        if guidance_scale > 1.0 and uncond_embeddings is not None:
            prompt_embeds = torch.cat([uncond_embeddings, prompt_embeds])
            
        # Prepare latents
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor
        
        if latents is None:
            # Generate random noise
            latents = self.prepare_latents(
                batch_size * num_images_per_prompt,
                4,  # 4 channels for latent space
                height,
                width,
                prompt_embeds.dtype,
                self.device,
                generator,
            )
        
        # Process source images if provided
        source_image_latents = None
        if source_images is not None:
            # Move source images to the correct device
            source_images = source_images.to(device=self.device)
            
            # Scale to [-1, 1] if in [0, 1] range
            if source_images.min() >= 0 and source_images.max() <= 1:
                source_images = 2 * source_images - 1
                
            # Encode source images to latent space
            with torch.no_grad():
                source_image_latents = self.vae.encode(source_images).latent_dist.sample()
                source_image_latents = source_image_latents * self.vae.config.scaling_factor
                print(f"Encoded source image to latent space: {source_image_latents.shape}")
        
        # Set timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=self.device)
        timesteps = self.scheduler.timesteps
        
        # Prepare extra kwargs for the UNet
        extra_kwargs = {}
        if source_camera is not None:
            extra_kwargs["source_camera"] = source_camera.to(self.device)
        if target_camera is not None:
            extra_kwargs["target_camera"] = target_camera.to(self.device)
        if source_image_latents is not None:
            extra_kwargs["source_image_latents"] = source_image_latents
        
        # Diffusion process
        for i, t in enumerate(self.progress_bar(timesteps)):
            # Expand latents for classifier-free guidance
            latent_model_input = torch.cat([latents] * 2) if guidance_scale > 1.0 else latents
            
            # Predict noise
            noise_pred = self.unet(
                sample=latent_model_input,
                timestep=t,
                encoder_hidden_states=prompt_embeds,
                cross_attention_kwargs=cross_attention_kwargs,
                **extra_kwargs  # Pass camera parameters and source image latents to UNet
            ).sample
            
            # Apply classifier-free guidance
            if guidance_scale > 1.0:
                noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
            
            # Compute previous noisy sample
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample
            
            # Call callback if provided
            if callback is not None and i % callback_steps == 0:
                callback(i, t, latents)
        
        # Decode latents
        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents).sample
        
        image = (image / 2 + 0.5).clamp(0, 1)
        
        # Convert to output format
        if output_type == "pil":
            image = image.cpu().permute(0, 2, 3, 1).float().numpy()
            image = self.numpy_to_pil(image)
        
        if not return_dict:
            return image
            
        return {"images": image}