from typing import Any, Callable, Dict, List, Optional, Union
from diffusers import StableDiffusionPipeline
from icecream import ic
import logging
import torch

logger = logging.getLogger(__name__)

class MVDPipeline(StableDiffusionPipeline):
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
        source_camera: Optional[torch.Tensor] = None,
        target_camera: Optional[torch.Tensor] = None,
        source_images: Optional[torch.Tensor] = None,
        ref_scale: float = None,
    ):
        
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]
            
        if prompt_embeds is None:
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
            
        if guidance_scale > 1.0 and uncond_embeddings is not None:
            logger.info(f"Classifier-free guidance enabled with scale {guidance_scale}")
            logger.info(f"Uncond embeddings shape: {uncond_embeddings.shape}")
            logger.info(f"Prompt embeddings shape: {prompt_embeds.shape}")
            prompt_embeds = torch.cat([uncond_embeddings, prompt_embeds])
            logger.info(f"Combined embeddings shape: {prompt_embeds.shape}")
            
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor
        ic(f"height: {height}, width: {width}")
        
        if latents is None:
            latents = self.prepare_latents(
                batch_size * num_images_per_prompt,
                4,
                height,
                width,
                prompt_embeds.dtype,
                self.device,
                generator,
            )
        
        source_image_latents = None
        if source_images is not None:
            source_images = source_images.to(device=self.device)
            
            # scale to [-1, 1] if in [0, 1] range
            if source_images.min() >= 0 and source_images.max() <= 1:
                source_images = 2 * source_images - 1
                
            with torch.no_grad():
                if source_images.shape[0] < batch_size:
                    repeats = batch_size // source_images.shape[0]
                    source_images = source_images.repeat(repeats, 1, 1, 1)
                
                source_image_latents = self.vae.encode(source_images).latent_dist.sample()
                source_image_latents = source_image_latents * self.vae.config.scaling_factor
        
        self.scheduler.set_timesteps(num_inference_steps, device=self.device)
        timesteps = self.scheduler.timesteps
        
        extra_kwargs = {}
        if source_camera is not None:
            logger.info(f"Source camera shape before device move: {source_camera.shape}")
            extra_kwargs["source_camera"] = source_camera.to(self.device)
        if target_camera is not None:
            logger.info(f"Target camera shape before device move: {target_camera.shape}")
            extra_kwargs["target_camera"] = target_camera.to(self.device)
        if source_image_latents is not None:
            logger.info(f"Source image latents shape: {source_image_latents.shape}")
            extra_kwargs["source_image_latents"] = source_image_latents
        
        if ref_scale is None:
            ref_scale = getattr(self, 'img_ref_scale', 0.3)
        
        cross_attention_kwargs = cross_attention_kwargs or {}

        for i, t in enumerate(self.progress_bar(timesteps)):
            latent_model_input = torch.cat([latents] * 2) if guidance_scale > 1.0 else latents
            
            noise_pred = self.unet(
                sample=latent_model_input,
                timestep=t,
                encoder_hidden_states=prompt_embeds,
                cross_attention_kwargs=cross_attention_kwargs,
                use_camera_embeddings=getattr(self, 'use_camera_embeddings', True),
                use_image_conditioning=getattr(self, 'use_image_conditioning', True),
                **extra_kwargs
            ).sample
            
            if guidance_scale > 1.0:
                noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
            
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample
            
            if callback is not None and i % callback_steps == 0:
                callback(i, t, latents)
        
        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents).sample
        
        image = (image / 2 + 0.5).clamp(0, 1)
        
        if output_type == "pil":
            image = image.cpu().permute(0, 2, 3, 1).float().numpy()
            image = self.numpy_to_pil(image)
        
        if not return_dict:
            return image
            
        return {"images": image}