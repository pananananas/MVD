from src.training.scheduler import compute_snr
from torchvision.transforms import Normalize
from torchvision.models import VGG16_Weights
import torchvision.models as models
import torch.nn.functional as F
from icecream import ic
import torch


class PerceptualLoss:
    def __init__(self, device="cuda"):
        self.vgg = (
            models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
            .features[:29]
            .to(device)
            .eval()
        )
        self.normalize = Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        self.device = device

        for param in self.vgg.parameters():
            param.requires_grad = False

    def __call__(self, x, y):
        current_device = x.device
        if current_device != self.device:
            self.to(current_device)

        x = (x + 1) / 2
        y = (y + 1) / 2

        x = self.normalize(x)
        y = self.normalize(y)

        x_features = self.vgg(x)
        y_features = self.vgg(y)

        return F.mse_loss(x_features, y_features)

    def to(self, device):
        self.device = device
        self.vgg = self.vgg.to(device)
        return self


def compute_losses(
    noise_pred,
    noise,
    noisy_latents=None,
    timesteps=None,
    target_latents=None,
    vae=None,
    scheduler=None,
    base_scheduler=None,
    perceptual_loss_fn=None,
    ssim_loss_fn=None,
    config=None,
):
    # Calculate base MSE loss per element
    noise_loss_per_element = F.mse_loss(noise_pred, noise, reduction="none")

    # Check if SNR loss weighting should be applied
    loss_config = config.get("loss_config", {})
    # use_snr = loss_config.get("use_snr_loss", False)

    device = noise_loss_per_element.device
    zero_tensor = torch.tensor(0.0, device=device)
    one_tensor = torch.tensor(1.0, device=device)

    metrics = {
        "latent_recon_loss": zero_tensor,
        "pixel_recon_loss": zero_tensor,
        "perceptual_loss": zero_tensor,
        "ssim_loss": zero_tensor,
        "ssim_value": zero_tensor,
        "mean_snr": zero_tensor,
        "mean_snr_weight": one_tensor,
    }

    # if use_snr and base_scheduler is not None and timesteps is not None:

    snr = compute_snr(timesteps, base_scheduler)
    snr_gamma = loss_config.get("snr_gamma", 5.0)

    # Compute loss weights as per Min-SNR paper (Section 3.4)
    # Note: The paper uses SNR = SNR_t = alpha_t^2 / sigma_t^2
    # The weight is min(SNR_t, gamma) / SNR_t
    snr_clipped = torch.stack([snr, snr_gamma * torch.ones_like(snr)], dim=1).min(
        dim=1
    )[0]
    mse_loss_weights = snr_clipped / snr

    # Reshape weights to match the loss tensor (B, C, H, W) -> (B,) -> (B, 1, 1, 1)
    mse_loss_weights = mse_loss_weights.flatten()
    while len(mse_loss_weights.shape) < len(noise_loss_per_element.shape):
        mse_loss_weights = mse_loss_weights.unsqueeze(-1)

    # Apply weights element-wise
    weighted_loss = noise_loss_per_element * mse_loss_weights
    noise_loss = weighted_loss.mean()

    # Log mean SNR and weight
    metrics["mean_snr"] = snr.mean().detach()
    metrics["mean_snr_weight"] = mse_loss_weights.mean().detach()

    # else:
    #     # Default: standard MSE loss (mean over all elements)
    #     noise_loss = noise_loss_per_element.mean()

    total_loss = (
        noise_loss  # For now, total_loss is just the (potentially weighted) noise loss
    )

    if (
        noisy_latents is not None
        and timesteps is not None
        and target_latents is not None
        and vae is not None
        and scheduler is not None
    ):
        with torch.no_grad():
            try:
                # Get scheduler's alphas_cumprod for the given timesteps
                alphas_cumprod_t = scheduler.alphas_cumprod[timesteps].to(device)
                # Ensure alphas_cumprod_t has the same number of dimensions as latents for broadcasting
                while len(alphas_cumprod_t.shape) < len(noisy_latents.shape):
                    alphas_cumprod_t = alphas_cumprod_t.unsqueeze(-1)

                sqrt_alphas_cumprod_t = alphas_cumprod_t.sqrt()
                sqrt_one_minus_alphas_cumprod_t = (1.0 - alphas_cumprod_t).sqrt()

                if scheduler.config.prediction_type == "epsilon":
                    denoised_latents = (
                        noisy_latents - sqrt_one_minus_alphas_cumprod_t * noise_pred
                    ) / sqrt_alphas_cumprod_t
                elif scheduler.config.prediction_type == "v_prediction":
                    denoised_latents = (
                        sqrt_alphas_cumprod_t * noisy_latents
                        - sqrt_one_minus_alphas_cumprod_t * noise_pred
                    )
                elif scheduler.config.prediction_type == "sample":
                    denoised_latents = noise_pred
                else:
                    # Fallback or error for unknown prediction_type
                    ic(
                        f"Unknown prediction_type: {scheduler.config.prediction_type}. Using epsilon prediction formula for denoised_latents."
                    )
                    denoised_latents = (
                        noisy_latents - sqrt_one_minus_alphas_cumprod_t * noise_pred
                    ) / sqrt_alphas_cumprod_t

                metrics["latent_recon_loss"] = F.mse_loss(
                    denoised_latents.float(), target_latents.float()
                ).detach()

                denoised_images = vae.decode(
                    denoised_latents / vae.config.scaling_factor
                ).sample
                target_images = vae.decode(
                    target_latents / vae.config.scaling_factor
                ).sample

                metrics["pixel_recon_loss"] = F.mse_loss(
                    denoised_images.float(), target_images.float()
                ).detach()

                if perceptual_loss_fn is not None:
                    metrics["perceptual_loss"] = perceptual_loss_fn(
                        denoised_images.float(), target_images.float()
                    ).detach()

                if ssim_loss_fn is not None:
                    ssim_val = ssim_loss_fn(
                        denoised_images.float(), target_images.float()
                    ).detach()
                    metrics["ssim_value"] = ssim_val
                    metrics["ssim_loss"] = 1.0 - ssim_val

            except Exception as e:
                ic(f"Exception during auxiliary metric computation: {e}")

        if torch.isnan(noise_pred).any() or torch.isinf(noise_pred).any():
            ic("NaN/Inf detected in noise_pred!")

        # if noisy_latents is not None:
        #     ic(f"Compute Losses Stats:")
        #     ic(f"  Input:  {noise_pred}, {noise}")
        #     ic(f"  Latents: {target_latents}, {noisy_latents}")

    return {"total_loss": total_loss, "noise_loss": noise_loss, **metrics}
