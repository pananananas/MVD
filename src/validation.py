import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import os
import logging
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
from PIL import Image
from torchvision.utils import make_grid, save_image

logger = logging.getLogger(__name__)

class CrossAttentionValidator:
    """
    Validator for the image cross-attention implementation.
    Performs ablation studies to measure the impact of cross-attention on the model's performance.
    """
    
    def __init__(self, model, output_dir="validation_results"):
        """
        Initialize the validator.
        
        Args:
            model: The model with image cross-attention to validate
            output_dir: Directory to save validation results and visualizations
        """
        self.model = model
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Find target model based on input type
        # Could be MVDLightningModule, pipeline, or direct UNet
        if hasattr(model, 'unet'):
            if hasattr(model, 'pipeline') and hasattr(model.pipeline, 'unet'):
                # Handle MVDLightningModule case
                self.unet = model.pipeline.unet
                logger.info("Using UNet from MVDLightningModule")
            else:
                # Handle Pipeline case
                self.unet = model.unet
                logger.info("Using UNet from pipeline")
        else:
            # Direct UNet case
            self.unet = model
            logger.info("Using UNet directly")
        
        # Find all cross-attention parameters
        self.ref_params = {}
        self.ref_scale_params = {}
        
        # Store the names and parameter values
        for name, param in self.unet.named_parameters():
            if 'ref' in name:
                self.ref_params[name] = param
                if 'ref_scale' in name:
                    self.ref_scale_params[name] = param
        
        logger.info(f"Found {len(self.ref_scale_params)} cross-attention scale parameters")
        
        # Store original scale values to restore later
        self.original_scales = {}
        with torch.no_grad():
            for name, param in self.ref_scale_params.items():
                self.original_scales[name] = param.clone()
    
    def set_all_scales(self, value):
        """
        Set all reference scales to a specific value.
        
        Args:
            value: Value to set for all scale parameters
        """
        with torch.no_grad():
            for name, param in self.ref_scale_params.items():
                param.fill_(value)
        logger.info(f"Set all {len(self.ref_scale_params)} scales to {value}")
    
    def restore_original_scales(self):
        """Restore original scale values."""
        with torch.no_grad():
            for name, original_value in self.original_scales.items():
                self.ref_scale_params[name].copy_(original_value)
        logger.info("Restored original scale values")
    
    def evaluate_batch(self, batch, metrics=None):
        """
        Run model evaluation on a batch and return metrics.
        
        Args:
            batch: Input batch to evaluate
            metrics: List of metrics to calculate (defaults to ['loss'])
            
        Returns:
            Dictionary of metrics
        """
        if metrics is None:
            metrics = ['loss']
            
        results = {}
        
        with torch.no_grad():
            # Different handling based on model type
            if hasattr(self.model, 'forward') and 'MVDLightningModule' in self.model.__class__.__name__:
                # Use the Lightning module's forward method
                noise_pred, noise, denoised_latents, target_latents, source_latents = self.model.forward(batch)
                
                # Calculate loss using model's loss function if available
                if hasattr(self.model, 'compute_loss') and 'loss' in metrics:
                    loss_dict = self.model.compute_loss(noise_pred, noise, denoised_latents, target_latents)
                    for k, v in loss_dict.items():
                        results[k] = v.item()
                else:
                    # Basic MSE loss on noise prediction
                    if 'loss' in metrics:
                        loss = F.mse_loss(noise_pred, noise)
                        results['loss'] = loss.item()
            elif hasattr(self.model, 'unet'):
                # Handle pipeline case - extract required inputs from batch
                # This matches the pattern in MVDLightningModule.forward
                
                # Prepare inputs for the UNet
                latents = batch['latents'] if 'latents' in batch else batch.get('noisy_latents')
                timesteps = batch['timesteps'] if 'timesteps' in batch else batch.get('timestep')
                encoder_hidden_states = batch['prompt_embeds'] if 'prompt_embeds' in batch else batch.get('text_embeddings')
                
                # Optional parameters
                source_camera = batch.get('source_camera')
                target_camera = batch.get('target_camera')
                source_image_latents = batch.get('source_latents')
                
                # Get the noise prediction
                noise_pred = self.unet(
                    sample=latents,
                    timestep=timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    source_camera=source_camera,
                    target_camera=target_camera,
                    source_image_latents=source_image_latents,
                    return_dict=True
                ).sample
                
                # Calculate loss
                if 'loss' in metrics:
                    target_noise = batch.get('noise', batch.get('target_noise'))
                    if target_noise is not None:
                        loss = F.mse_loss(noise_pred, target_noise)
                        results['loss'] = loss.item()
            else:
                # Direct UNet case
                # This is a fallback and may need additional adaptation
                sample = batch.get('sample', batch.get('latents', batch.get('noisy_latents')))
                timestep = batch.get('timestep', batch.get('timesteps'))
                encoder_hidden_states = batch.get('encoder_hidden_states', batch.get('text_embeddings', batch.get('prompt_embeds')))
                
                # Run the model
                output = self.unet(
                    sample=sample,
                    timestep=timestep,
                    encoder_hidden_states=encoder_hidden_states,
                    **{k: v for k, v in batch.items() if k not in ['sample', 'timestep', 'encoder_hidden_states']}
                )
                
                # Calculate loss
                if 'loss' in metrics:
                    if hasattr(output, 'sample'):
                        pred = output.sample
                    else:
                        pred = output
                    
                    # Find target
                    target = batch.get('target_noise', batch.get('noise', None))
                    if target is not None:
                        loss = F.mse_loss(pred, target)
                        results['loss'] = loss.item()
        
        return results
    
    def run_ablation_study(self, dataloader, scales=None, metrics=None, save_images=True):
        """
        Run ablation study with different scale values.
        
        Args:
            dataloader: DataLoader with validation data
            scales: List of scale values to test (default: [0.0, 0.1, 0.5, 1.0, 2.0])
            metrics: Metrics to calculate
            save_images: Whether to save generated images
            
        Returns:
            Dictionary of results for each scale value
        """
        if scales is None:
            scales = [0.0, 0.1, 0.5, 1.0, 2.0]
        
        if metrics is None:
            metrics = ['loss']
        
        results = {scale: {} for scale in scales}
        
        # Create directories for saving images
        if save_images:
            for scale in scales:
                scale_dir = self.output_dir / f"scale_{scale}"
                scale_dir.mkdir(exist_ok=True)
        
        logger.info(f"Running ablation study with scales: {scales}")
        
        # Run evaluation for each scale
        for scale in scales:
            self.set_all_scales(scale)
            logger.info(f"Testing scale = {scale}")
            
            batch_results = []
            for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Scale {scale}")):
                if batch_idx >= 10:  # Limit batches for efficiency
                    break
                
                # Move batch to the same device as model
                batch = self._prepare_batch(batch)
                
                # Evaluate metrics
                metrics_dict = self.evaluate_batch(batch, metrics)
                batch_results.append(metrics_dict)
                
                # Save generated images
                if save_images and batch_idx < 5:  # Save only a few for visualization
                    self._save_batch_results(batch, scale, batch_idx)
            
            # Average results across batches
            if batch_results:
                for k in batch_results[0].keys():
                    results[scale][k] = sum(d.get(k, 0) for d in batch_results) / len(batch_results)
                logger.info(f"Scale {scale} results: {results[scale]}")
        
        # Generate and save plots
        self._plot_ablation_results(results)
        
        # Restore original scales
        self.restore_original_scales()
        
        return results
    
    def per_layer_analysis(self, dataloader, metrics=None, num_batches=5):
        """
        Analyze contribution of each layer's cross-attention.
        
        Args:
            dataloader: DataLoader with validation data
            metrics: Metrics to calculate
            num_batches: Number of batches to evaluate
            
        Returns:
            Dictionary with per-layer results
        """
        if metrics is None:
            metrics = ['loss']
        
        # First disable all cross-attention
        self.set_all_scales(0.0)
        
        # Get representative batches for testing
        test_batches = []
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= num_batches:
                break
            test_batches.append(self._prepare_batch(batch))
        
        if not test_batches:
            logger.error("No batches available for evaluation")
            return {}
        
        # Baseline with no cross-attention
        baseline_results = []
        for batch in test_batches:
            baseline_results.append(self.evaluate_batch(batch, metrics))
        
        # Average baseline results
        baseline_metrics = {}
        for k in baseline_results[0].keys():
            baseline_metrics[k] = sum(d.get(k, 0) for d in baseline_results) / len(baseline_results)
        
        logger.info(f"Baseline (no cross-attention): {baseline_metrics}")
        
        # Test each layer individually
        layer_results = {}
        for layer_name, param in tqdm(self.ref_scale_params.items(), desc="Analyzing layers"):
            # Reset all scales to 0
            self.set_all_scales(0.0)
            
            # Enable only this layer
            with torch.no_grad():
                param.fill_(1.0)
            
            # Evaluate on all test batches
            layer_batch_results = []
            for batch in test_batches:
                layer_batch_results.append(self.evaluate_batch(batch, metrics))
            
            # Average results for this layer
            layer_metrics = {}
            for k in layer_batch_results[0].keys():
                layer_metrics[k] = sum(d.get(k, 0) for d in layer_batch_results) / len(layer_batch_results)
            
            # Calculate improvement
            improvements = {}
            for k in baseline_metrics.keys():
                if k in layer_metrics:
                    # For loss metrics, improvement is decrease
                    if 'loss' in k.lower():
                        improvements[k] = baseline_metrics[k] - layer_metrics[k]
                    else:
                        # For other metrics like PSNR, improvement is increase
                        improvements[k] = layer_metrics[k] - baseline_metrics[k]
            
            layer_results[layer_name] = {
                'metrics': layer_metrics,
                'improvements': improvements
            }
            
            logger.info(f"Layer {layer_name}: {layer_metrics}, Improvements: {improvements}")
        
        # Generate and save layer impact visualizations
        self._plot_layer_impact(layer_results, baseline_metrics)
        
        # Restore original scales
        self.restore_original_scales()
        
        return layer_results
    
    def _prepare_batch(self, batch):
        """
        Prepare batch for processing (move to correct device).
        
        Args:
            batch: Input batch
            
        Returns:
            Batch on the correct device
        """
        device = next(self.unet.parameters()).device
        
        # Move all tensors in batch to the correct device
        if isinstance(batch, dict):
            return {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                   for k, v in batch.items()}
        elif isinstance(batch, list):
            return [x.to(device) if isinstance(x, torch.Tensor) else x for x in batch]
        elif isinstance(batch, tuple) and hasattr(batch, '_fields'):  # namedtuple
            return type(batch)(*(v.to(device) if isinstance(v, torch.Tensor) else v for v in batch))
        elif isinstance(batch, torch.Tensor):
            return batch.to(device)
        else:
            return batch
    
    def _save_batch_results(self, batch, scale, batch_idx):
        """
        Save visualization of model outputs for a batch.
        
        Args:
            batch: Input batch
            scale: Current scale value
            batch_idx: Batch index
        """
        # Create directory for this scale if it doesn't exist
        scale_dir = self.output_dir / f"scale_{scale}"
        scale_dir.mkdir(exist_ok=True)
        
        with torch.no_grad():
            # Generate outputs based on model type
            if hasattr(self.model, 'forward') and 'MVDLightningModule' in self.model.__class__.__name__:
                noise_pred, noise, denoised_latents, target_latents, source_latents = self.model.forward(batch)
                
                # Save visualizations
                if source_latents is not None:
                    self._save_tensor_as_image(
                        source_latents,
                        scale_dir / f"batch_{batch_idx}_source.png"
                    )
                
                if target_latents is not None:
                    self._save_tensor_as_image(
                        target_latents,
                        scale_dir / f"batch_{batch_idx}_target.png"
                    )
                
                if denoised_latents is not None:
                    self._save_tensor_as_image(
                        denoised_latents,
                        scale_dir / f"batch_{batch_idx}_generated.png"
                    )
                
                if denoised_latents is not None and target_latents is not None:
                    self._save_comparison_image(
                        denoised_latents,
                        target_latents,
                        scale_dir / f"batch_{batch_idx}_comparison.png"
                    )
            else:
                # Extract images from the batch based on common keys
                # This is more generic and might need adjustment
                for key_pattern, name in [
                    (['source_latents', 'source_image_latents'], 'source'),
                    (['target_latents', 'target'], 'target')
                ]:
                    for key in key_pattern:
                        if key in batch and batch[key] is not None:
                            self._save_tensor_as_image(
                                batch[key], 
                                scale_dir / f"batch_{batch_idx}_{name}.png"
                            )
                            break
                
                # Try to generate an output image
                try:
                    # This is a minimal forward pass to get visualizable output
                    # It needs to be adapted to your specific model
                    if hasattr(self.model, 'unet'):
                        # Get required inputs
                        latents = batch.get('latents', batch.get('noisy_latents'))
                        timesteps = batch.get('timesteps', batch.get('timestep'))
                        encoder_hidden_states = batch.get('prompt_embeds', batch.get('text_embeddings'))
                        
                        if all(x is not None for x in [latents, timesteps, encoder_hidden_states]):
                            # Generate output
                            output = self.unet(
                                sample=latents,
                                timestep=timesteps,
                                encoder_hidden_states=encoder_hidden_states,
                                source_camera=batch.get('source_camera'),
                                target_camera=batch.get('target_camera'),
                                source_image_latents=batch.get('source_latents')
                            )
                            
                            if hasattr(output, 'sample'):
                                output = output.sample
                            
                            # Save output
                            self._save_tensor_as_image(
                                output,
                                scale_dir / f"batch_{batch_idx}_prediction.png"
                            )
                            
                            # Save comparison if target is available
                            target = batch.get('target_latents', batch.get('target'))
                            if target is not None:
                                self._save_comparison_image(
                                    output,
                                    target,
                                    scale_dir / f"batch_{batch_idx}_comparison.png"
                                )
                except Exception as e:
                    logger.warning(f"Failed to generate visualization: {str(e)}")
    
    def _save_tensor_as_image(self, tensor, path):
        """
        Save a tensor as an image.
        
        Args:
            tensor: Image tensor [B, C, H, W]
            path: Output path
        """
        # Ensure the tensor is on CPU
        tensor = tensor.detach().cpu()
        
        # Normalize if needed (e.g., for latent space)
        if tensor.min() < 0 or tensor.max() > 1:
            tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min())
        
        # Create a grid if batch size > 1
        if tensor.size(0) > 1:
            grid = make_grid(tensor, nrow=int(np.sqrt(tensor.size(0))))
            save_image(grid, path)
        else:
            save_image(tensor, path)
    
    def _save_comparison_image(self, generated, target, path):
        """
        Save a side-by-side comparison of generated and target images.
        
        Args:
            generated: Generated image tensor [B, C, H, W]
            target: Target image tensor [B, C, H, W]
            path: Output path
        """
        # Ensure tensors are on CPU
        generated = generated.detach().cpu()
        target = target.detach().cpu()
        
        # Normalize if needed
        if generated.min() < 0 or generated.max() > 1:
            generated = (generated - generated.min()) / (generated.max() - generated.min())
        
        if target.min() < 0 or target.max() > 1:
            target = (target - target.min()) / (target.max() - target.min())
        
        # Take only the first few images if batch is large
        batch_size = min(4, generated.size(0))
        generated = generated[:batch_size]
        target = target[:batch_size]
        
        # Concatenate along width dimension
        comparison = torch.cat([generated, target], dim=0)
        
        # Create a grid
        grid = make_grid(comparison, nrow=batch_size)
        save_image(grid, path)
        
        # Also save a difference visualization
        if generated.shape == target.shape:
            diff = torch.abs(generated - target)
            # Normalize difference to [0, 1]
            diff = diff / diff.max()
            
            # Add a heatmap colorization to the difference
            diff_colored = self._apply_heatmap_colorization(diff)
            diff_grid = make_grid(diff_colored, nrow=batch_size)
            
            # Save the difference visualization
            diff_path = str(path).replace('.png', '_diff.png')
            save_image(diff_grid, diff_path)
    
    def _apply_heatmap_colorization(self, tensor):
        """
        Apply a heatmap colorization to a grayscale tensor.
        
        Args:
            tensor: Input tensor [B, C, H, W]
            
        Returns:
            Colorized tensor [B, 3, H, W]
        """
        # Convert to 3-channel
        if tensor.size(1) == 1:
            tensor = tensor.repeat(1, 3, 1, 1)
        
        # Apply different colormaps to each channel for visualization
        # (this is a simplified version - a proper implementation would use matplotlib colormaps)
        result = tensor.clone()
        
        # Red channel emphasizes high differences
        result[:, 0, :, :] = tensor[:, 0, :, :] ** 0.5
        
        # Green channel emphasizes mid differences
        result[:, 1, :, :] = tensor[:, 1, :, :] * (1 - tensor[:, 1, :, :])
        
        # Blue channel emphasizes low differences
        result[:, 2, :, :] = 1 - tensor[:, 2, :, :] ** 2
        
        return result
    
    def _plot_ablation_results(self, results):
        """
        Plot and save the results of the ablation study.
        
        Args:
            results: Dictionary with results for each scale value
        """
        # Create a DataFrame for easier plotting
        data = []
        for scale, metrics in results.items():
            for metric_name, value in metrics.items():
                data.append({
                    'Scale': scale,
                    'Metric': metric_name,
                    'Value': value
                })
        
        df = pd.DataFrame(data)
        
        # Create directory for plots
        plots_dir = self.output_dir / "plots"
        plots_dir.mkdir(exist_ok=True)
        
        # Plot each metric separately
        for metric in df['Metric'].unique():
            metric_df = df[df['Metric'] == metric]
            
            plt.figure(figsize=(10, 6))
            sns.lineplot(data=metric_df, x='Scale', y='Value')
            plt.title(f'Effect of Cross-Attention Scale on {metric}')
            plt.xlabel('Cross-Attention Scale')
            plt.ylabel(metric)
            plt.grid(True)
            
            # Add value annotations
            for i, row in metric_df.iterrows():
                plt.annotate(f"{row['Value']:.4f}", 
                            (row['Scale'], row['Value']),
                            textcoords="offset points",
                            xytext=(0, 10),
                            ha='center')
            
            plt.tight_layout()
            plt.savefig(plots_dir / f"ablation_{metric}.png")
            plt.close()
        
        # Create a comprehensive summary plot
        if len(df['Metric'].unique()) > 1:
            plt.figure(figsize=(12, 8))
            for metric in df['Metric'].unique():
                # Normalize values for comparison
                metric_df = df[df['Metric'] == metric].copy()
                min_val = metric_df['Value'].min()
                max_val = metric_df['Value'].max()
                if max_val > min_val:
                    metric_df['Normalized'] = (metric_df['Value'] - min_val) / (max_val - min_val)
                else:
                    metric_df['Normalized'] = 0.5  # Constant value if all values are the same
                
                sns.lineplot(data=metric_df, x='Scale', y='Normalized', label=metric)
            
            plt.title('Normalized Effect of Cross-Attention Scale on All Metrics')
            plt.xlabel('Cross-Attention Scale')
            plt.ylabel('Normalized Value')
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.savefig(plots_dir / "ablation_summary.png")
            plt.close()
        
        # Save raw data as CSV
        df.to_csv(plots_dir / "ablation_results.csv", index=False)
        
        logger.info(f"Saved ablation plots to {plots_dir}")
    
    def _plot_layer_impact(self, layer_results, baseline_metrics):
        """
        Plot and save the impact of each layer.
        
        Args:
            layer_results: Dictionary with results for each layer
            baseline_metrics: Baseline metrics with no cross-attention
        """
        # Create directory for plots
        plots_dir = self.output_dir / "plots"
        plots_dir.mkdir(exist_ok=True)
        
        # Extract improvement data
        improvements_data = []
        for layer_name, results in layer_results.items():
            for metric_name, improvement in results['improvements'].items():
                # Extract block and type from layer name for better grouping
                parts = layer_name.split('_')
                
                # Classify into block type and position
                if 'down' in layer_name:
                    block_type = 'down'
                    position = int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else 0
                elif 'mid' in layer_name:
                    block_type = 'mid'
                    position = 0
                elif 'up' in layer_name:
                    block_type = 'up'
                    position = int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else 0
                else:
                    block_type = 'other'
                    position = 0
                
                # Determine if self or cross attention
                attn_type = 'self' if 'self' in layer_name else 'cross'
                
                improvements_data.append({
                    'Layer': layer_name,
                    'Block Type': block_type,
                    'Position': position,
                    'Attention Type': attn_type,
                    'Metric': metric_name,
                    'Improvement': improvement
                })
        
        if not improvements_data:
            logger.warning("No improvement data available for plotting")
            return
        
        improvements_df = pd.DataFrame(improvements_data)
        
        # Sort by improvement for the bar plot
        for metric in improvements_df['Metric'].unique():
            metric_df = improvements_df[improvements_df['Metric'] == metric].copy()
            metric_df = metric_df.sort_values('Improvement', ascending=False)
            
            plt.figure(figsize=(14, 8))
            bar_plot = sns.barplot(x='Layer', y='Improvement', data=metric_df)
            plt.title(f'Impact of Each Layer on {metric}')
            plt.xlabel('Layer')
            plt.ylabel(f'Improvement in {metric}')
            plt.xticks(rotation=90)
            plt.grid(True, axis='y')
            
            # Add value annotations
            for i, p in enumerate(bar_plot.patches):
                bar_plot.annotate(f"{p.get_height():.4f}", 
                                (p.get_x() + p.get_width() / 2., p.get_height()),
                                ha='center', va='center', 
                                xytext=(0, 10), 
                                textcoords='offset points')
            
            plt.tight_layout()
            plt.savefig(plots_dir / f"layer_impact_{metric}.png")
            plt.close()
        
        # Create grouped plots by block type and position
        for metric in improvements_df['Metric'].unique():
            metric_df = improvements_df[improvements_df['Metric'] == metric].copy()
            
            # Create a hierarchical index for better visualization
            plt.figure(figsize=(14, 8))
            pivot_df = metric_df.pivot_table(
                index=['Block Type', 'Position'], 
                columns='Attention Type', 
                values='Improvement',
                aggfunc='mean'
            )
            
            pivot_df = pivot_df.sort_index(level=[0, 1])
            pivot_df.plot(kind='bar')
            
            plt.title(f'Average Layer Impact by Block Type and Position ({metric})')
            plt.xlabel('Block Type and Position')
            plt.ylabel(f'Improvement in {metric}')
            plt.grid(True, axis='y')
            plt.legend(title='Attention Type')
            plt.tight_layout()
            plt.savefig(plots_dir / f"layer_impact_grouped_{metric}.png")
            plt.close()
        
        # Create a heatmap visualization of layer impact
        for metric in improvements_df['Metric'].unique():
            metric_df = improvements_df[improvements_df['Metric'] == metric].copy()
            
            plt.figure(figsize=(12, 8))
            pivot_df = metric_df.pivot_table(
                index='Block Type', 
                columns=['Position', 'Attention Type'], 
                values='Improvement',
                aggfunc='mean'
            )
            
            # Plot heatmap
            sns.heatmap(pivot_df, annot=True, cmap='viridis', fmt=".4f", cbar_kws={'label': f'Improvement in {metric}'})
            plt.title(f'Heatmap of Layer Impact ({metric})')
            plt.tight_layout()
            plt.savefig(plots_dir / f"layer_impact_heatmap_{metric}.png")
            plt.close()
        
        # Save raw data as CSV
        improvements_df.to_csv(plots_dir / "layer_impact_results.csv", index=False)
        
        logger.info(f"Saved layer impact plots to {plots_dir}")
        
        # Identify and log the most impactful layers
        for metric in improvements_df['Metric'].unique():
            metric_df = improvements_df[improvements_df['Metric'] == metric].copy()
            top_layers = metric_df.nlargest(5, 'Improvement')
            
            logger.info(f"Top 5 most impactful layers for {metric}:")
            for i, (_, row) in enumerate(top_layers.iterrows(), 1):
                logger.info(f"  {i}. {row['Layer']}: {row['Improvement']:.6f}")