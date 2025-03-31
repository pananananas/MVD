#!/usr/bin/env python
"""
Validate the image cross-attention implementation through ablation studies.
This script runs comprehensive validation tests to measure the impact of image cross-attention
on the model's performance.
"""

import os
import argparse
import logging
import torch
import yaml
from pathlib import Path
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import our modules correctly matching train.py
from src.data.objaverse_dataset import ObjaverseDataModule
from src.models.mvd_unet import create_mvd_pipeline
from src.training.training import MVDLightningModule
from src.validation import CrossAttentionValidator

def parse_args():
    parser = argparse.ArgumentParser(description="Validate image cross-attention implementation")
    parser.add_argument("--config", type=str, default="config/train_config.yaml", 
                       help="Path to config file")
    parser.add_argument("--checkpoint", type=str, default=None,
                       help="Path to model checkpoint (optional)")
    parser.add_argument("--output_dir", type=str, default="validation_results",
                       help="Directory to save validation results")
    parser.add_argument("--data_dir", type=str, default=None,
                       help="Directory containing validation data")
    parser.add_argument("--dataset_path", type=str, default=None,
                       help="Path to dataset (if different from config)")
    parser.add_argument("--scales", type=float, nargs="+", default=[0.0, 0.1, 0.5, 1.0, 2.0],
                       help="Scale values to test")
    parser.add_argument("--batch_size", type=int, default=4,
                       help="Batch size for validation")
    parser.add_argument("--num_batches", type=int, default=10,
                       help="Number of batches to validate on")
    parser.add_argument("--skip_ablation", action="store_true",
                       help="Skip the ablation study")
    parser.add_argument("--skip_layer_analysis", action="store_true",
                       help="Skip the per-layer analysis")
    
    return parser.parse_args()

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def main():
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    logger.info(f"Loaded configuration from {args.config}")
    
    # Set output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"validation_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save the run configuration
    with open(output_dir / "validation_config.yaml", 'w') as f:
        yaml.dump(vars(args), f)
    
    # Determine the device
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Update config for validation
    if isinstance(config['image_size'], list):
        config['image_size'] = tuple(config['image_size'])
    
    # Set appropriate num_workers based on device
    if device.type == "cuda":
        config['num_workers'] = 16
    elif device.type == "mps":
        config['num_workers'] = 6
    else:
        config['num_workers'] = 1
    
    # Override batch size if specified
    if args.batch_size:
        config['batch_size'] = args.batch_size
    
    # Create model following the same pattern as train.py
    logger.info("Creating model...")
    pipeline = create_mvd_pipeline(
        pretrained_model_name_or_path=config["architecture"],
        dtype=getattr(torch, config['torch_dtype']),
        use_memory_efficient_attention=config.get("use_memory_efficient_attention", True),
        enable_gradient_checkpointing=config.get("enable_gradient_checkpointing", True),
    )
    
    # Create Lightning module (same as in training)
    model = MVDLightningModule(
        pipeline=pipeline,
        config=config,
        output_dir=str(output_dir)
    )
    
    # Load from checkpoint if provided
    if args.checkpoint:
        logger.info(f"Loading checkpoint from {args.checkpoint}")
        model = MVDLightningModule.load_from_checkpoint(
            args.checkpoint,
            pipeline=pipeline,
            config=config,
            output_dir=str(output_dir)
        )
    
    # Move model to device and set to evaluation mode
    model = model.to(device)
    model.eval()
    
    # Create data module (same as in training)
    logger.info("Setting up data module...")
    dataset_path = args.dataset_path or args.data_dir or config.get("dataset_path", None)
    # dataset_path = "/Users/ewojcik/Code/pwr/MVD/objaverse"
    SCRATCH = os.getenv('SCRATCH', '/net/tscratch/people/plgewoj')
    HUGGINGFACE_CACHE = os.path.join(SCRATCH, 'huggingface_cache')
    os.makedirs(HUGGINGFACE_CACHE, exist_ok=True)
    os.environ['HF_HOME'] = HUGGINGFACE_CACHE


    # dataset_path = "/Users/ewojcik/Code/pwr/MVD/objaverse"
    dataset_path = "/net/pr2/projects/plgrid/plggtattooai/MeshDatasets/objaverse/"
    if not dataset_path:
        raise ValueError("Dataset path not specified in args or config")
    
    data_module = ObjaverseDataModule(
        data_root=dataset_path,
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        target_size=(config['image_size'][0], config['image_size'][1]),
        max_views_per_object=config['max_views_per_object'],
        max_samples=config.get('max_samples', None),
    )
    
    # Setup the data module (necessary to create the dataloader)
    data_module.setup()
    
    # Create validation dataloader
    val_dataloader = data_module.val_dataloader()
    
    # Create validator with the Lightning module
    logger.info("Creating validator...")
    validator = CrossAttentionValidator(model, output_dir=output_dir)
    
    # Run ablation study
    if not args.skip_ablation:
        logger.info("Running ablation study...")
        ablation_results = validator.run_ablation_study(
            dataloader=val_dataloader,
            scales=args.scales,
            save_images=True
        )
        
        logger.info("Ablation study results:")
        for scale, metrics in ablation_results.items():
            logger.info(f"  Scale {scale}: {metrics}")
    
    # Run per-layer analysis
    if not args.skip_layer_analysis:
        logger.info("Running per-layer analysis...")
        layer_results = validator.per_layer_analysis(
            dataloader=val_dataloader,
            num_batches=min(args.num_batches, len(val_dataloader))
        )
        
        logger.info("Layer analysis completed.")
    
    logger.info(f"Validation completed. Results saved to {output_dir}")
    
    # Print guidance for interpreting results
    print("\n" + "="*80)
    print("INTERPRETATION GUIDE FOR VALIDATION RESULTS")
    print("="*80)
    print("\nAblation Study Results:")
    print("  - Examine how performance changes as cross-attention scale increases")
    print("  - If loss decreases as scale increases, cross-attention is helping")
    print("  - Look for the optimal scale value (typically where metrics plateau)")
    print("\nPer-Layer Analysis:")
    print("  - Identify which layers contribute most to performance improvements")
    print("  - Patterns across down/mid/up blocks reveal architectural insights")
    print("  - Self vs cross attention comparisons show which mechanism is more effective")
    print("\nImage Comparisons:")
    print("  - Compare generated images across different scale values")
    print("  - Look for improved details, consistency with source images")
    print("  - Difference visualizations highlight areas most affected by cross-attention")
    print("\nConclusions:")
    print("  - If results show improvements with cross-attention, implementation is working")
    print("  - Layer analysis can guide architectural refinements")
    print("  - Optimal scale values should be used in final model configuration")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()