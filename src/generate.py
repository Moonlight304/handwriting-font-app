"""
Enhanced Character Generation Module
Generates high-quality handwriting samples with improved post-processing and device handling.
"""

import os
import logging
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Tuple, Dict
from model import CVAE
from dataset import ALL_CLASSES
import json

logger = logging.getLogger(__name__)

def setup_logging():
    """Configure logging format and level"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def postprocess_image(
    image: np.ndarray,
    threshold: float = 0.5,
    invert: bool = True,
    smooth: bool = True
) -> np.ndarray:
    """
    Enhanced post-processing for generated images.
    
    Args:
        image: Raw generated image array
        threshold: Binarization threshold (0-1)
        invert: Whether to invert black/white
        smooth: Apply Gaussian smoothing
        
    Returns:
        Processed uint8 image array
    """
    # Normalize to [0, 1] with epsilon to avoid division by zero
    image = (image - image.min()) / (image.max() - image.min() + 1e-8)
    
    # Optional smoothing
    if smooth:
        from scipy.ndimage import gaussian_filter
        image = gaussian_filter(image, sigma=0.5)
    
    # Binarize and invert
    image = (image > threshold).astype(np.float32)
    if invert:
        image = 1.0 - image
    
    # Convert to 8-bit and ensure proper range
    return (image * 255).clip(0, 255).astype(np.uint8)

def save_generation_metadata(
    output_dir: Path,
    config: Dict,
    char_indices: Dict[str, int]
) -> None:
    """Save generation parameters and label mapping"""
    metadata = {
        'generation_config': config,
        'character_mapping': char_indices
    }
    
    with open(output_dir / 'generation_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

def generate_characters(
    model: CVAE,
    output_dir: str = './data/generated_characters',
    num_samples: int = 10,
    image_size: Tuple[int, int] = (64, 64),
    device: Optional[str] = None,
    temperature: float = 0.7,
    threshold: float = 0.5
) -> None:
    """
    Generate high-quality character samples with enhanced control.
    
    Args:
        model: Trained CVAE model
        output_dir: Output directory path
        num_samples: Samples per character
        image_size: (height, width) of output images
        device: Device to use (cuda/cpu)
        temperature: Generation randomness (0.1-1.0)
        threshold: Binarization threshold (0-1)
    """
    try:
        setup_logging()
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Device handling
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        model.eval()
        
        logger.info(f"Generation Parameters:\n"
                   f"- Output directory: {output_path.resolve()}\n"
                   f"- Samples per character: {num_samples}\n"
                   f"- Image size: {image_size}\n"
                   f"- Device: {device}\n"
                   f"- Temperature: {temperature}\n"
                   f"- Threshold: {threshold}")

        # Create character mapping
        char_indices = {char: idx for idx, char in enumerate(ALL_CLASSES)}
        save_generation_metadata(output_path, {
            'num_samples': num_samples,
            'temperature': temperature,
            'threshold': threshold
        }, char_indices)

        with torch.no_grad():
            for char_idx, char in enumerate(ALL_CLASSES):
                logger.info(f"Generating {num_samples} samples for '{char}' (class {char_idx})")
                
                for sample_idx in range(num_samples):
                    # Prepare inputs
                    y = torch.tensor([char_idx], device=device)
                    z = torch.randn(1, model.latent_dim, device=device) * temperature
                    
                    # Generate and process image
                    gen_image = model.decode(z, y).squeeze().cpu().numpy()
                    gen_image = postprocess_image(gen_image, threshold=threshold)
                    
                    # Save with metadata
                    filename = output_path / f"{char_idx:03d}_{char}_{sample_idx:03d}.png"
                    plt.figure(figsize=(1, 1), dpi=max(image_size))
                    plt.imshow(gen_image, cmap='gray', vmin=0, vmax=255)
                    plt.axis('off')
                    plt.savefig(filename, bbox_inches='tight', pad_inches=0)
                    plt.close()

        logger.info(f"Successfully generated {len(ALL_CLASSES)*num_samples} character samples")

    except Exception as e:
        logger.error(f"Generation failed: {str(e)}", exc_info=True)
        raise

def load_model(model_path: str, device: str) -> CVAE:
    """Load trained model with error handling"""
    try:
        model = CVAE(num_classes=len(ALL_CLASSES)).to(device)
        state_dict = torch.load(model_path, map_location=device)
        
        # Handle both full checkpoint and state_dict
        if 'model_state_dict' in state_dict:
            model.load_state_dict(state_dict['model_state_dict'])
        else:
            model.load_state_dict(state_dict)
            
        return model
    except Exception as e:
        logger.error(f"Model loading failed: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {device}")
        
        # Configuration
        config = {
            'model_path': 'saved_models/handwriting_cvae_final.pth',
            'output_dir': './data/generated_characters',
            'num_samples': 10,
            'temperature': 0.7,
            'threshold': 0.6
        }
        
        # Load and generate
        model = load_model(config['model_path'], device)
        generate_characters(
            model,
            output_dir=config['output_dir'],
            num_samples=config['num_samples'],
            device=device,
            temperature=config['temperature'],
            threshold=config['threshold']
        )
    except Exception as e:
        logger.error(f"Generation process failed: {str(e)}")