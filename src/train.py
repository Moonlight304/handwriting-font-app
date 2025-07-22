"""
Robust Training Module for Handwriting CVAE
With proper input normalization and error handling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from datetime import datetime
from dataset import HandwritingDataset
from model import CVAE
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TrainConfig:
    def __init__(self):
        self.latent_dim = 32
        self.num_classes = 62  # 0-9, a-z, A-Z
        self.batch_size = 64
        self.epochs = 50
        self.learning_rate = 1e-3
        self.beta = 0.5  # Weight for KLD loss
        self.save_interval = 5  # Save model every N epochs
        self.model_dir = Path('saved_models')
        self.log_dir = Path('logs')

def setup_environment(config):
    """Create directories and set up device"""
    config.model_dir.mkdir(exist_ok=True)
    config.log_dir.mkdir(exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    return device

def get_data_loaders(csv_file, root_dir, batch_size):
    """Create training data loaders with augmentation"""
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.RandomRotation(5),
        transforms.RandomAffine(0, translate=(0.05, 0.05)),
        transforms.RandomResizedCrop(64, scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]
    ])
    
    dataset = HandwritingDataset(csv_file, root_dir, transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

def loss_function(recon_x, x, mu, logvar, beta=0.5):
    """Enhanced loss function with proper sigmoid activation"""
    # Apply sigmoid to ensure values are in [0,1] range for BCE
    recon_x = torch.sigmoid(recon_x)
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + beta * KLD

def train_model(csv_file, root_dir, config):
    """Main training loop"""
    device = setup_environment(config)
    
    # Initialize components
    dataloader = get_data_loaders(csv_file, root_dir, config.batch_size)
    model = CVAE(latent_dim=config.latent_dim, 
                num_classes=config.num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), 
                               lr=config.learning_rate)
    
    logger.info(f"Starting training with {len(dataloader.dataset)} samples")
    logger.info(f"Latent dim: {config.latent_dim}, Classes: {config.num_classes}")

    for epoch in range(config.epochs):
        model.train()
        epoch_loss = 0
        
        for batch_idx, (images, labels) in enumerate(dataloader):
            images = (images + 1) / 2  # Convert from [-1,1] to [0,1] for BCE
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            recon_images, mu, logvar = model(images, labels)
            
            # Loss calculation
            loss = loss_function(recon_images, images, mu, logvar, config.beta)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            if batch_idx % 10 == 0:
                logger.info(
                    f"Epoch {epoch+1}/{config.epochs} | "
                    f"Batch {batch_idx}/{len(dataloader)} | "
                    f"Loss: {loss.item():.4f}"
                )
        
        # Epoch statistics
        avg_loss = epoch_loss / len(dataloader)
        logger.info(f"Epoch {epoch+1} Complete | Avg Loss: {avg_loss:.4f}")
        
        # Save model checkpoint
        if (epoch + 1) % config.save_interval == 0 or (epoch + 1) == config.epochs:
            model_path = config.model_dir / f"cvae_epoch_{epoch+1}.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'config': vars(config)
            }, model_path)
            logger.info(f"Model saved to {model_path}")

    return model

if __name__ == "__main__":
    # Initialize configuration
    config = TrainConfig()
    
    # Training parameters
    csv_file = 'data/labels.csv'
    root_dir = 'data/segmented_characters'
    
    try:
        # Run training
        trained_model = train_model(csv_file, root_dir, config)
        
        # Save final model
        final_path = config.model_dir / "handwriting_cvae_final.pth"
        torch.save(trained_model.state_dict(), final_path)
        logger.info(f"Final model successfully saved to {final_path}")
    except Exception as e:
        logger.error(f"Training failed: {str(e)}", exc_info=True)
        raise