#!/usr/bin/env python3
"""
Script to train a vision model with Tiny AI Model Trainer.

This script demonstrates how to train a vision model
with custom configuration overrides.
"""

import hydra
from omegaconf import DictConfig, OmegaConf
import os
import sys

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tinyai.train import main


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def train_vision(cfg: DictConfig):
    """
    Train a vision model with custom configuration.
    
    Args:
        cfg: Hydra configuration
    """
    # Override configuration for vision training
    cfg.model.type = "vision"
    cfg.model.num_classes = 10
    cfg.model.hidden_size = 64
    cfg.model.num_blocks = [2, 2, 2, 2]  # ResNet-18 style
    
    cfg.training.num_epochs = 5
    cfg.training.batch_size = 32
    cfg.training.learning_rate = 1e-3
    
    cfg.data.type = "image"
    cfg.data.num_examples = 1000  # Use synthetic data
    cfg.data.image_size = 224
    
    cfg.logging.wandb = True
    cfg.logging.project_name = "tiny-vision-demo"
    cfg.logging.run_name = "vision-training"
    
    print("Configuration:")
    print(OmegaConf.to_yaml(cfg))
    
    # Start training
    main(cfg)


if __name__ == "__main__":
    train_vision() 