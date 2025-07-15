#!/usr/bin/env python3
"""
Demo script for Tiny AI.

This script demonstrates the basic usage of the trainer
with both LLM and vision models.
"""

import os
import sys
import torch
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, os.path.dirname(__file__))

from tinyai.models import get_model
from tinyai.data import get_data_loader
from tinyai.training import Trainer
from tinyai.utils.logging import setup_logging, get_logger
from tinyai.utils.metrics import MetricsTracker


def demo_llm_training():
    """Demo LLM training with synthetic data."""
    print("Starting LLM Training Demo")
    print("=" * 50)

    # Configuration
    config = {
        'model': {
            'type': 'transformer',
            'vocab_size': 1000,
            'hidden_size': 128,
            'num_layers': 4,
            'num_heads': 4,
            'max_length': 64
        },
        'data': {
            'type': 'text',
            'num_examples': 100,
            'max_length': 64,
            'vocab_size': 1000,
            'batch_size': 8
        },
        'training': {
            'num_epochs': 2,
            'batch_size': 8,
            'learning_rate': 1e-3,
            'optimizer': 'adam',
            'scheduler': 'none',
            'gradient_clip': 1.0,
            'log_interval': 10,
            'save_model': False,
            'save_best': False,
            'save_checkpoints': False
        },
        'logging': {
            'level': 'INFO',
            'wandb': False
        }
    }

    # Setup logging
    setup_logging(config['logging'])
    logger = get_logger(__name__)

    # Get device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Create model
    logger.info("Creating LLM model...")
    model = get_model(config['model'], device=device)
    logger.info(f"Model parameters: {model.get_num_parameters():,}")

    # Create data loaders
    logger.info("Creating data loaders...")
    train_loader = get_data_loader(config['data'], split="train")
    val_loader = get_data_loader(config['data'], split="val")

    # Create trainer
    logger.info("Creating trainer...")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config['training'],
        device=device,
        metrics_tracker=MetricsTracker()
    )

    # Start training
    logger.info("Starting training...")
    trainer.train()

    print("LLM Training Demo Completed!")
    print()


def demo_vision_training():
    """Demo vision model training with synthetic data."""
    print("Starting Vision Model Training Demo")
    print("=" * 50)

    # Configuration
    config = {
        'model': {
            'type': 'vision',
            'num_classes': 5,
            'in_channels': 3,
            'hidden_size': 32,
            'num_blocks': [1, 1, 1, 1]  # s
        },
        'data': {
            'type': 'image',
            'num_examples': 100,
            'image_size': 64,  # Smaller images for demo
            'num_classes': 5,
            'batch_size': 8
        },
        'training': {
            'num_epochs': 2,
            'batch_size': 8,
            'learning_rate': 1e-3,
            'optimizer': 'adam',
            'scheduler': 'none',
            'gradient_clip': 1.0,
            'log_interval': 10,
            'save_model': False,
            'save_best': False,
            'save_checkpoints': False
        },
        'logging': {
            'level': 'INFO',
            'wandb': False
        }
    }

    # Setup logging
    setup_logging(config['logging'])
    logger = get_logger(__name__)

    # Get device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Create model
    logger.info("Creating vision model...")
    model = get_model(config['model'], device=device)
    logger.info(f"Model parameters: {model.get_num_parameters():,}")

    # Create data loaders
    logger.info("Creating data loaders...")
    train_loader = get_data_loader(config['data'], split="train")
    val_loader = get_data_loader(config['data'], split="val")

    # Create trainer
    logger.info("Creating trainer...")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config['training'],
        device=device,
        metrics_tracker=MetricsTracker()
    )

    # Start training
    logger.info("Starting training...")
    trainer.train()

    print("Vision Model Training Demo Completed!")
    print()


def main():
    """Run the demo."""
    print("Tiny AI Demo")
    print("=" * 60)
    print()

    try:
        # Demo LLM training
        demo_llm_training()

        # Demo vision training
        demo_vision_training()

        print("All demos completed successfully!")
        print()
        print("Next steps:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Run with real data: python -m tinyai.train model=llm")
        print("3. Use wandb logging: python -m tinyai.train logging.wandb=true")
        print("4. Customize configs in the configs/ directory")

    except Exception as e:
        print(f"❌ Demo failed: {str(e)}")
        print("Make sure you have installed the dependencies:")
        print("pip install -r requirements.txt")


if __name__ == "__main__":
    main()
