#!/usr/bin/env python3
"""
Script to find optimal learning rate using Tiny AI Model Trainer.

This script demonstrates how to use the learning rate finder
to find optimal learning rates for your models.
"""

import os
import sys
import torch
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tinyai.models import get_model
from tinyai.data import get_data_loader
from tinyai.training import get_optimizer
from tinyai.utils.logging import setup_logging, get_logger
from tinyai.utils.lr_finder import find_lr


def find_lr_for_llm():
    """Find optimal learning rate for LLM model."""
    print("🔍 Finding optimal learning rate for LLM model")
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
            'num_examples': 200,
            'max_length': 64,
            'vocab_size': 1000,
            'batch_size': 8
        },
        'training': {
            'learning_rate': 1e-4,
            'optimizer': 'adam'
        },
        'logging': {
            'level': 'INFO'
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
    
    # Create data loader
    logger.info("Creating data loader...")
    train_loader = get_data_loader(config['data'], split="train")
    
    # Create optimizer
    logger.info("Creating optimizer...")
    optimizer = get_optimizer(model, config['training'])
    
    # Find learning rate
    logger.info("Starting learning rate finder...")
    suggestions = find_lr(
        model=model,
        train_loader=train_loader,
        optimizer=optimizer,
        device=device,
        start_lr=1e-7,
        end_lr=1e-1,
        num_iter=50,
        plot=True,
        save_plot="llm_lr_finder.png"
    )
    
    print("✅ LLM Learning Rate Finder Completed!")
    print(f"Suggested learning rate: {suggestions['suggested_lr']:.2e}")
    print()


def find_lr_for_vision():
    """Find optimal learning rate for vision model."""
    print("🔍 Finding optimal learning rate for Vision model")
    print("=" * 50)
    
    # Configuration
    config = {
        'model': {
            'type': 'vision',
            'num_classes': 5,
            'in_channels': 3,
            'hidden_size': 32,
            'num_blocks': [1, 1, 1, 1]
        },
        'data': {
            'type': 'image',
            'num_examples': 200,
            'image_size': 64,
            'num_classes': 5,
            'batch_size': 8
        },
        'training': {
            'learning_rate': 1e-3,
            'optimizer': 'adam'
        },
        'logging': {
            'level': 'INFO'
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
    
    # Create data loader
    logger.info("Creating data loader...")
    train_loader = get_data_loader(config['data'], split="train")
    
    # Create optimizer
    logger.info("Creating optimizer...")
    optimizer = get_optimizer(model, config['training'])
    
    # Find learning rate
    logger.info("Starting learning rate finder...")
    suggestions = find_lr(
        model=model,
        train_loader=train_loader,
        optimizer=optimizer,
        device=device,
        start_lr=1e-7,
        end_lr=1e-1,
        num_iter=50,
        plot=True,
        save_plot="vision_lr_finder.png"
    )
    
    print("✅ Vision Learning Rate Finder Completed!")
    print(f"Suggested learning rate: {suggestions['suggested_lr']:.2e}")
    print()


def main():
    """Run the learning rate finder for both models."""
    print("🎯 Learning Rate Finder Demo")
    print("=" * 60)
    print()
    
    try:
        # Find LR for LLM
        find_lr_for_llm()
        
        # Find LR for Vision
        find_lr_for_vision()
        
        print("🎉 Learning rate finder completed successfully!")
        print()
        print("Generated plots:")
        print("- llm_lr_finder.png")
        print("- vision_lr_finder.png")
        print()
        print("Next steps:")
        print("1. Use the suggested learning rates in your training")
        print("2. Run: python -m tinyai.train training.learning_rate=<suggested_lr>")
        print("3. Experiment with different learning rate ranges")
        
    except Exception as e:
        print(f"❌ Learning rate finder failed: {str(e)}")
        print("Make sure you have installed the dependencies:")
        print("pip install -r requirements.txt")


if __name__ == "__main__":
    main() 