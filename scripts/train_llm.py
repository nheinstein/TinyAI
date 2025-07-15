#!/usr/bin/env python3
"""
Script to train an LLM with Tiny AI.

This script demonstrates how to train a language model
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
def train_llm(cfg: DictConfig):
    """
    Train an LLM with custom configuration.

    Args:
        cfg: Hydra configuration
    """
    # Override configuration for LLM training
    cfg.model.type = "transformer"
    cfg.model.vocab_size = 1000
    cfg.model.hidden_size = 256
    cfg.model.num_layers = 6
    cfg.model.num_heads = 8

    cfg.training.num_epochs = 5
    cfg.training.batch_size = 16
    cfg.training.learning_rate = 1e-4

    cfg.data.type = "text"
    cfg.data.num_examples = 500  # Use synthetic data

    cfg.logging.wandb = True
    cfg.logging.project_name = "tiny-llm-demo"
    cfg.logging.run_name = "llm-training"

    print("Configuration:")
    print(OmegaConf.to_yaml(cfg))

    # Start training
    main(cfg)


if __name__ == "__main__":
    train_llm()
