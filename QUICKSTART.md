# Quick Start Guide

This guide will help you get started with Tiny AI Model Trainer in just a few minutes.

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/nheinstein/TinyAI.git
   cd TinyAI
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install the package in development mode:**
   ```bash
   pip install -e .
   ```

##  Quick Demo

Run the demo to see the trainer in action:

```bash
python demo.py
```

This will train both an LLM and a vision model with synthetic data to demonstrate the functionality.

##  Basic Usage

### Train an LLM

```bash
# Train with default configuration
python -m tinyai.train model=llm

# Train with custom parameters
python -m tinyai.train model=llm training.batch_size=16 training.learning_rate=1e-4

# Train with wandb logging
python -m tinyai.train model=llm logging.wandb=true logging.project_name=my-llm
```

### Train a Vision Model

```bash
# Train with default configuration
python -m tinyai.train model=vision

# Train with custom parameters
python -m tinyai.train model=vision training.batch_size=32 training.num_epochs=20

# Train with wandb logging
python -m tinyai.train model=vision logging.wandb=true logging.project_name=my-vision
```

##  Configuration

The trainer uses Hydra for configuration management. You can:

1. **Override parameters via command line:**
   ```bash
   python -m tinyai.train training.batch_size=64 training.learning_rate=1e-3
   ```

2. **Use different config files:**
   ```bash
   python -m tinyai.train model=vision training=custom
   ```

3. **Create your own configs** in the `configs/` directory

## Experiment Tracking

Enable wandb logging for experiment tracking:

```bash
# Set up wandb (first time only)
wandb login

# Run training with wandb
python -m tinyai.train logging.wandb=true logging.project_name=my-experiment
```

## Custom Models

To use your own model:

1. **Create a custom model class:**
   ```python
   from tinyai.models.base import BaseModel

   class MyModel(BaseModel):
       def __init__(self, config, device=None):
           super().__init__(config, device)
           # Your model implementation

       def forward(self, x):
           # Forward pass
           return output
   ```

2. **Register it in the model factory:**
   ```python
   # In tinyai/models/__init__.py
   def get_model(config, device=None):
       if config.type == "my_model":
           return MyModel(config, device)
   ```

##  Project Structure

```
TinyAI/
├── tinyai/                 # Main package
│   ├── models/            # Model implementations
│   ├── data/              # Data loading utilities
│   ├── training/          # Training loop and optimizers
│   └── utils/             # Logging and metrics
├── configs/               # Configuration files
│   ├── model/             # Model configurations
│   ├── training/          # Training configurations
│   ├── data/              # Data configurations
│   └── logging/           # Logging configurations
├── scripts/               # Example scripts
├── demo.py               # Demo script
└── requirements.txt      # Dependencies
```

##  Examples

### Example 1: Train a Small LLM

```bash
# Train a small transformer for text generation
python -m tinyai.train \
    model=llm \
    model.vocab_size=1000 \
    model.hidden_size=256 \
    model.num_layers=6 \
    training.num_epochs=10 \
    training.batch_size=16 \
    logging.wandb=true
```

### Example 2: Train a Vision Classifier

```bash
# Train a ResNet-style model for image classification
python -m tinyai.train \
    model=vision \
    model.num_classes=10 \
    model.hidden_size=64 \
    training.num_epochs=20 \
    training.batch_size=32 \
    logging.wandb=true
```

### Example 3: Use Custom Data

```bash
# Train with your own text data
python -m tinyai.train \
    model=llm \
    data.train_path=path/to/train.txt \
    data.val_path=path/to/val.txt \
    data.max_length=512
```

##  Troubleshooting

### Common Issues

1. **Import errors:** Make sure you've installed the package with `pip install -e .`

2. **CUDA out of memory:** Reduce batch size or model size
   ```bash
   python -m tinyai.train training.batch_size=8 model.hidden_size=128
   ```

3. **Wandb not working:** Make sure you're logged in with `wandb login`

### Getting Help

- Check the logs for detailed error messages
- Use the demo script to verify installation
- Review the configuration files in `configs/`

## Next Steps

1. **Read the full documentation** in `README.md`
2. **Explore the configuration files** in `configs/`
3. **Try the example scripts** in `scripts/`
4. **Customize for your use case** by modifying configs or adding custom models

Happy training!
