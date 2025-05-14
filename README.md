# ViT_CIFAR10
# Vision Transformer on CIFAR10

This project demonstrates how to implement the Vision Transformer (ViT) model on the CIFAR-10 dataset.

## Overview
Vision Transformer (ViT) uses a Transformer-based architecture that operates on image patches. This implementation shows how to set up the ViT model using the `transformers` library from Hugging Face.

## Steps
1. **Data Loading**: We load the CIFAR-10 dataset using PyTorch's torchvision.
2. **Model Setup**: We use the ViT model from the Hugging Face `transformers` library.
3. **Training**: The model is trained using standard settings on CIFAR-10.
4. **Evaluation**: Model evaluation is performed to check the performance on the CIFAR-10 dataset.

## Requirements
- Python 3.7+
- PyTorch
- Hugging Face Transformers

## Installation
```bash
pip install torch torchvision transformers
```

## Usage
1. Clone this repository.
2. Run the provided scripts to train and evaluate the model on CIFAR-10.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
