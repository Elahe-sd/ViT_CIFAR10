
# Vision Transformer (ViT) and ResNet101 on Flowers102 Dataset

## Overview

This project implements and compares two popular image classification models — **Vision Transformer (ViT)** and **ResNet101** — fine-tuned on the Flowers102 dataset. It demonstrates data loading, model training, evaluation, and visualization of training progress and final accuracy.

## Features

* Preprocessing and loading the Flowers102 dataset with PyTorch
* Fine-tuning pretrained ViT (from TIMM) and ResNet101 (from torchvision)
* Training with cross-entropy loss and Adam optimizer
* Test accuracy evaluation for both models
* Visualization of training accuracy across epochs
* Bar chart comparison of final test accuracy
* Saving trained model weights

## Dataset

[Flowers102](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/) consists of 102 flower categories. The dataset is automatically downloaded and loaded using torchvision.

## Requirements

* Python 3.7+
* PyTorch
* torchvision
* timm
* matplotlib

## Installation

```bash
pip install torch torchvision timm matplotlib
```

## Usage

1. Clone this repository.

2. Run the training script:

   ```bash
   python train_vit_resnet_flowers102.py
   ```

3. The script will:

   * Download and prepare the Flowers102 dataset
   * Fine-tune ViT and ResNet101 models for 5 epochs each
   * Evaluate and print test accuracy for both models
   * Save model weights as `vit_dog_classifier.pth` and `resnet101_dog_classifier.pth`
   * Show plots comparing training accuracy and final test accuracy

## Results

* ViT typically achieves higher accuracy than ResNet101 on this dataset with the given settings.
* Training accuracy and test accuracy comparison plots are generated for visual analysis.

## Future Improvements

* Add validation set and early stopping
* Hyperparameter tuning for learning rate and batch size
* Longer training epochs for better convergence
* Expand model comparison to other architectures

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.


