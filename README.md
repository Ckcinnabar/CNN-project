# CNN-project

## Introduction

This project demonstrates advanced machine learning techniques using TensorFlow, Keras, and Hugging Face's libraries. It's designed to showcase skills in deep learning, focusing on image classification, zero-shot learning, and image generation.

## Installation and Environment Setup

### Dependencies

Required libraries:

- Python 3.8+
- TensorFlow 2.8.0+
- Keras
- Hugging Face's Transformers and Diffusers
- NumPy, Matplotlib, etc.

### Installation

Install the libraries using pip:

```bash
pip install tensorflow
pip install keras
pip install transformers datasets evaluate
pip install diffusers
pip install ftfy accelerate
```

## Data Preparation
### Dataset Acquisition
- The dataset is available for download from a provided Google Drive link.
- After downloading, unzip the dataset to a specified directory.
### Data Loading
- Use Keras utilities to load the image dataset.
- The dataset should be divided into training, validation, and test sets.

## Model Building and Training
### Convolutional Neural Network (CNN) Model
- Model Architecture: Utilize EfficientNetV2S with custom layers (dropout, dense).
- Training: Train using the Adam optimizer. Implement a callback for model checkpointing based on validation loss.
- Data Augmentation: To prevent overfitting, apply techniques like random flipping, rotation, zooming, cropping, and contrast adjustment.

### Zero-Shot Learning with CLIP
- Setup: Load the CLIP model ('openai/clip-vit-large-patch14') and its processor.
- Implementation: Employ CLIP for zero-shot image classification on the test dataset.

### Image Generation with Stable Diffusion
- Image Creation: Generate images using textual prompts with the Stable Diffusion model.
- Evaluation: Resize and analyze these images using the CNN model to assess model performance.

## Project Structure
- data_loader.py: Script for loading and preprocessing the dataset.
- model.py: Contains the CNN model architecture and training routines.
- zero_shot_clip.py: Script for implementing zero-shot learning with CLIP.
- image_generation.py: Contains the image generation process using Stable Diffusion.
- utils/: Directory containing utility functions and classes.

## Conclusion
This project provides a comprehensive overview of using deep learning for image-related tasks. It covers CNN-based image classification, explores the capabilities of zero-shot learning using CLIP, and delves into the realm of AI-driven image generation with Stable Diffusion.
