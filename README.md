# Denoising Autoencoder

A Streamlit web application that demonstrates a denoising autoencoder trained on the MNIST dataset. The app allows users to interactively add noise to digit images and see how the autoencoder reconstructs them.

## Features

- Interactive noise level adjustment using a slider
- Selection of different MNIST digit images
- Real-time visualization of:
  - Original image
  - Noisy image (with adjustable noise)
  - Denoised reconstruction
- MSE loss calculation between original and reconstructed images

## Technical Details

### Model Architecture

The project implements a convolutional autoencoder with the following architecture:

- **Encoder**:
  - Conv2d layers with increasing channels (1→32→64)
  - BatchNorm2d and ReLU activations
  - Stride-2 convolutions for downsampling

- **Decoder**:
  - ConvTranspose2d layers with decreasing channels (64→32→1)
  - BatchNorm2d and ReLU activations
  - Stride-2 deconvolutions for upsampling
  - Final Sigmoid activation

### Training

The model was trained on the MNIST dataset with:
- Dynamic noise levels during training
- MSE loss function
- Adam optimizer
- Batch normalization for stable training

## Installation

1. Clone the repository:

```bash
git clone https://github.com/martinijfb/autoencoder.git
cd autoencoder
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Run the Streamlit app:

```bash
streamlit run main.py
```

The app will open in your default web browser. You can then:
1. Select a digit image (0-9)
2. Adjust the noise level using the slider (0.0 to 1.0)
3. Observe the denoising results in real-time

## Project Structure

```
├── main.py                 # Streamlit app
├── utils/
│   └── utils.py           # Model definition and helper functions
├── models/
│   └── autoencoder_0.pth  # Trained model weights
├── data/                  # Sample MNIST images
└── experiments/
    └── autoencoder.ipynb  # Training notebook
```

## Model Training

The model training process is documented in `experiments/autoencoder.ipynb`. Key aspects include:

- Training on the full MNIST dataset
- Dynamic noise levels during training for better generalization
- Monitoring of training and validation losses
- Visual inspection of denoising performance during training
- Model checkpointing to save best weights

## Hardware Acceleration

The application automatically detects and uses available hardware acceleration:
- CUDA for NVIDIA GPUs
- MPS for Apple Silicon
- Falls back to CPU if neither is available

