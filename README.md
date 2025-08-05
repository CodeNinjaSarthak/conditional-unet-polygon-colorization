# UNet Polygon Colorizer - Ayna ML Assignment

## Overview

This project involves training a UNet-based deep learning model to generate a colored polygon image, given:

- A grayscale image of a polygon (e.g., triangle, square, hexagon), and
- A color name (e.g., "red", "blue", "green").

The model is conditioned on the color input and trained end-to-end using PyTorch.

---

## Hyperparameters

| Hyperparameter              | Value     | Rationale                                                             |
| --------------------------- | --------- | --------------------------------------------------------------------- |
| Learning Rate               | `1e-3`    | A typical starting point for Adam optimizer, allowed fast convergence |
| Optimizer                   | `Adam`    | Robust optimizer suitable for image generation tasks                  |
| Batch Size                  | `16`      | Balanced memory usage and gradient estimation stability               |
| Epochs                      | `50`      | Allowed enough time to converge without overfitting                   |
| Image Size                  | `128x128` | Balanced resolution for clarity and fast training                     |
| Scheduler                   | `StepLR`  | Reduced LR over time to allow fine-tuning                             |
| Conditioning Embedding Size | `32`      | Embeds color name into a dense representation for conditioning        |

---

## Architecture: UNet Design

### Base Model

- A standard **UNet** with:
  - Downsampling path: 3 encoder blocks with Conv → ReLU → BatchNorm → MaxPool
  - Bottleneck: two Conv layers with dropout
  - Upsampling path: 3 decoder blocks with ConvTranspose2d → Conv → ReLU
  - Skip connections from encoder to decoder

### Conditioning Strategy

- Color name is converted into an **embedding vector**
- This vector is concatenated (broadcasted spatially) with the polygon image at multiple UNet stages (either input or mid-level features)

### Ablations Tried

- **No conditioning** baseline: Model could not generalize to different colors
- **Conditioning only at input**: Output showed weak color relevance
- **Conditioning at bottleneck only**: Improved performance, but less fine-grained detail
- **Best result**: Conditioning at multiple stages (input + bottleneck + decoder start)

---

## Training Dynamics

### Loss Function

- `MSELoss` between predicted and target RGB pixel values

### Metrics

- Mean Absolute Error (MAE) used as qualitative measure

### Observations

- Training stabilized after ~20 epochs
- Model initially learns to copy polygon shape, color accuracy improves later
- Validation loss mirrors training loss, no severe overfitting

### Typical Failure Modes

| Issue                        | Fix Attempted                      |
| ---------------------------- | ---------------------------------- |
| Slight color bleeding        | Added dropout + batch norm         |
| Incorrect color prediction   | Improved color embedding dimension |
| Misalignment in output edges | Added data augmentation (rotation) |

---

## WandB Logs

All training runs were logged on [Weights & Biases (wandb)](https://wandb.ai/). You can find:

- Loss and metric curves
- Sample outputs across epochs
- Comparisons of ablation experiments

**Project Link:** https://api.wandb.ai/links/project2034acc-n-a/qfu94ts0

---

## Key Learnings

- Conditioning in image generation is powerful but non-trivial — how and where to inject the signal matters.
- Embedding discrete inputs (like color names) into continuous latent space allows smooth integration with CNNs.
- Visualizing intermediate outputs is essential to debug model learning issues.
- Training segmentation/generation models benefits heavily from augmentation and normalization.

---

## Directory Structure

```
conditional-unet-polygon-colorization/
├── src/train.ipynb         # Training code with wandb logging
├── inference.ipynb     # Inference notebook with visualization
├── .gitignore            # UNet architecture with conditioning
├── data/dataset          # Training/validation data
├── README.md           # This file
```

## Deliverables

- Training code (`train.ipynb`)
- Inference demo (`inference.ipynb`)
- wandb project link
- README.md with insights
