# Generating Faces with GANs (DCGAN vs WGAN-GP)

This project investigates generating human face images using two deep generative models: Deep Convolutional GAN (DCGAN) and Wasserstein GAN with Gradient Penalty (WGAN-GP). Both models are trained on the CelebA dataset at 128×128 resolution.
The goal is to compare training stability, visual quality of generated samples, and quantitative evaluation using FID and KID metrics.

This repository includes implementation of both models in PyTorch, training loops, evaluation logic, and a detailed analysis of results.

## Features

- Implemented DCGAN and WGAN-GP architectures in PyTorch  
- Preprocessing pipeline for CelebA (central crop, resizing, normalization)  
- Training loops with stable optimization setups  
- Automatic evaluation using FID and KID  
- Generation of 10,000 synthetic samples for statistical comparison  
- Visualization of intermediate training outputs  
- Side-by-side comparison and detailed discussion of model performance

## Model Overview

### DCGAN
- Generator uses ConvTranspose2D layers  
- Discriminator uses Conv2D with LeakyReLU  
- BatchNorm widely used  
- Loss: Binary Cross-Entropy  
- Known issues: mode collapse, gradient saturation, unstable optimization

### WGAN-GP
- Replaces BCE loss with Wasserstein distance  
- Discriminator replaced by a critic producing real-valued scores  
- Enforces 1-Lipschitz constraint via gradient penalty  
- More stable and consistent gradient flow  
- Loss functions:

```
L_D = E[D(fake)] - E[D(real)] + λ * GP
L_G = - E[D(fake)]
```

## Dataset: CelebA

- 202,599 face images  
- Large diversity in lighting, background, pose, and expression  
- Preprocessing steps:
  - Center crop to 178×178  
  - Resize to 128×128  
  - Normalize pixel values to the range [-1, 1]  
- Only RGB images are used (attributes not included)

## Training

### DCGAN Training
- 16 epochs total  
- Adam optimizer  
- Uses instance noise in early epochs for additional stability  
- Generator and Discriminator updated alternately  
- EMA generator used for evaluation output  

### WGAN-GP Training
- 100 + 10 fine-tuning epochs  
- Critic updated multiple times per generator step (n_critic = 7)  
- Gradient penalty coefficient λ = 10  
- Training behavior shows steady critic loss, minimal oscillations  
- EMA generator also maintained for evaluation

## Evaluation: FID and KID

After training each model, 10,000 generated samples were compared with 10,000 real CelebA images.

Key observations:

- DCGAN achieves lower FID and KID (better numerically)
- WGAN-GP often produces visually more coherent and stable images
- Numerical disadvantages of WGAN-GP arise due to:
  - sensitivity of Inception-based metrics to local texture statistics  
  - softer textures produced by WGAN-GP  
  - limited hyperparameter tuning in this experiment  

## Possible Future Work

- Complete hyperparameter search for WGAN-GP (n_critic, GP weight, LR schedules)  
- Training at different resolutions (e.g., 64×64, 256×256 or 512×512)  
- Implementing StyleGAN/StyleGAN2  
- Conditional GANs using CelebA attributes  
- Additional evaluation metrics (Precision/Recall for GANs, FDD, Intra-FID)  
