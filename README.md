# 🎭 Face Generation with DCGANs

This project implements a **Deep Convolutional Generative Adversarial Network (DCGAN)** in TensorFlow/Keras to generate realistic human face images using the **CelebA dataset**. It follows the standard GAN pipeline with a custom training loop and visualizes how the generator improves over training epochs.  

<img width="1600" height="1600" alt="gen_images_epoch_20" src="https://github.com/user-attachments/assets/b676e646-1d56-4635-96e7-21747d4bd360" />
---

## 📌 Project Overview
- **Goal:** Train a GAN that learns the distribution of celebrity face images and generates new, fake faces that resemble real ones.  
- **Approach:**  
  - **Generator**: Transposed convolutions with BatchNorm and ReLU activations.  
  - **Discriminator**: Convolutions with LeakyReLU activations to classify real vs fake.  
  - **Training loop**: Implemented via a subclassed `tf.keras.Model`, with separate optimizers for D and G.  
  - **Tricks used:** Label smoothing, label noise, and balanced training steps to stabilize GAN learning.  
- **Visualization:** A callback generates and saves 6×6 image grids after each epoch to monitor progress.  

---

## 🚀 Features
- Loads and preprocesses **CelebA dataset** automatically from Kaggle.  
- Custom **GAN training loop** with `train_step`.  
- Includes **label smoothing + noise** to reduce discriminator overconfidence.  
- Uses **`tanh` normalization** for stable generator outputs.  
- Saves generated face grids to a `generated/` folder at each epoch.  

---

