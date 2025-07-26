# ISTVT: Interpretable Spatial-Temporal Video Transformer for Deepfake Detection

This project implements a simplified version of the **ISTVT** (Interpretable Spatial-Temporal Video Transformer) model for detecting Deepfake videos. The model is designed to learn both **spatial** (frame-wise) and **temporal** (sequence-wise) inconsistencies using a transformer-based architecture.

---

## 📌 Key Features

- ✅ Uses **ViT (Vision Transformer)** as a base for video-level classification
- ✅ Preprocessing with **MTCNN** for face extraction from videos
- ✅ Converts images to tensor sequences for model input
- ✅ Frame-based training with aggregated video-level predictions
- ✅ Visual outputs: confusion matrix, ROC curve, classification report
- ✅ Modular codebase for easy customization

---

## 📁 Project Structure

