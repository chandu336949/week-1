# 🫁 Pneumonia Detection CNN

**Detect pneumonia from chest X-rays using AI**

---

## 🎯 Project Overview

Automated pneumonia detection using a **Convolutional Neural Network (CNN)** trained on chest X-ray images. Classifies X-rays as **Normal** or **Pneumonia** with ~87% accuracy.

**Key Features:**
- ⚡ Lightweight (CPU-friendly, no GPU needed)
- 🚀 Fast inference (~milliseconds per image)
- 📚 Perfect learning project for deep learning basics
- 🎓 Great for placement portfolio

---

## 📊 Dataset & Model

| Aspect | Details |
|--------|---------|
| **Dataset** | Kaggle Chest X-Ray (240 training + 60 test) |
| **Model** | 2-layer CNN (16 → 32 filters) |
| **Accuracy** | ~87% |
| **Framework** | TensorFlow/Keras |
| **Training Time** | ~10 minutes (CPU) |

---

## 🚀 Quick Start

### Step 1: Install

### Step 2: Organize Data
python copy_images.py

text

### Step 3: Train Model
python train_model.py

text

**Output:** `pneumonia_model.h5` + `training_results.png`

---

## 📁 Structure

PNEUMONIA_DETECT/
├── copy_images.py # Dataset organizer
├── train_model.py # Training script
├── train/
│ ├── normal/ (120 images)
│ └── pneumonia/ (120 images)
└── test/
├── normal/ (30 images)
└── pneumonia/ (30 images)

text

## 💡 Technologies

Python • TensorFlow • Keras • NumPy • Pandas • Matplotlib


## 🔮 Future Improvements

- Transfer Learning (ResNet) → 95%+ accuracy
- Deploy as Flask web app
- Grad-CAM visualizations
- Multi-class classification

---

## 📚 References

- [Kaggle Dataset](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia)
- [TensorFlow Docs](https://tensorflow.org)
- [CS231n CNN Guide](https://cs231n.github.io/)

---

**Built for learning & placements 🚀**  
GitHub: [@chandu336949](https://github.com/chandu336949)