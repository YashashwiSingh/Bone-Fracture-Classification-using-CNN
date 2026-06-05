<div align="center">

# 🦴 Bone Fracture Classification
### *Automated X-Ray Diagnosis using Deep Learning*

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![Keras](https://img.shields.io/badge/Keras-ResNet50-D00000?style=for-the-badge&logo=keras&logoColor=white)](https://keras.io)
[![Kaggle](https://img.shields.io/badge/Dataset-Kaggle-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)](https://kaggle.com)
[![License](https://img.shields.io/badge/License-MIT-22C55E?style=for-the-badge)](LICENSE)

<br/>

> **Classifying bone fractures from X-ray images into Simple vs. Comminuted types**  
> using ResNet50 Transfer Learning, two-phase fine-tuning, and Grad-CAM interpretability.

<br/>

| 🎯 Accuracy | 📈 AUC | 🏆 Best val_AUC | ⚡ Macro F1 |
|:-----------:|:------:|:---------------:|:-----------:|
| **70.78%** | **0.7843** | **0.8186** | **0.71** |

</div>

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Dataset](#-dataset)
- [Architecture](#-model-architecture)
- [Training Strategy](#-training-strategy)
- [Results](#-results)
- [Grad-CAM Visualisation](#-grad-cam-visualisation)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
- [Configuration](#-configuration--hyperparameters)
- [Recommendations](#-recommendations-for-improvement)
- [Tech Stack](#-tech-stack)

---

## 🔬 Overview

This project develops a **Convolutional Neural Network** pipeline for automated classification of bone fractures from X-ray images. The model distinguishes between two clinically distinct fracture types:

| Fracture Type | Description |
|---------------|-------------|
| 🟦 **Simple (Non-displaced)** | Clean break with bone fragments remaining aligned |
| 🟥 **Comminuted** | Bone shattered into three or more fragments — more severe |

### Project Objectives

- ✅ Build and evaluate a CNN model for binary fracture classification from X-ray images
- ✅ Preprocess and augment X-ray images to improve generalisation
- ✅ Compare training and validation metrics across accuracy, loss, AUC, precision, and recall
- ✅ Reduce overfitting using Dropout regularisation and data augmentation
- ✅ Evaluate on unseen test data with confusion matrix and classification report
- ✅ Visualise model attention regions using Grad-CAM

---

## 📦 Dataset

**Source:** [Kaggle — Simple vs. Comminuted Fractures X-Ray Data](https://www.kaggle.com/datasets/orvile/simple-vs-comminuted-fractures-x-ray-data)

```python
import kagglehub
path = kagglehub.dataset_download('orvile/simple-vs-comminuted-fractures-x-ray-data')
```

| Property | Detail |
|----------|--------|
| **Classes** | Simple Bone Fracture, Comminuted Bone Fracture |
| **Image Format** | JPEG / PNG |
| **Input Resolution** | 224 × 224 pixels (resized) |
| **Primary Subset** | Augmented (more images, better generalisation) |

### Data Split — 80 / 10 / 10

```
fracture_split/
├── train/          # 80% — Model parameter optimisation
│   ├── Simple Bone Fracture/
│   └── Comminuted Bone Fracture/
├── val/            # 10% — Hyperparameter tuning & early stopping
│   ├── Simple Bone Fracture/
│   └── Comminuted Bone Fracture/
└── test/           # 10% — Final unbiased evaluation
    ├── Simple Bone Fracture/
    └── Comminuted Bone Fracture/
```

> **Note:** The dataset folder contains a typo — `Orignal` instead of `Original`. This is handled programmatically with case-insensitive path matching.

---

## 🏗 Model Architecture

**Backbone:** ResNet50 pre-trained on ImageNet (`include_top=False`)

```
Input (224 × 224 × 3)
        ↓
ResNet50 Backbone (frozen in Phase 1)
        ↓  [7 × 7 × 2048]
GlobalAveragePooling2D
        ↓  [2048]
BatchNormalization
        ↓
Dense(256, ReLU)  +  L2 reg=1e-4
        ↓
Dropout(0.5)
        ↓
Dense(128, ReLU)  +  L2 reg=1e-4
        ↓
Dropout(0.25)
        ↓
Dense(1, Sigmoid)   →   P(Comminuted)
```

| Setting | Value | Rationale |
|---------|-------|-----------|
| Optimiser | Adam (lr=1e-4) | Adaptive learning rate |
| Loss | Binary Cross-Entropy | Standard for binary classification |
| Metrics | Accuracy, AUC, Precision, Recall | Comprehensive evaluation |

---

## 🎓 Training Strategy

### Phase 1 — Head Training (Backbone Frozen)

The ResNet50 backbone is entirely frozen. Only the custom classification head is trained, rapidly learning task-specific features without disrupting ImageNet weights.

| Setting | Value |
|---------|-------|
| Epochs | 15 (with early stopping) |
| Learning Rate | 1e-4 |
| Trainable Layers | Custom head only |

### Phase 2 — Fine-Tuning (Last 30 Layers Unfrozen)

The last 30 layers of ResNet50 are unfrozen and trained at a significantly lower learning rate, adapting high-level features to the X-ray domain.

```python
base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False

model.compile(optimizer=Adam(learning_rate=LR / 10), ...)
```

| Setting | Value |
|---------|-------|
| Epochs | 30 (with early stopping) |
| Learning Rate | 1e-5 (LR / 10) |
| Unfrozen Layers | Last 30 of ResNet50 |
| Best val_AUC achieved | **0.8186** at Epoch 30 |

### Callbacks

| Callback | Configuration | Purpose |
|----------|--------------|---------|
| `EarlyStopping` | monitor=val_auc, patience=7 | Stops training on plateau; restores best weights |
| `ReduceLROnPlateau` | monitor=val_loss, factor=0.3, patience=3 | Reduces LR by 70% on stagnation |
| `ModelCheckpoint` | monitor=val_auc, save_best_only=True | Saves best model weights |

### Data Augmentation

Applied only to the training set to prevent data leakage:

| Parameter | Value |
|-----------|-------|
| Rotation | ±30° |
| Zoom | 20% |
| Brightness | ±20% |
| Horizontal / Vertical Flip | ✅ |
| Width / Height Shift | 10% |
| Fill Mode | nearest |

---

## 📊 Results

### Test Set Performance

| Metric | Score |
|--------|-------|
| **Accuracy** | 70.78% |
| **AUC** | 0.7843 |
| **Macro F1** | 0.71 |
| **Best val_AUC** | 0.8186 |

### Classification Report (Default Threshold = 0.50)

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Comminuted Bone Fracture | 0.79 | 0.63 | 0.70 | 737 |
| Simple Bone Fracture | 0.65 | 0.80 | 0.72 | 632 |
| **Macro Average** | **0.72** | **0.71** | **0.71** | **1369** |
| **Weighted Average** | **0.72** | **0.71** | **0.71** | **1369** |

### Key Observations

- 📌 **AUC vs Accuracy Gap** — AUC of 0.78 with 71% accuracy indicates good discriminative ability, but the 0.5 threshold is suboptimal
- 📌 **Class Asymmetry** — Higher recall for Simple fractures (0.80) vs Comminuted (0.63); the model finds comminuted fractures harder to detect — clinically the more critical class
- 📌 **Threshold Optimisation** — Lowering the decision threshold to 0.44 (optimal F1) improves sensitivity for Comminuted detection

---

## 🔥 Grad-CAM Visualisation

Gradient-weighted Class Activation Mapping (Grad-CAM) visualises which X-ray regions drive the model's predictions. Gradients of the class score are computed with respect to the final convolutional layer (`conv5_block3_out`) and overlaid as a heatmap (60/40 blend).

```python
LAST_CONV = 'conv5_block3_out'   # ResNet50 final conv layer

grad_model = tf.keras.Model(
    inputs=model.inputs,
    outputs=[model.get_layer(LAST_CONV).output, model.output]
)

with tf.GradientTape() as tape:
    conv_out, preds = grad_model(img_array)
    class_score = preds[:, 0]

grads = tape.gradient(class_score, conv_out)
pooled = tf.reduce_mean(grads, axis=(0, 1, 2))
heatmap = conv_out[0] @ pooled[..., tf.newaxis]
overlay = 0.6 * original_img + 0.4 * heatmap_colored
```

> 🟥 **Hot regions (red/yellow)** indicate areas of high model attention — confirmed to align with clinically relevant bone fracture zones, not image artifacts.

---

## 📁 Project Structure

```
bone-fracture-classification/
│
├── Bone_Fracture_Classification_using_CNN.ipynb   # Main notebook
├── bone_fracture_classification_report.pdf        # Full project report
│
├── outputs/
│   ├── bone_fracture_classifier.h5   # Full trained model (weights + architecture)
│   ├── best_model.h5                 # Best checkpoint weights (by val_auc)
│   ├── training_history.png          # Training/validation metric curves
│   ├── confusion_matrix.png          # Confusion matrices at both thresholds
│   └── gradcam.png                   # Grad-CAM activation visualisations
│
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install tensorflow kagglehub numpy matplotlib seaborn scikit-learn opencv-python reportlab
```

### Run in Google Colab

1. Open `Bone_Fracture_Classification_using_CNN.ipynb` in [Google Colab](https://colab.research.google.com/)
2. Set up your Kaggle API key:
   ```python
   import os
   os.environ['KAGGLE_USERNAME'] = 'your_username'
   os.environ['KAGGLE_KEY'] = 'your_api_key'
   ```
3. Run all cells sequentially — dataset download, training, and evaluation are fully automated

### Run Locally

```bash
git clone https://github.com/your-username/bone-fracture-classification.git
cd bone-fracture-classification
jupyter notebook Bone_Fracture_Classification_using_CNN.ipynb
```

---

## ⚙️ Configuration & Hyperparameters

All key hyperparameters are defined in a single configuration cell for easy reproducibility:

```python
IMG_SIZE     = (224, 224)   # ResNet50 required input
BATCH_SIZE   = 32           # Mini-batch size
EPOCHS       = 30           # Max epochs per phase
LR           = 1e-4         # Initial Adam learning rate
SEED         = 42           # Reproducibility seed
MODEL_NAME   = 'ResNet50'   # Switchable to VGG16
DROPOUT_1    = 0.5          # First dropout layer
DROPOUT_2    = 0.25         # Second dropout layer
L2_REG       = 1e-4         # L2 regularisation on Dense layers
```

---

## 💡 Recommendations for Improvement

| Strategy | Details |
|----------|---------|
| 🎯 **Threshold Optimisation** | Use optimal F1 threshold (~0.44) instead of 0.5 |
| ⚖️ **Class Weighting** | Apply `class_weight` to `model.fit()` to address class imbalance |
| 🔓 **Deeper Fine-Tuning** | Unfreeze last 50 layers instead of 30 for more domain adaptation |
| 🔁 **Longer Training** | Increase EPOCHS to 50 and Phase 2 LR to LR/5 |
| 🧠 **Alternative Backbone** | Try `EfficientNetB3` or `DenseNet121` — strong performers on medical imaging |
| 📊 **More Data** | Medical models typically need 5,000+ images per class for >90% accuracy |

---

## 🛠 Tech Stack

| Tool | Purpose |
|------|---------|
| ![Python](https://img.shields.io/badge/-Python-3776AB?logo=python&logoColor=white&style=flat) | Core language |
| ![TensorFlow](https://img.shields.io/badge/-TensorFlow-FF6F00?logo=tensorflow&logoColor=white&style=flat) | Deep learning framework |
| ![Keras](https://img.shields.io/badge/-Keras-D00000?logo=keras&logoColor=white&style=flat) | Model building & training |
| ![NumPy](https://img.shields.io/badge/-NumPy-013243?logo=numpy&logoColor=white&style=flat) | Numerical computation |
| ![Matplotlib](https://img.shields.io/badge/-Matplotlib-11557C?style=flat) | Plotting & visualisation |
| ![scikit-learn](https://img.shields.io/badge/-scikit--learn-F7931E?logo=scikit-learn&logoColor=white&style=flat) | Metrics & data splitting |
| ![OpenCV](https://img.shields.io/badge/-OpenCV-5C3EE8?logo=opencv&logoColor=white&style=flat) | Grad-CAM image overlay |
| ![Kaggle](https://img.shields.io/badge/-Kaggle-20BEFF?logo=kaggle&logoColor=white&style=flat) | Dataset source |
| ![Google Colab](https://img.shields.io/badge/-Google%20Colab-F9AB00?logo=googlecolab&logoColor=white&style=flat) | Development environment |

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

<div align="center">

Made with ❤️ by **Yashashwi Singh**

[![LinkedIn](https://img.shields.io/badge/LinkedIn-yashashwi--singh-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://linkedin.com/in/yashashwi-singh)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com)

*Report generated: March 27, 2026 | Model: ResNet50 | Framework: TensorFlow/Keras*

</div>
