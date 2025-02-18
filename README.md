# AI-Image-Detector
STAD68: Advanced Machine Learning/Deep Learning Final Project

## 🚀 Project Overview
This project is part of the STAD68 course at the University of Toronto, focusing on **AI-generated image detection**. Our goal is to develop a **deep learning ensemble model** that accurately classifies images as either AI-generated or human-created.

We implement a **Late Fusion ensemble** combining CNN and Transformer architectures to improve classification performance.

## 📂 Dataset
The dataset consists of labeled images:  
- **AI-generated images** (labeled as `1`)  
- **Human-created images** (labeled as `0`)  

### 🔹 Challenges
- **Class imbalance** → AI-generated images may be underrepresented.
- **Style variability** → Human-created images have diverse textures and structures.
- **Risk of overfitting** → The model might memorize patterns instead of generalizing.

### 🔹 Preprocessing
- **Resizing based on model requirements**:
  - EfficientNet-B4 → `380×380`
  - ResNet-50 → `224×224`
  - Swin Transformer → `256×256`
- **Normalization**
- **Data augmentation** → MixUp, CutMix, affine transformations

## 🏗️ Model Architecture
We use a **Late Fusion ensemble** combining multiple architectures:
- **EfficientNet-B4** → Captures fine texture details
- **ResNet-50** → Extracts structural features
- **Swin Transformer** → Detects global consistency  

The final prediction is obtained using **Weighted Averaging**:
\[
f_{\text{final}}(x) = w_1 f_{\text{EfficientNet}}(x) + w_2 f_{\text{ResNet}}(x) + w_3 f_{\text{Swin Transformer}}(x)
\]
where the weights are dynamically optimized.

## 🛠️ Optimization Strategies
### 🔹 Individual Model Optimization
- **EfficientNet-B4** → Transfer learning, MixUp, Cosine Annealing
- **ResNet-50** → Batch Normalization, Gradient Clipping, Focal Loss
- **Swin Transformer** → AdamW, Warmup Scheduling, Attention Heatmaps

### 🔹 Ensemble Model Optimization
- **Weighted Averaging** → Dynamic weight assignment
- **Bayesian Optimization** → Fine-tune ensemble weights
- **Stacking Model** → Logistic regression on individual model outputs
- **Cross-Validation** → K-fold validation for stability

## 📊 Evaluation Metrics
- **AUC-ROC** → Measures classification ability (higher is better)
- **F1-Score** → Balances Precision and Recall (higher is better)

## 🚀 Installation & Usage
### 1️⃣ Clone the repository
```bash
git clone https://github.com/your-username/AI-Image-Detector.git
cd AI-Image-Detector
