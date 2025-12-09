# **Wound Classification with JAX / Flax | 使用 JAX / Flax 的创伤图像分类项目**

---

## **📌 Overview | 项目简介**

This project implements **multiple deep-learning models for wound image classification**, built using **JAX** and **Flax**.
The system includes:

* Traditional CNN models
* ResNet18 / ResNet34
* Vision Mamba (a state-space model for vision tasks)
* Hybrid Mamba + CNN
* Hybrid Mamba + ResNet
* Full training and evaluation pipelines
* Dataset cleaning, augmentation, splitting, and loading utilities

本项目实现了一个 **基于 JAX / Flax 的伤口图像分类任务**，提供：

* 传统 CNN 模型
* ResNet18 / ResNet34
* Vision Mamba（视觉状态空间模型）
* 混合 Mamba + CNN
* 混合 Mamba + ResNet
* 训练与推理全流程脚本
* 数据清洗、增强、划分、加载工具

---

## **📁 Project Structure | 项目结构**

```text
WOUND_CLASSIFICATION_JAX
│  requirements.txt
│  terminal_commands.txt
│
├─data
│   └─dataset                 # Cleaned dataset (after processing)
│
├─nets                        # Model architectures
│   └─ BaselineCNN.py
│      CNN.py
│      Hybrid.py
│      Mamba.py               # The implementations of Vision Mamba and the VisionMamba.py file are different.
│      ResNet.py
│      VisionMamba.py
│
├─references
│   └─ Hatamizadeh_MambaVision_CVPR2025.pdf
│
└─scripts                     # Training / Testing / Data Processing
    └─ dataset.py
       data_clean.py
       download_data.py
       test.py
       train.py
```

---

# **🚀 Features | 功能特点**

### **1. Multiple Model Architectures 多模型支持**

* ✔ SimpleCNN（CNN.py）
* ✔ BaselineCNN
* ✔ ResNet18 / ResNet34
* ✔ Vision Mamba
* ✔ Hybrid Mamba + CNN
* ✔ Hybrid Mamba + ResNet

### **2. Data Processing Toolkit 数据处理工具**

* Automatic corruption detection 自动检测损坏图片
* Dataset cleaning & renaming 数据集清洗与重新命名
* Train/Test splitting 自动划分训练/测试集
* On-the-fly augmentation 在线增强（旋转、亮度、对比度、模糊等）

### **3. Full Training Pipeline 完整训练流程**

* Train/eval steps with BatchNorm/Dropout
    * Learning rate, dropout, optimizer configurable
* Checkpoint saving & loading
* Gradient clipping

### **4. Inference / Evaluation 推理与评估**

* Load checkpoint and evaluate on test set
* Supports all model types
* Outputs accuracy and loss metrics

---

# **📦 Installation | 安装**

```bash
pip install -r requirements.txt
```

---

# **📂 Dataset Preparation | 数据准备**

### **Download from Kaggle**

```bash
python scripts/download_data.py
```

### **Clean dataset and remove corrupted images**

```bash
python scripts/data_clean.py
```

This creates:

```text
data/dataset/
    000001_ClassA.jpg
    000002_ClassB.jpg
```

### **Split into train/test**

```bash
python scripts/data_clean.py --build_split
```

This generates:

```text
data/dataset_split/train/
data/dataset_split/test/
```

---

# **🧠 Model Training | 模型训练**

Example:

```bash
python scripts/train.py \
    --model mamba \
    --batch_size 16 \
    --num_epochs 50 \
    --learning_rate 5e-5 \
    --use_augmentation True
```

### **Supported models (choose with `--model`)**

```text
cnn
baseline_cnn
resnet18
resnet34
mamba
vision_mamba
hybrid_mamba_cnn
hybrid_mamba_resnet
```

---

# **🧪 Model Testing / Evaluation | 模型测试与评估**

Example:

```bash
python scripts/test.py \
    --model mamba \
    --ckpt_path ../checkpoints/mamba/best.pkl
```

Outputs:

* Accuracy (准确率)
* Loss (损失值)
* Batch-wise prediction statistics (批次预测统计)

---

# **🧩 Key Files | 关键文件说明**

| File                    | Description                               |
|-------------------------|-------------------------------------------|
| `scripts/train.py`      | Full training pipeline（训练主脚本）             |
| `scripts/test.py`       | Inference and evaluation（推理评估脚本）          |
| `scripts/dataset.py`    | Dataset loader + augmentation（数据加载器 + 增强） |
| `scripts/data_clean.py` | Clean dataset and split（数据清洗与划分）          |
| `nets/`                 | All neural network architectures（所有网络结构）  |

---

# **💡 Model Highlights | 模型亮点**

### **Vision Mamba**

Implements Mamba state-space blocks for vision tasks, including:

* Patch embedding
* Conv + SSM dual-branch encoder
* SwiGLU feed-forward
* Positional embeddings
* Optional class token

### **Hybrid Models**

Fuse Mamba features with CNN/ResNet outputs:

* Weighted sum 融合
* Gated sum 门控融合
* Concatenation + MLP 连接头

---

# **▶ Example Code | 示例代码**

### **Loading an image manually**

```python
from scripts.dataset import data_loader

loader = data_loader(data_path="../data/dataset", use_augmentation=True)

img, label_idx, img_idx = loader[0]
```

### **Running a forward pass**

```python
from nets.CNN import SimpleCNN
import jax
from scripts.dataset import data_loader

loader = data_loader(data_path="../data/dataset", use_augmentation=False)
model = SimpleCNN(num_classes=loader.num_classes)
params = model.init(jax.random.PRNGKey(0), jax.numpy.zeros((1, 224, 224, 3)))
logits = model.apply(params, jax.numpy..zeros((1, 224, 224, 3)))
```

---

# **📌 Requirements | 依赖**

See `requirements.txt`.

---

# **📄 License | 许可证**

MIT License or your preferred license.
