# ED-visit-12ML-prediction

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![XGBoost](https://img.shields.io/badge/XGBoost-GPU-red.svg)](https://xgboost.ai)
[![LightGBM](https://img.shields.io/badge/LightGBM-GPU-green.svg)](https://lightgbm.readthedocs.io)
[![CatBoost](https://img.shields.io/badge/CatBoost-GPU-orange.svg)](https://catboost.ai)

> MSc Thesis | Tsinghua University | Vanke School of Public Health
> 
> "Prioritizing sensitivity for high-risk events over overall AUC" — Thesis Proposal Section 3.4

---

## Table of Contents

- [Overview](#-overview)
- [⚠️ Important: Data Access Requirements](#️-important-data-access-requirements)
- [Hardware Requirements](#-hardware-requirements)
- [Software Requirements](#-software-requirements)
- [Installation Guide](#-installation-guide)
- [Clinical Prediction Tasks](#-clinical-prediction-tasks)
- [12 ML Algorithms Implemented](#-12-ml-algorithms-implemented)
- [Feature Engineering](#-feature-engineering-29-admission-available-features)
- [Methodology Highlights](#-methodology-highlights)
- [Output Structure](#-output-structure)
- [Running the Code](#-running-the-code)
- [Expected Performance](#-expected-performance)
- [Troubleshooting](#-troubleshooting)
- [Citation](#-citation)
- [Contact](#-contact)

---

## Overview

This repository contains the complete code for my MSc thesis comparing **12 machine learning algorithms** for predicting clinical outcomes in emergency department patients using **only admission-available data** (≤90 minutes post-arrival).

### The 12 Algorithms:

| # | Algorithm | Type | Acceleration |
|---|-----------|------|--------------|
| 1 | **XGBoost** | Gradient Boosting | ✅ GPU |
| 2 | **LightGBM** | Gradient Boosting | ✅ GPU |
| 3 | **CatBoost** | Gradient Boosting | ✅ GPU |
| 4 | **Random Forest** | Ensemble (Bagging) | CPU (Parallel) |
| 5 | **Logistic Regression** | Linear | CPU |
| 6 | **SVM (LinearSVC)** | Kernel Method | CPU |
| 7 | **FT-Transformer** | Deep Learning (Transformer) | ✅ GPU |
| 8 | **MLP** | Deep Learning (Multi-Layer Perceptron) | CPU |
| 9 | **TabNet** | Deep Learning (Attention) | ✅ GPU |
| 10 | **Self-Normalizing NN** | Deep Learning (SELU) | CPU |
| 11 | **Tabular ResNet** | Deep Learning (Residual) | CPU |
| 12 | **Voting Ensemble** | Ensemble (Soft Voting) | CPU |

The code is fully reproducible and produces thesis-ready outputs including:
- Performance metrics (sensitivity, specificity, AUC-ROC, AUC-PR)
- SHAP interpretability analysis
- Decision Curve Analysis for clinical utility
- Self-contained HTML reports with embedded visualizations

---

## Important: Data Access Requirements

### 🔒 Credentialed Data Access

This code uses the **MC-MED Dataset** (Kansal et al., 2025), which is a **credentialed access dataset** on PhysioNet. You cannot run this code without authorized access to the data.

### 📋 Steps to Obtain Data Access:

1. **Complete CITI Training** (Required by PhysioNet)
   - Go to: https://about.citiprogram.org/
   - Complete **"Data or Specimens Only Research"** course
   - Save your completion certificate

2. **Create PhysioNet Account**
   - Go to: https://physionet.org/
   - Create account using your institutional email

3. **Apply for MC-MED Access**
   - Navigate to: https://physionet.org/content/mc-med/1.0/
   - Click **"Request Access"**
   - Upload your CITI completion certificate
   - Sign the data use agreement

4. **Download the Data** (After Approval)
   - Download all CSV files to `D:\Thesis\Data\` (or modify path in code)
   - Required files: `visits.csv`, `meds.csv`, `pmh.csv`
   - Chronological split files: `split_chrono_train.csv`, `split_chrono_val.csv`, `split_chrono_test.csv`

### 🚫 Important Note:
> **Without authorized data access, the code will not run.** This restriction protects patient privacy and complies with HIPAA Safe Harbor standards. Do not attempt to use alternative data sources without proper authorization.

---

## 💻 Hardware Requirements

### GPU vs CPU Requirements by Algorithm

| Algorithm | GPU Required? | GPU Benefit | CPU Fallback |
|-----------|---------------|-------------|--------------|
| **XGBoost** | Optional | 3-10x faster | ✅ Yes |
| **LightGBM** | Optional | 5-20x faster | ✅ Yes |
| **CatBoost** | Optional | 2-5x faster | ✅ Yes |
| **Random Forest** | No | N/A | ✅ Yes (CPU parallel) |
| **Logistic Regression** | No | N/A | ✅ Yes |
| **SVM (LinearSVC)** | No | N/A | ✅ Yes |
| **FT-Transformer** | **Recommended** | 10-50x faster | ⚠️ Very Slow |
| **MLP** | Optional | 3-10x faster | ✅ Yes |
| **TabNet** | **Recommended** | 10-50x faster | ⚠️ Very Slow |
| **Self-Normalizing NN** | No | N/A | ✅ Yes (SELU optimized) |
| **Tabular ResNet** | No | N/A | ✅ Yes (Residual connections) |
| **Voting Ensemble** | No | N/A | ✅ Yes (inference only) |

### Minimum Hardware Specifications

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **CPU** | 4 cores | 8+ cores |
| **RAM** | 16 GB | 32 GB |
| **Storage** | 10 GB free | 20 GB free |
| **GPU (Optional)** | NVIDIA GTX 1060 (6GB) | RTX 3060+ (8GB+) |
| **CUDA Version** | 11.0+ | 12.0+ |

### CUDA Setup (For GPU Acceleration)

```bash
# Check CUDA version
nvidia-smi

# For XGBoost GPU support, you need:
# - CUDA 11.0 or higher
# - XGBoost >= 1.7.0
# - cuDNN installed

# Install CUDA-aware XGBoost
pip install xgboost --upgrade

# Verify GPU detection (run Cell 0 in notebook)

# Core ML Libraries
xgboost>=2.0.0
lightgbm>=4.0.0
catboost>=1.2.0
scikit-learn>=1.3.0

# Deep Learning
torch>=2.0.0
pytorch-tabnet>=4.0.0

# Self-Normalizing NN & Tabular ResNet (built into PyTorch)
# No additional packages needed beyond torch

# Interpretability
shap>=0.42.0

# Data Processing
numpy>=1.24.0
pandas>=2.0.0

# Visualization
matplotlib>=3.7.0
seaborn>=0.12.0

# Utilities
jupyter>=1.0.0
ipykernel>=6.25.0
tqdm>=4.65.0


pip install -r requirements.txt



git clone https://github.com/Drglazizzo/ED-visit-12ML-prediction.git
cd ED-visit-12ML-prediction

# Citation when you consult this repository
@mastersthesis{Onwurah2026,
  author = {Onyedikachen Ikenna Onwurah},
  title = {Machine Learning–Based Clinical Outcome Prediction in Emergency Care},
  school = {Tsinghua University, Vanke School of Public Health},
  year = {2026},
  address = {Beijing, China}
}


# Dataset citation
@article{Kansal2025,
  author = {Kansal, A. and others},
  title = {MC-MED: A Large-Scale Emergency Department Dataset for Machine Learning},
  journal = {PhysioNet},
  year = {2025},
  doi = {10.13026/xxxx}
}


Disclaimer

This code is for research purposes only. The models are not validated for clinical deployment. Any use of these models in clinical practice requires:

    External validation on local data

    Regulatory approval (FDA, NMPA, etc.)

    Integration with clinical workflows

    Prospective validation studies

The author assumes no responsibility for clinical decisions made using this code.


Email: drglazcode@gmail.com
