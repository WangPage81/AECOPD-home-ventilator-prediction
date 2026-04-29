# AECOPD Prediction from Home Ventilator Data
### Time-Aware Transformer for Binary Classification and Time-to-Event Estimation

> **Paper**: *Time-Aware Transformer for Home-Based AECOPD Prediction and Time-to-Event Estimation*  
> Dongyang Wang, Weihao Qu, Ling Zheng, Jiacun Wang, Haowen Pan  
> CSSE Department, Monmouth University, West Long Branch, NJ, USA

---

## Overview

This repository contains the full implementation of a two-stage deep learning framework for predicting **Acute Exacerbation of COPD (AECOPD)** from continuous home ventilator waveform data.

Patients with severe COPD use home non-invasive ventilators nightly. This system passively monitors the resulting **pressure** and **flow** waveforms — recorded at **5 Hz (one reading every 0.2 seconds)** — to detect imminent deterioration before it requires emergency intervention.

### Two Models

| Model | Data | Task | Best Result |
|---|---|---|---|
| **30-day classification** | All 7 channels, jump-point compressed | Binary risk classification | 0.91 accuracy (128-dim, RF) |
| **7-day two-stage** | Raw pressure + flow only | Classification → time-to-event regression | 1.000 accuracy (64-dim, DT) / MAE 0.87 days |

### Why High-Frequency Data Matters

At 5 Hz, each breath cycle (~4–6 seconds) contains 20–30 readings — enough to reconstruct the full sinusoidal morphology of inspiratory and expiratory flow. At 1 Hz you lose peak shape; at 1 reading/5 seconds you see only one point per breath. The **inspiratory peak, expiratory trough, and timing between them** are precisely the clinical signals used to detect airway obstruction in COPD.

### Why Only Pressure and Flow

Of the seven ventilator channels, pressure and flow are:
- The most dynamically variable (largest signal range per breath)
- The primary waveforms used in clinical waveform interpretation
- Sufficient to derive respiratory rate, tidal volume approximations, and airway resistance estimates
- Computationally tractable at full 5 Hz resolution on GPU

---

## Repository Structure

```
aecopd-home-ventilator-prediction/
│
├── data/
│   ├── raw/                          # NOT committed — keep on Google Drive
│   ├── processed/                    # NOT committed — keep on Google Drive
│   └── sample/
│       └── small_sample_100rows.csv  # Safe demo sample (no PHI)
│
├── models/
│   ├── classification/               # Transformer encoder weights
│   │   ├── last7days_32dim/
│   │   │   └── best_day_embedding_transformer.pth
│   │   ├── last7days_64dim/
│   │   │   └── best_day_embedding_transformer.pth
│   │   └── last7days_128dim/
│   │       └── best_day_embedding_transformer.pth
│   │
│   ├── regression/                   # Time-to-event regression weights
│   │   ├── 32dim/
│   │   │   └── best_model.pth
│   │   ├── 64dim/
│   │   │   └── best_model.pth
│   │   └── 128dim/
│   │       └── best_model.pth
│   │
│   └── 5models/                      # Downstream sklearn classifiers
│       └── last7days/
│           ├── 32dim/
│           │   ├── xgboost/          model.joblib  best_params.json  metrics.json
│           │   ├── random_forest/
│           │   ├── svm/
│           │   ├── logistic_regression/
│           │   ├── decision_tree/
│           │   └── pca_xgboost/
│           ├── 64dim/                # same structure
│           └── 128dim/               # same structure
│
├── notebooks/
│   ├── 01_preprocessing/
│   │   └── step1_cleaning.ipynb
│   ├── 02_classification/
│   │   ├── last7days_32dim.ipynb
│   │   ├── last7days_64dim.ipynb
│   │   └── last7days_128dim.ipynb
│   ├── 03_regression/
│   │   ├── Medical_AI_regression_TTE_32dim.ipynb
│   │   ├── Medical_AI_regression_TTE_64dim.ipynb
│   │   └── Medical_AI_regression_TTE_128dim.ipynb
│   ├── 04_classifier/
│   │   └── last7days_5models_32_64_128dim.ipynb
│   └── 05_explainability/
│       └── step9_attention_analysis.ipynb
│
├── results/
│   ├── classification/
│   │   ├── per_patient_all_splits.csv
│   │   ├── confusion_matrices_32dim.png
│   │   ├── confusion_matrices_64dim.png
│   │   ├── confusion_matrices_128dim.png
│   │   └── model_comparison_barchart.png
│   ├── regression/
│   │   ├── per_patient_test.csv
│   │   ├── scatter_test.png
│   │   ├── per_day_tte_line.png
│   │   └── residuals_test.png
│   └── explainability/
│       ├── attention_rollout.png
│       ├── head_importance_pruning.png
│       ├── head_specialisation.png
│       └── head_specialisation.csv
│
├── paper/
│   ├── main.tex
│   ├── references.bib
│   └── figures/
│       ├── architecture_diagram.png
│       ├── waveform_sample.png
│       └── ablation_plot.png
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

## Dataset

| Split | Patients | Rows (regression) | Label 0 | Label 1 |
|---|---|---|---|---|
| Train | 50 | 32,455,977 | 31 | 19 |
| Val | 15 | 8,778,214 | 10 | 5 |
| Test | 22 | 5,877,474 | 17 | 5 |
| **Total** | **87** | **~47M** | **57** | **29*** |

> *One label-1 patient excluded from regression due to data quality. Classification uses all 30 label-1 patients.

**Label definition:**
- **Label 0** — no acute exacerbation during the monitoring period
- **Label 1** — severe AECOPD requiring ICU-level emergency admission

**Data not included in this repository.** Raw ventilator waveforms contain protected health information and are stored securely on institutional servers. The `data/sample/` folder contains a 100-row anonymised excerpt for code testing only.

---

## Model Architecture

### Time-Aware Transformer Encoder

Each ventilator reading is represented as a token with three fused embeddings:

```
token_i = E_type(sensor_type) + E_value(reading) + Time2Vec(time_from_start)
```

**Time2Vec** (Kazemi et al., 2019) encodes time as one linear trend component plus learnable sinusoidal components — capturing both monotonic drift and periodic respiratory patterns:

```
Time2Vec(t)_j = w_0·t + b_0          if j = 0   (linear trend)
              = sin(w_j·t + b_j)      if j ≥ 1   (periodic patterns)
```

| Hyperparameter | Value |
|---|---|
| Embedding dims | 32 / 64 / 128 |
| Attention heads | 4 |
| Encoder layers | 2 |
| FFN hidden dim | 4 × embed_dim |
| Chunk size | 24,000 rows (~40 min at 5 Hz) |
| Pooling | Masked mean over valid tokens |

### Stage 1 — Classification

Per-day embeddings (days 0–6) are concatenated into a `7 × D` patient feature vector and passed to five downstream classifiers: LR, SVM, DT, RF, XGBoost.

### Stage 2 — Time-to-Event Regression

Label-1 patients receive a regression prediction: days remaining until severe event. Target is log₁(1 + days) scaled to [0, 1]. Loss: MSE. Reported metrics: MAE (days), R².

---

## Results

### Classification — 7-day Raw Pressure + Flow

| Dim | Best model | Test accuracy | Test Acc L1 | Test AUROC |
|---|---|---|---|---|
| 32 | Random Forest | 0.955 | 1.000 | 0.988 |
| **64** | **Decision Tree** | **1.000** | **1.000** | **1.000** |
| 128 | SVM | 0.909 | 0.800 | 0.977 |

> 64-dim Decision Tree achieves perfect discrimination on held-out test patients.  
> 128-dim shows mild overfitting — expected given 87 patients and ~400k parameters.

### Time-to-Event Regression (label-1 patients only)

| Dim | MAE (days) | R² | MSE |
|---|---|---|---|
| 32 | 0.933 | 0.711 | 1.185 |
| 64 | 0.868 | 0.760 | 1.010 |
| 128 | 0.868 | 0.760 | 0.993 |

> On average the model predicts time to ICU admission within **0.87 days** of the true event date.

### 30-day Jump-Point Classification (all 7 channels)

| Configuration | Best classifier | Test accuracy |
|---|---|---|
| time128 | Random Forest | **0.91** |
| time64 | Random Forest | 0.82 |
| time32 | SVM / DT / XGB | 0.82 |
| notime128 | DT | 0.77 |

---

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/aecopd-home-ventilator-prediction.git
cd aecopd-home-ventilator-prediction
pip install -r requirements.txt
```

### Requirements

```
torch>=2.0
scikit-learn>=1.3
xgboost>=2.0
pandas>=2.0
pyarrow
joblib
tqdm
matplotlib
psutil
numpy
```

---

## Running the Notebooks

All notebooks are designed for **Google Colab** with GPU (T4 or A100).

**Recommended order:**

```
01_preprocessing  →  02_classification  →  04_classifier  →  05_explainability
                  →  03_regression      →  05_explainability
```

Mount your Google Drive and update `BASE` path in each notebook:
```python
BASE = "/content/drive/My Drive/YOUR_PATH/Cleaned_Data"
```

**Training time estimates (T4 GPU):**

| Notebook | Epochs | Est. time |
|---|---|---|
| Classification 32dim | ~60 (early stop) | 4–8 hrs |
| Classification 64dim | ~60 | 6–10 hrs |
| Classification 128dim | ~60 | 10–14 hrs |
| Regression 32dim | ~60 | 4–8 hrs |

> Pre-cache the dataset into RAM before training to reduce epoch time by ~3×.  
> See caching cell in each regression notebook.

---

## Explainability (Step 9)

The attention analysis notebook produces three interpretability outputs:

| Output | Description |
|---|---|
| `attention_rollout.png` | Which part of the 7-day window the model attends to most |
| `head_importance_pruning.png` | Which attention heads are critical vs redundant (pruning test) |
| `head_specialisation.png` | Whether each head focuses on early/late readings, flow vs pressure, local vs global patterns |

Based on Abnar & Zuidema (2020) attention rollout and head pruning methodology (Michel et al., 2019).

---

## Citation

If you use this code or dataset structure in your research, please cite:

```bibtex
@inproceedings{wang2025aecopd,
  title     = {Time-Aware Transformer for Home-Based {AECOPD} Prediction
               and Time-to-Event Estimation},
  author    = {Wang, Dongyang and Qu, Weihao and Zheng, Ling and
               Wang, Jiacun and Pan, Haowen},
  booktitle = {Proceedings of MedInfo 2025},
  year      = {2025}
}
```

---

## License

Code: MIT License. Data: not included — subject to institutional data sharing agreements.

---

## Contact

Weihao Qu — `wqu@monmouth.edu`  
CSSE Department, Monmouth University, West Long Branch, NJ 07764, USA
