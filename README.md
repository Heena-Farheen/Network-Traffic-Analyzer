#  UNSW-NB15 Supervised Intrusion Detection System (IDS)

A complete machine learning pipeline for network intrusion detection using the **UNSW-NB15** dataset. This project trains, evaluates, and compares multiple supervised classifiers to distinguish between normal and malicious network traffic, and includes an interactive CLI for real-time packet classification.

---

##  Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Features](#features)
- [Requirements](#requirements)
- [Setup](#setup)
- [Usage](#usage)
- [Pipeline Steps](#pipeline-steps)
- [Models](#models)
- [Outputs](#outputs)
- [Project Structure](#project-structure)

---

## Overview

This project implements a full supervised IDS pipeline:

- Loads and preprocesses the UNSW-NB15 dataset (parquet format)
- Handles class imbalance with **SMOTE**
- Trains and tunes **6 classifiers** using cross-validation and GridSearchCV
- Evaluates models with accuracy, precision, recall, F1, and ROC-AUC
- Builds a **Stacking Ensemble** from the best-performing models
- Optimizes decision thresholds to maximize recall (catching more attacks)
- Plots feature importances, confusion matrices, and ROC curves
- Launches an **interactive CLI** for live packet classification

---

## Dataset

**UNSW-NB15** — A modern network intrusion dataset created by the Australian Centre for Cyber Security (ACCS).

- Download from: [https://research.unsw.edu.au/projects/unsw-nb15-dataset](https://research.unsw.edu.au/projects/unsw-nb15-dataset)
- Expected format: `.parquet` files (training and testing splits)
- Target column: `label` (binary: `0` = Normal, `1` = Attack)

> The `attack_cat` column (multi-class attack category) is dropped during preprocessing but can be used as the target by changing `TARGET_COLUMN = "attack_cat"` in the config section.

---

## Features

-  Automatic missing value imputation (median for numeric, mode for categorical)
-  Duplicate removal
-  StandardScaler + OneHotEncoder via `ColumnTransformer`
-  SMOTE oversampling for imbalanced classes
-  GridSearchCV hyperparameter tuning
-  Stacking Ensemble (RF + GB + LR)
-  Threshold tuning for recall maximization
-  Feature importance visualization (Random Forest)
-  Interactive CLI for predicting individual packets

---

## Requirements

```
Python >= 3.8
numpy
pandas
matplotlib
seaborn
scikit-learn
imbalanced-learn
tqdm
pyarrow          # for reading .parquet files
```

Install all dependencies:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn imbalanced-learn tqdm pyarrow
```

---

## Setup

1. **Clone the repository:**

```bash
git clone https://github.com/your-username/unsw-supervised-ids.git
cd unsw-supervised-ids
```

2. **Download the UNSW-NB15 dataset** and convert to parquet (or use the parquet files directly if available).

3. **Update the file paths** in the configuration section at the top of `unsw_supervised_ids.py`:

```python
TRAIN_PATH = r"path/to/UNSW_NB15_training-set.parquet"
TEST_PATH  = r"path/to/UNSW_NB15_testing-set.parquet"
TARGET_COLUMN = "label"   # or "attack_cat" for multi-class
RANDOM_STATE = 42
```

---

## Usage

Run the full pipeline:

```bash
python unsw_supervised_ids.py
```

At the end of the run, you will be prompted to launch the **interactive prediction CLI**, where you can enter feature values for a network packet and get a real-time NORMAL / MALICIOUS prediction with confidence scores.

---

## Pipeline Steps

| Step | Description |
|------|-------------|
| 1 | Load training and testing data |
| 2 | Preprocess — dedup, impute, scale, encode, SMOTE |
| 3 | Train 6 classifiers with cross-validation |
| 4 | Evaluate and compare all models |
| 5 | Visualize — accuracy chart, confusion matrices, ROC curves |
| 6 | Threshold tuning to maximize recall |
| 7 | Train a Stacking Ensemble |
| 8 | Plot Random Forest feature importances |
| 9 | Extended hyperparameter search (RF + GB) |
| 10 | Interactive CLI for live packet prediction |

---

## Models

| Model | Tuning |
|-------|--------|
| Logistic Regression | Default (max_iter=1000) |
| Decision Tree | Default |
| Naive Bayes (Gaussian) | Default |
| K-Nearest Neighbors | GridSearchCV (k neighbors) |
| Random Forest | GridSearchCV (n_estimators, max_depth) |
| Gradient Boosting | Default + Extended search |
| **Stacking Ensemble** | RF + GB + LR as base; LR as meta |

---

## Outputs

After a successful run, the following files are saved in the working directory:

| File | Description |
|------|-------------|
| `model_comparison_results.csv` | Accuracy, Precision, Recall, F1, ROC-AUC for all models |
| `accuracy_comparison.png` | Bar chart comparing model accuracies |
| `cm_<ModelName>.png` | Confusion matrix for each model |
| `roc_curves.png` | ROC curve comparison (binary classification only) |
| `rf_feature_importance.png` | Top 20 feature importances from Random Forest |

---

## Project Structure

```
unsw-supervised-ids/
│
├── unsw_supervised_ids.py       # Main pipeline script
├── README.md                    # Project documentation
│
├── data/                        # Place your .parquet files here
│   ├── UNSW_NB15_training-set.parquet
│   └── UNSW_NB15_testing-set.parquet
│
└── outputs/                     # Generated after running the script
    ├── model_comparison_results.csv
    ├── accuracy_comparison.png
    ├── cm_*.png
    ├── roc_curves.png
    └── rf_feature_importance.png
```

---

## Notes

- SMOTE is only applied when the minority class is below 40% of the training data.
- Permutation importance (Step 9) is commented out by default — uncomment in the script if needed.
- The interactive CLI uses the **best Extended Random Forest** model for predictions.
- Scikit-learn version compatibility is handled automatically for `OneHotEncoder` (`sparse` vs `sparse_output`).

---

## License

This project is for academic and research purposes. The UNSW-NB15 dataset is provided by the University of New South Wales — please cite appropriately if used in published work.
