# Gaussian Naive Bayes (GNB) from Scratch

A complete, vectorized Python implementation of a **Gaussian Naive Bayes** classifier built from scratch using only **NumPy**, alongside custom data-splitting and evaluation utilities, behavioral experiments, visualizations, and verification against `scikit-learn`.

---

## 📋 Table of Contents
- [Overview](#overview)
- [Features & Implemented Components](#features--implemented-components)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Team Division of Work](#team-division-of-work)
- [Key Experiments & Findings](#key-experiments--findings)

---

## 🎯 Overview
This project explores the mathematical foundations and practical behavior of Gaussian Naive Bayes[cite: 1]:
- **Bayes' Theorem & Conditional Independence:** Estimating class priors and per-class Gaussian distributions ($\mu_{c,j}, \sigma^2_{c,j}$)[cite: 1].
- **Numerical Stability:** Computing log-posteriors in log-space to prevent floating-point underflow and applying variance smoothing ($\epsilon$) to avoid division by zero[cite: 1].
- **Vectorized Execution:** Performing predictions without Python loops over samples.
- **Model Verification:** Validating learned parameters (`theta_`, `var_`, `class_prior_`) and predicted probabilities against `sklearn.naive_bayes.GaussianNB`.

---

## 🛠️ Features & Implemented Components
All core algorithms and evaluation tools are implemented from scratch without relying on `scikit-learn` helpers:
1. **Stratified Train-Test Split (`train_test_split`):** Shuffles and splits data while preserving exact class proportions (`seed=42`).
2. **Gaussian Naive Bayes Classifier (`GaussianNB`):**
   - `fit(X, y)`: Computes class priors, per-feature means, and smoothed variances[cite: 1].
   - `predict_proba(X)` & `predict(X)`: Fully vectorized log-likelihood computation and class prediction[cite: 1, 3].
   - **Sanity Check:** Hand-calculated verification on a synthetic 2-class, 1-feature dataset[cite: 2].
3. **Evaluation Metrics:**
   - `accuracy(y_true, y_pred)`
   - `confusion_matrix(y_true, y_pred, classes)` with Matplotlib heatmap visualizations
   - `precision_recall_f1(y_true, y_pred, classes)` (per-class and macro-averaged, safely handling zero-division)[cite: 2]
4. **Experiments & Visualizations:**
   - **Feature Duplication Experiment:** Measuring accuracy vs. prediction confidence when duplicating informative features (violating the naive independence assumption)[cite: 2].
   - **Feature Scaling Analysis:** Comparing model performance with and without standardization[cite: 2].
   - **Distribution & Decision Boundary Plots:** Visualizing fitted 1D Gaussian curves per class and curved 2D decision boundaries on a mesh grid[cite: 2].

---

## 📊 Dataset
- **Breast Cancer Wisconsin Dataset** (loaded via `sklearn.datasets.load_breast_cancer()`)[cite: 1]
  - **Samples:** 569
  - **Features:** 30 numerical features
  - **Classes:** 2 (212 Malignant, 357 Benign)[cite: 1]

---

## 📁 Project Structure
```text
├── GNB_ID1_ID2.ipynb    # Main executable Jupyter Notebook with code, plots, and explanations
├── GNB_ID1_ID2.pdf      # Final report (≤ 4 pages) containing figures, metric tables, and theoretical answers
└── README.md            # Project documentation
