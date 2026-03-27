## Insurance Cost Analysis

A data analysis and machine learning project that explores which factors drive annual medical insurance costs, and builds predictive models to estimate them.

## Overview

This project follows a complete data science workflow applied to a medical insurance dataset with ~2,700 records:

1. **Data Loading & Cleaning** — Loading raw CSV data, assigning headers, replacing missing values
2. **Exploratory Data Analysis (EDA)** — Regression plots, correlation matrices to identify key drivers
3. **Model Development** — Linear Regression (single & multi-variable), Polynomial pipeline
4. **Model Refinement** — Ridge Regression to improve generalization

## Dataset

| Feature | Description | Type |
|---|---|---|
| Age | Age in years | int |
| Gender | 1 = Male, 2 = Female | int |
| BMI | Body Mass Index | float |
| No_of_Children | Number of dependents | int |
| Smoker | 1 = Smoker, 0 = Non-smoker | int |
| Region | US region: NW(1), NE(2), SW(3), SE(4) | int |
| Charges | Annual insurance charges (USD) | float |

**Source:** IBM Developer Skills Network — Data Analysis with Python (Coursera)

## Key Findings

- **Smoking status** is the single strongest predictor of insurance charges
- A **polynomial pipeline** (degree=2) achieves the best R² score (~0.845)
- **Ridge regression** improves generalization on unseen data vs. plain linear regression

## Model Results

| Model | R² Score |
|---|---|
| Linear Regression — Single Variable (Smoker) | ~0.622 |
| Linear Regression — Multi-Variable | ~0.750 |
| Polynomial Pipeline (degree=2) | ~0.845 |
| Ridge Regression (α=0.1) | ~0.676 |
| Ridge + Polynomial Features (degree=2) | ~0.784 |

## Tech Stack

- **Python 3**
- `pandas` — data manipulation
- `numpy` — numerical operations
- `seaborn` / `matplotlib` — data visualization
- `scikit-learn` — machine learning models and pipelines

## Getting Started

```bash
# Clone the repo
git clone https://github.com/your-username/insurance-cost-analysis.git
cd insurance-cost-analysis

# Install dependencies
pip install pandas numpy seaborn matplotlib scikit-learn

# Open the notebook
jupyter notebook insurance_cost_analysis.ipynb
```

> The dataset is loaded directly from a public URL — no local file needed.

## Project Structure

```
insurance-cost-analysis/
├── insurance_cost_analysis.ipynb   # Main analysis notebook
└── README.md
```
