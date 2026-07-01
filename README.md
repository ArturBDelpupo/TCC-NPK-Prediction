# Soil NPK Prediction using Non-Linear Stacking

This repository contains the source code and development files for the Undergraduate Thesis focused on predicting soil macronutrients (Nitrogen, Phosphorus, and Potassium).

## Description

A soil nutrient prediction system for Nitrogen (N), Phosphorus (P), and Potassium (K) using data from the LUCAS Soil 2015 project. The main objective is to evaluate the performance of Non-Linear Stacking techniques to improve the accuracy of regression and classification models applied to remote sensing and climate data.

## Methodology

The development pipeline is structured into the following stages:

- Dataset A: LUCAS 2015 + External data integration (21,527 samples, 704 features).
- Feature Engineering: Temporal indices (NDVI, NDMI, BSI, MODIS) and climate data (ERA5).
- Agronomic Interactions: EC/pH ratio, vegetation vigor, and thermal stress.
- Dimensionality Reduction: PCA and correlation filter (r > 0.95).
- Models Evaluated: Random Forest, Extra Trees, XGBoost, LightGBM, CatBoost, KNN, and MLP.
- Non-Linear Stacking: Meta-model powered by XGBoost.
- Feature Selection: SHAP importance ranking.
- Validation: 10-Fold Cross-Validation (CV) and an 80/20 train/test split.

## Results

The models were evaluated under two main approaches:

1. Regression: Assessed using R-squared, MAE, and RMSE metrics.
2. Classification: Accuracy and Weighted F1-Score (Low, Medium, and High classes).
3. Feature Subsets: Comparative tests performed across different subsets including All, Top 50, Top 25, Top 10, and Top 5 features.

## Technologies and Tools

The project was fully developed in Python, leveraging the following main libraries:

- Machine Learning: Scikit-learn, XGBoost, LightGBM, CatBoost
- Model Explainability: SHAP
- Data Manipulation and Analysis: Pandas, NumPy

## Repository Structure

```text
├── data/              # Public datasets or download instructions
├── notebooks/         # Jupyter Notebooks for exploratory data analysis and testing
├── src/               # Source code (preprocessing scripts, modeling, etc.)
├── requirements.txt   # Project dependencies
└── README.md          # Project documentation
