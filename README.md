# Soil NPK Prediction using Non-Linear Stacking

This repository contains the source code and development files for the Master's Thesis focused on predicting soil macronutrients (Nitrogen, Phosphorus, and Potassium).

## Description

A soil nutrient prediction system for Nitrogen (N), Phosphorus (P), and Potassium (K) using data from the LUCAS Soil 2015 project.

The main objective is to evaluate the performance of Non-Linear Machine Learning methods and Stacking techniques to improve the accuracy of regression and classification models applied to remote sensing and climate data.

## Methodology

The development pipeline is structured into the following stages:

- Dataset A: LUCAS 2015 + External data integration (21,527 samples, 704 features).
- Feature Engineering: Temporal indices (NDVI, NDMI, BSI, MODIS) and climate data (ERA5).
- Agronomic Interactions: EC/pH ratio, vegetation vigor, and thermal stress.
- Dimensionality Reduction: PCA and correlation filter (r > 0.95).
- Models evaluated: Random Forest, Extra Trees, XGBoost, LightGBM, CatBoost, KNN, and MLP.
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
- Data Manipulation and Analysis: Pandas, NumPy, Rasterio, Pyproj, EarthEngine-API

## Repository Structure

```text
├── data/              # Public datasets or download instructions
├── src/               # Source code modules
│   ├── gee_extractor_lucas2015.py  # Google Earth Engine data extractor
│   ├── enricher_soilgrids.py       # Direct cloud raster read from SoilGrids
│   ├── juntos_teste_dataset.py     # Dataset merging and RFE-SHAP mapping
│   ├── kfold_geral.py              # Baseline loop using K-Fold with KNN and MLP
│   ├── cod_tcc.py                  # Code to evaluate ML, clean data and export figures
├── requirements.txt   # Project dependencies
└── README.md          # Project documentation
```

## How to Run

### 1. Clone the repository

```bash
git clone https://github.com/ArturBDelpupo/TCC-NPK-Prediction.git
cd TCC-NPK-Prediction
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3.Dataset

### Building Your Own Dataset

If you would like to recreate the dataset used in this research, follow the steps below.

### 3.1. Request access to the LUCAS Topsoil 2015 dataset

Request access through the official European Commission page:

https://esdac.jrc.ec.europa.eu/content/lucas2015-topsoil-data

### 3.2. Generate the dataset

Execute the following scripts **in the presented order**:

```text
1. gee_extractor_lucas2015.py
2. enricher_soilgrids.py
3. juntos_teste_dataset.py
```

These scripts will extract and integrate the remote sensing, climate, and soil data required to build the final dataset.

### Dataset Access

If you prefer not to recreate the dataset from scratch, you may request access to the complete dataset using the following link:

https://drive.google.com/file/d/1ef3T0Nn4_CrM7ZOKqGZ8GOUPwyTKH00o/view?usp=sharing

**Please note:** Access to the dataset is subject to approval by the repository owner and may not be granted immediately.

For any questions regarding the dataset, methodology, or repository, please contact:

**Email:** arturdelpupo.pt@gmail.com

### 4. Evaluate the models

Run the following scripts:

```text
1. kfold_geral.py
2. cod_tcc.py
```

The first script performs the baseline K-Fold evaluation, while the second executes the complete pipeline, including feature selection, model training, evaluation, and figure generation.


## Academic Use

This repository and the research presented herein are intended **exclusively for academic, educational, and research purposes**.

Commercial use, redistribution for commercial purposes, or incorporation into proprietary products without prior written permission from the author is **not permitted**.

If you use this work in your research or academic publications, please cite it using the following reference:

```bibtex
@mastersthesis{ArturD2026,
  author = {Artur B. Delpupo},
  title  = {Soil NPK Prediction using Non-Linear Stacking},
  school = {IPB/UTFPR},
  year   = {2026},
  type   = {Master's Thesis}
}
```
