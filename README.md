# Explainable Machine Learning for Predicting Voting Intentions: A Study of Italian Politics

[![License: Research-Only](https://img.shields.io/badge/License-Research--Only-lightgrey.svg)](./LICENSE)


This repository contains the code and anonymised data accompanying the paper:

> **Pennella, L., & Fabbrucci Barbagli, A. G.**  
> *Explainable Machine Learning for Predicting Voting Intentions: A Study of Italian Politics*.  
> *International Journal of Data Science and Analytics, 2025.*

---

## Abstract

Understanding voting intentions is a central challenge at the intersection of political science and machine learning.  
This study introduces an **explainable ML framework** that integrates demographic variables with **value-based attitudes** to identify the ideological drivers of Italian electoral behaviour.  

- Dataset: **4,500 survey responses** (2017–2019) from SWG & Rachael Monitoring  
- Methods: **Random Forest, XGBoost, LightGBM** with **Bayesian optimisation**  
- Explainability: **SHAP** values for feature attribution + **Recursive Feature Elimination (RFE)** for model compactness  
- Key findings:
  - Values concerning **globalisation, immigration, and populism** improve predictive accuracy beyond demographics.  
  - Compact models (93 features vs. 188) retain predictive performance.  
  - Coalition-specific profiles emerge for **centre-right, centre-left, and M5S voters**.  

---

## Reproducibility

### Requirements
- Python ≥ 3.9
- Main libraries:
  - `pandas`, `numpy`
  - `scikit-learn`
  - `lightgbm`, `xgboost`
  - `optuna`
  - `shap`
  - `matplotlib`, `seaborn`


