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

## 📊 Results

* **Best classifier**: LightGBM (macro-F1 = 0.669, Accuracy = 0.706 on 2019 hold-out)
* **Reduced feature set**: 93 predictors (performance comparable to full 188 features)
* **Key ideological dimensions**:

  * **Immigration & multiculturalism** (conservatives vs. progressives)
  * **European integration** (pro-EU left vs. Eurosceptic M5S)
  * **Populism** (threat acknowledged by left, denied by M5S)

---

## Data Availability

The repository includes an **anonymised survey dataset** (2017–2019).
Original data collected by **SWG** and **Rachael Monitoring**.
No personally identifiable information is included.

---

## Authors

* **Luca Pennella** – [@LucaPennella](https://github.com/LucaPennella)
* **Amin Gino Fabbrucci Barbagli** 

---

## License

This project is released under a **Research and Academic Use License (Non-Commercial)**.
You are free to use, share, and adapt the materials for **academic and research purposes only**.
Commercial use is **not permitted**.

See the [LICENSE](./LICENSE) file for details.

---

## 🔗 Citation

If you use this code or dataset, please cite:

```bibtex
@article{pennella2025voting,
  title={Explainable Machine Learning for Predicting Voting Intentions: A Study of Italian Politics},
  author={Pennella, Luca and Fabbrucci Barbagli, Amin Gino},
  journal={International Journal of Data Science and Analytics},
  year={2025},
  publisher={Springer}
}
```

---
