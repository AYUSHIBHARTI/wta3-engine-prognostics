# WTA³ Ensemble Framework for Prognostics

**Robust Ensemble-Based Framework for Extrapolation and Uncertainty Quantification using Limited Time Series Data**

A Winner-Takes-All³ (WTA³) ensemble framework for Remaining Useful Life (RUL) prediction with calibrated uncertainty quantification, applied to NASA C-MAPSS turbofan engine degradation data.

---

## Overview

This framework addresses a core challenge in prognostics: making reliable RUL predictions with uncertainty bounds when only limited historical data is available. It combines six heterogeneous machine learning models under a MAD-robust inverse-power weighting scheme, repeated across 100 randomised experiments to produce a stable meta-ensemble prediction with a calibrated 90% confidence interval.

**Key results on NASA C-MAPSS FD001 (Engine #52):**

| Metric | Value |
|---|---|
| RMSE | 3.73 cycles |
| MAE | 3.19 cycles |
| R² | 0.9929 |
| NASA Score | 55.25 |
| Within ±10 cycles | 100% |
| CI Coverage (90% nominal) | 100% |

---

## Framework: WTA³

**Six base models:**
- Random Forest (RF)
- XGBoost (XGB)
- Support Vector Regression (SVR)
- LSTM
- 1-D CNN
- Transformer

**Weighting (inverse-power, β = −3):**
```
weight(i) = val_RMSE(i)^β,   normalised over retained models
```

**MAD outlier exclusion (two-sided, threshold = 2.5):**
```
exclude if |val_RMSE(i) − median| > 2.5 × MAD
```

**Meta-ensemble** = arithmetic mean of 100 per-experiment WTA³ predictions

**Uncertainty** = variance-based 90% CI with k90 = 3.04, calibrated on 195,900 validation residuals

---

## Dataset

[NASA C-MAPSS](https://www.nasa.gov/intelligent-systems-division/) FD001–FD004 turbofan engine degradation dataset.

| Dataset | Conditions | Fault Modes | Test Engine | RMSE |
|---|---|---|---|---|
| FD001 | 1 | 1 | #52 | 3.73 |
| FD002 | 6 | 1 | #154 | 5.50 |
| FD003 | 1 | 2 | #52 | 14.51 |
| FD004 | 6 | 2 | #143 | 11.31 |

---

## Feature Engineering (22 features)

| Group | Features | Count |
|---|---|---|
| Raw sensors (top 5 by monotonicity) | sensor_11, sensor_12, sensor_4, sensor_7, sensor_15 | 5 |
| Cycle features | cycle_norm, cycle_squared | 2 |
| Rolling statistics (mean + std + slope) | per sensor, window = 50 | 15 |

- Sequence length for LSTM/CNN/Transformer: 50 timesteps
- Train/val split: 87/12 engines, stratified by lifecycle quartile
- Test engine excluded from all stages including scaler fitting

---

## Repository Structure

```
├── step1_setup.py                  # Data loading, feature engineering, train/val split
├── step3_models.py                 # Model definitions (RF, XGB, SVR, LSTM, CNN, Transformer)
├── step4_training.py               # WTA³ training — MAD weighting, 100 experiments
├── step5_evaluation.py             # Evaluation, NASA score, k90 calibration
├── step6_Visualization_and_Summary.py  # All figures and final_metrics.json
├── app.py                          # Streamlit dashboard (RUL countdown + CI band)
├── task3_ablation.py               # Ablation study
├── task6_representativeness.py     # Engine representativeness analysis
├── data/
│   └── train_FD001.txt             # NASA C-MAPSS FD001 training data
├── results/
│   ├── final_metrics.json          # All final reported metrics
│   ├── all_predictions.json        # Per-experiment predictions
│   ├── aggregate_metrics.json      # Per-experiment RMSE statistics
│   ├── val_calibration_data.json   # k90 calibration data
│   └── figures/                    # All generated figures (fig1–fig10, sensitivity, ablation)
└── requirements.txt
```

---

## Pipeline

Run the steps in order:

```bash
python step1_setup.py                      # Feature engineering + checkpoints
python step3_models.py                     # Model architecture definitions
python step4_training.py                   # Train 100 experiments (takes time)
python step5_evaluation.py                 # Evaluate on test engine
python step6_Visualization_and_Summary.py  # Generate figures + final_metrics.json
```

---

## Streamlit Dashboard

```bash
streamlit run app.py
```

Features: cycle slider, RUL countdown, health status (green/yellow/red), 90% CI band, σ and CI width plots.

---

## Requirements

```bash
pip install -r requirements.txt
```

Main dependencies: `scikit-learn`, `xgboost`, `torch`, `streamlit`, `matplotlib`, `numpy`, `pandas`

---

## Results

### Individual Model Performance (mean over 100 experiments, FD001)

| Model | RMSE | MAE | R² |
|---|---|---|---|
| RF | 7.74 | 5.55 | 0.9678 |
| XGB | 11.68 | 9.76 | 0.8970 |
| SVR | 6.93 | 5.25 | 0.9752 |
| LSTM | 8.19 | 6.62 | 0.9638 |
| CNN | 10.47 | 8.46 | 0.9401 |
| Transformer | 5.13 | 3.85 | 0.9830 |
| **WTA³ Meta-Ensemble** | **3.73** | **3.19** | **0.9929** |

### Robustness (WTA³ vs single best-by-validation model)

| | WTA³ | Single Best |
|---|---|---|
| Worst-case RMSE | 9.54 | 15.04 |
| RMSE std | 1.28 | 2.71 |
| **Worst-case improvement** | **36.6%** | — |

---

## Citation

If you use this work, please cite:

```
Bharti, A. (2026). Robust Ensemble-Based Framework for Extrapolation and
Uncertainty Quantification using Limited Time Series Data for Prognostics.
[Thesis / EWSHM 2026 Conference Paper]
```

---

## License

For academic and research use only. Dataset sourced from NASA Prognostics Center of Excellence.
