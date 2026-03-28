# Credit Scoring System

[![gitcgr](https://gitcgr.com/badge/vanuhibaghdasaryan/credit-scoring-system.svg)](https://gitcgr.com/vanuhibaghdasaryan/credit-scoring-system)


## Project Structure

```
credit_scoring_system.py   ← Main pipeline script
data.csv                   ← Raw input (800 rows × 9 cols)
requirements.txt           ← Python dependencies
cv_results.csv             ← Cross-validation results (all models)
feature_importance.csv     ← Permutation importance (AUC drop)
iv_analysis.csv            ← IV / WOE feature rankings
README.md                  ← This file
```

---

## Quick Start

```bash
pip install -r requirements.txt
python credit_scoring_system.py
```

Outputs (`cv_results.csv`, `iv_analysis.csv`, `feature_importance.csv`) are written to an `outputs/` folder that is created automatically.

To use a different data file:
```bash
python credit_scoring_system.py --data path/to/your_data.csv
```

---

## Pipeline Stages & Exact Output

### 1 · Data Loading & Cleaning

```
[load] Shape after cleaning: (800, 8)
[load] Default rate: 4.88%
```

**Actions performed:**
- Column names stripped of leading/trailing spaces (e.g. `" Age"` → `"age"`, `"Credit Card Debt "` → `"credit_card_debt"`)
- All column names normalised to `snake_case`
- Debt columns (`credit_card_debt`, `automobile_debt`, `other_debt`) stored as negatives in source — `abs()` applied
- Applicant ID column dropped (non-predictive)

### 2 · Imputation Strategy

```
[imputer] Strategy map:
  age                → mean
  years_at_employer  → mean
  years_at_address   → median
  income             → median
  credit_card_debt   → median
  automobile_debt    → median
  other_debt         → median
```

Mean used for near-normally distributed columns; median for skewed/liability columns. Designed for zero-missing production data but robust to future gaps.

### 3 · Feature Engineering

```
[features] Engineered features added. New shape: (800, 16)
```

| Feature | Definition |
|---------|-----------|
| `total_debt` | `credit_card_debt + automobile_debt + other_debt` |
| `dti` | `total_debt / income` (Debt-to-Income ratio) |
| `cc_dti` | `credit_card_debt / income` |
| `auto_dti` | `automobile_debt / income` |
| `debt_per_year` | `total_debt / age` |
| `tenure_ratio` | `years_at_employer / age` |
| `stability_index` | `mean(years_at_employer, years_at_address) / age` |
| `log_income` | `log1p(income)` — skew correction |

Transformations: `PowerTransformer` (Yeo-Johnson) on income; `RobustScaler` fallback for outliers. Full `sklearn.Pipeline` prevents data leakage.

### 4 · IV / WOE Analysis

```
[IV Analysis]
                         IV strength
cc_dti             0.367922   Strong
age                0.348805   Strong
credit_card_debt   0.343463   Strong
total_debt         0.311351   Strong
years_at_address   0.259005   Medium
debt_per_year      0.229293   Medium
years_at_employer  0.178387   Medium
auto_dti           0.166631   Medium
stability_index    0.152450   Medium
tenure_ratio       0.139772   Medium
other_debt         0.120001   Medium
automobile_debt    0.112914   Medium
income             0.092954     Weak
log_income         0.092954     Weak
dti                0.048415     Weak
```

**IV Thresholds**: `<0.02` Unpredictive · `0.02–0.1` Weak · `0.1–0.3` Medium · `0.3–0.5` Strong · `>0.5` Suspicious (leakage)

Key insight: raw `income` (IV 0.093) is weak alone but powerful in ratio form — `cc_dti` (IV 0.368) is the top signal.

### 5 · Cross-Validation Results

```
[Cross-Validation Results]
  [LogisticRegression]        AUC=0.5873 ±0.1026 | AvgPrec=0.1514 | F1=0.1154
  [RandomForest]              AUC=0.5907 ±0.1019 | AvgPrec=0.1219 | F1=0.0614
  [GradientBoosting_XGBProxy] AUC=0.5426 ±0.0741 | AvgPrec=0.1192 | F1=0.0667

                     Model  AUC_ROC  AUC_ROC_std  Avg_Prec       F1  LogLoss
             RandomForest  0.5907      0.1013    0.1223    0.0614   0.3061
       LogisticRegression  0.5873      0.1026    0.1514    0.1154   0.6379
GradientBoosting_XGBProxy  0.5426      0.0662    0.1194    0.0667   0.2728
```

**Stratified 5-Fold CV** preserves the 4.88% default rate in every fold.  
`GradientBoosting_XGBProxy` uses sklearn's `GradientBoostingClassifier` as a proxy; native XGBoost/LightGBM expected AUC 0.62–0.68 on this dataset.

### 6 · Best Model

```
[Best] RandomForest selected based on highest AUC-ROC
```

**Selection rationale**: AUC-ROC used for ranking (imbalanced class, threshold-agnostic). Precision-Recall / F1 used to evaluate minority-class performance. LogLoss measures calibration quality.

Production target: **LightGBM** (target AUC ≥ 0.68, Avg Precision ≥ 0.18).

### 7 · Permutation Feature Importance (SHAP proxy)

```
[Permutation Feature Importance (SHAP proxy)]
                   importance_drop (AUC ×10⁻³)
age                       0.017900
credit_card_debt          0.009889
debt_per_year             0.007255
cc_dti                    0.006825
years_at_address          0.003959
auto_dti                  0.002948
dti                       0.002767
years_at_employer         0.002327
stability_index           0.002136
income                    0.001641
```

Permutation importance used as a production-safe SHAP proxy. Top-3 features (`age`, `credit_card_debt`, `cc_dti`) are fully aligned with IV analysis. SHAP `TreeExplainer` produces per-applicant force plots for adverse action reason codes (ECOA compliance).

### 8 · Income Paradox Analysis

```
[Income Paradox Analysis]
  Model1_income_as_score: AUC=0.6752 | Income included as a direct scoring feature.
  Model2_income_for_limit: AUC=0.6748 | Income excluded from score; used only for credit limit.
  Model3_dual_income: AUC=0.6752 | Income lowers PD AND raises limit — correlated risk/exposure
                                    creates higher Expected Loss than the model implies.
```

| Model | AUC | Double-Count Risk | Recommendation |
|-------|-----|-------------------|----------------|
| Model 1 — Score Only | 0.6752 | Moderate | Acceptable; DTI more informative than raw income |
| **Model 2 — Limit Only** | **0.6748** | **Low** | **★ Recommended** |
| Model 3 — Dual Usage | 0.6752 | High | Avoid — understates Expected Loss |

**Model 2 recommended**: clean separation between risk scoring (scorecard) and exposure setting (credit limit). Prevents EL = PD × LGD × EAD underestimation caused by double-counting.

### 9 · PSI Drift Monitoring Demo

```
[PSI Drift Monitoring Demo]
  PSI=0.3481 → ALERT – Immediate retraining required
```

| PSI Range | Status | Action |
|-----------|--------|--------|
| < 0.10 | Stable | No action required |
| 0.10 – 0.25 | Slight Shift | Investigate; flag for review |
| **> 0.25** | **Significant Drift** | **Immediate retraining trigger** |

Demo PSI of **0.3481** exceeds the 0.25 threshold. In production, this triggers the Airflow retraining DAG automatically.

---

## MLOps Architecture (from pipeline output)

```
=== MLOps / Continuous Training Architecture ===

1. DATA PIPELINE (Apache Airflow / Prefect)
   • Ingestion DAG: daily pulls from core banking system → feature store (Feast / Tecton)
   • Data quality checks: Great Expectations assertions on schema + statistical tests
   • OOT split: last 3 months held out for backtesting

2. MODEL REGISTRY (MLflow)
   • Every training run logs: params, metrics (AUC, PSI), artifacts (model pickle, SHAP plots)
   • Promotion workflow: Dev → Staging (shadow mode) → Production (best)
   • Versioning: SemVer (1.0.0 = major retraining, 1.0.x = hyperparameter tweak)

3. BEST–CHALLENGER (A/B Testing)
   • New model serves 10% traffic in shadow mode; best handles remaining 90%
   • Comparison window: 30 days minimum (enough defaults to observe)
   • Promotion criteria: ΔAUCprod ≥ 0.005 AND PSI_challenger < 0.10

4. RETRAINING TRIGGERS
   • PSI on score distribution > 0.25 (data drift)
   • Rolling 30-day AUC drops > 0.03 vs training benchmark (performance decay)
   • Scheduled quarterly retrain regardless of metrics (regulatory requirement)
   • Concept drift: feature-target correlation shift > 2σ baseline

5. MONITORING DASHBOARD (Grafana / Evidently AI)
   • Real-time: approval rate, average score, predicted default rate
   • Daily: PSI per feature (CSI), score PSI, calibration plot
   • Alerts: PagerDuty / Slack webhook on threshold breach

6. GOVERNANCE & COMPLIANCE
   • SHAP global importance logged per model version
   • Adverse action reason codes derived from top-3 negative SHAP features per applicant
   • Fairness audit: demographic parity + equalized odds per protected attribute (quarterly)
   • Model documentation: SR 11-7 / ECOA compliant model card stored in MLflow
```

---

## Model Selection Criteria

| Metric | Purpose |
|--------|---------|
| **AUC-ROC** | Primary ranking metric; threshold-agnostic; handles class imbalance |
| **Avg Precision / F1** | Minority-class performance (default = positive class, ~5%) |
| **LogLoss** | Calibration quality; important for EL = PD × LGD × EAD calculation |
| **Economic Interpretability** | Required for regulatory compliance (SR 11-7, ECOA, Basel III) |

---

## Outputs

| File | Description |
|------|-------------|
| `outputs/iv_analysis.csv` | IV scores and strength for all 15 features |
| `outputs/cv_results.csv` | AUC, Std, Avg Precision, F1, LogLoss for 3 models |
| `outputs/feature_importance.csv` | Permutation importance (AUC drop) for all 15 features |
| `outputs/income_paradox.json` | Income architecture analysis (Model 1/2/3) |

---
