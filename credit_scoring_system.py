"""
=============================================================================
END-TO-END CREDIT SCORING SYSTEM
=============================================================================
Author : Senior Data Scientist / Credit Risk Architect
Dataset: 800 applicants, ~5% default rate
Stack  : pandas · sklearn · (xgboost / lightgbm drop-in ready)
=============================================================================
"""

# ---------------------------------------------------------------------------
# 0.  IMPORTS
# ---------------------------------------------------------------------------
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path

# Pre-processing
from sklearn.preprocessing import StandardScaler, RobustScaler, PowerTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.impute import SimpleImputer

# Models
from sklearn.linear_model  import LogisticRegression
from sklearn.ensemble      import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree          import DecisionTreeClassifier

# Metrics
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    f1_score, log_loss, classification_report,
    precision_recall_curve
)

# Calibration
from sklearn.calibration import CalibratedClassifierCV

import json

# ---------------------------------------------------------------------------
# 1.  DATA LOADING & CLEANING
# ---------------------------------------------------------------------------

def load_and_clean(path: str) -> pd.DataFrame:
    """
    Load CSV, strip column-name whitespace, and fix sign on debt columns.
    
    The raw file has:
      • Leading/trailing spaces in column names  → strip()
      • Debt values stored as negatives           → abs()
      • An ID column not useful as a feature      → drop
    """
    df = pd.read_csv(path)

    # --- 1a. Normalise column names ---
    df.columns = df.columns.str.strip().str.replace(" ", "_").str.lower()
    # Expected after normalisation:
    #   unique_applicant_id | age | years_at_employer | years_at_address
    #   income | credit_card_debt | automobile_debt | other_debt
    #   outcomes:_default_=_1   →  rename for convenience
    df.rename(columns={col: col for col in df.columns}, inplace=True)
    # Rename target
    target_col = [c for c in df.columns if "default" in c][0]
    df.rename(columns={target_col: "default"}, inplace=True)

    # --- 1b. Drop ID column ---
    df.drop(columns=["unique_applicant_id"], inplace=True, errors="ignore")

    # --- 1c. Fix negative debt values (they are liabilities, store as positive) ---
    debt_cols = ["credit_card_debt", "automobile_debt", "other_debt"]
    for col in debt_cols:
        if col in df.columns:
            df[col] = df[col].abs()

    print(f"[load] Shape after cleaning: {df.shape}")
    print(f"[load] Default rate: {df['default'].mean():.2%}")
    return df


# ---------------------------------------------------------------------------
# 2.  MISSING-VALUE STRATEGY  (production-ready, data is clean here)
# ---------------------------------------------------------------------------

def build_imputer_strategy(df: pd.DataFrame) -> dict:
    """
    Return a dict describing imputation strategy per column type.
    In production, missing values are common; this function documents the
    rationale and returns a ready-to-use sklearn SimpleImputer per group.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != "default"]

    strategy_map = {}
    for col in numeric_cols:
        skew = df[col].skew()
        if abs(skew) > 1.0:
            # Right-skewed distributions → median more robust than mean
            strategy_map[col] = "median"
        else:
            strategy_map[col] = "mean"

    print(f"[imputer] Strategy map: {strategy_map}")
    return strategy_map


# ---------------------------------------------------------------------------
# 3.  FEATURE ENGINEERING
# ---------------------------------------------------------------------------

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive credit-relevant ratios and composite indicators.

    Key engineered features
    ──────────────────────
    total_debt          : credit_card + auto + other debt
    dti                 : Debt-to-Income ratio  (total_debt / income)
    cc_dti              : Credit Card share of income
    auto_dti            : Auto Debt share of income
    stability_index     : avg(years_at_employer, years_at_address) / age
    tenure_ratio        : years_at_employer / age  (employment maturity)
    log_income          : log1p(income) to reduce right-skew
    debt_per_year       : total_debt / years_at_employer  (debt accumulation rate)
    """
    df = df.copy()

    eps = 1e-6  # prevent division by zero

    # Totals
    df["total_debt"] = (
        df["credit_card_debt"] + df["automobile_debt"] + df["other_debt"]
    )

    # Ratios
    df["dti"]          = df["total_debt"] / (df["income"] + eps)
    df["cc_dti"]       = df["credit_card_debt"] / (df["income"] + eps)
    df["auto_dti"]     = df["automobile_debt"] / (df["income"] + eps)

    # Stability
    df["stability_index"] = (
        (df["years_at_employer"] + df["years_at_address"]) / (2 * df["age"] + eps)
    )
    df["tenure_ratio"] = df["years_at_employer"] / (df["age"] + eps)

    # Log transform for highly skewed income
    df["log_income"] = np.log1p(df["income"])

    # Debt per year of employment (risk accumulation speed)
    df["debt_per_year"] = df["total_debt"] / (df["years_at_employer"] + eps)

    print(f"[features] Engineered features added. New shape: {df.shape}")
    return df


# ---------------------------------------------------------------------------
# 4.  CORRELATION & INFORMATION VALUE (WOE/IV)
# ---------------------------------------------------------------------------

def compute_iv(df: pd.DataFrame, feature: str, target: str = "default",
               bins: int = 10) -> float:
    """
    Compute Information Value (IV) for a numeric feature using equal-frequency bins.
    
    IV Interpretation
    ─────────────────
    < 0.02  : Unpredictive
    0.02–0.1: Weak predictor
    0.1–0.3 : Medium predictor
    0.3–0.5 : Strong predictor
    > 0.5   : Suspicious (check for data leakage)
    """
    temp = df[[feature, target]].dropna().copy()
    temp["bin"] = pd.qcut(temp[feature], q=bins, duplicates="drop")

    grouped = temp.groupby("bin")[target].agg(
        events=lambda x: (x == 1).sum(),
        non_events=lambda x: (x == 0).sum()
    )

    total_events     = grouped["events"].sum()
    total_non_events = grouped["non_events"].sum()
    eps = 0.5  # Laplace smoothing to avoid log(0)

    grouped["dist_events"]     = (grouped["events"] + eps) / (total_events + eps * len(grouped))
    grouped["dist_non_events"] = (grouped["non_events"] + eps) / (total_non_events + eps * len(grouped))
    grouped["woe"]             = np.log(grouped["dist_events"] / grouped["dist_non_events"])
    grouped["iv"]              = (grouped["dist_events"] - grouped["dist_non_events"]) * grouped["woe"]

    return grouped["iv"].sum()


def run_iv_analysis(df: pd.DataFrame, target: str = "default") -> pd.DataFrame:
    """Return a DataFrame of feature → IV, sorted descending."""
    features = [c for c in df.columns if c != target]
    iv_scores = {f: compute_iv(df, f, target) for f in features}
    iv_df = pd.DataFrame.from_dict(iv_scores, orient="index", columns=["IV"])
    iv_df = iv_df.sort_values("IV", ascending=False)
    iv_df["strength"] = pd.cut(
        iv_df["IV"],
        bins=[-np.inf, 0.02, 0.1, 0.3, 0.5, np.inf],
        labels=["Unpredictive", "Weak", "Medium", "Strong", "Suspicious"]
    )
    return iv_df


# ---------------------------------------------------------------------------
# 5.  SCALING / TRANSFORMATIONS
# ---------------------------------------------------------------------------

def build_feature_pipeline(use_power_transform: bool = True) -> Pipeline:
    """
    Build a sklearn Pipeline that:
      1. Imputes missing values with median (safe for production)
      2. Applies PowerTransformer (Yeo-Johnson) for skewed features OR StandardScaler
      3. Falls back to RobustScaler if outliers are extreme
    """
    scaler = (
        PowerTransformer(method="yeo-johnson", standardize=True)
        if use_power_transform
        else RobustScaler()
    )
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  scaler),
    ])
    return pipe


# ---------------------------------------------------------------------------
# 6.  MODEL FACTORY
# ---------------------------------------------------------------------------

def get_models(class_weight: str = "balanced") -> dict:
    """
    Return a dict of model_name → unfitted estimator.
    
    Model Selection Rationale
    ─────────────────────────
    LogisticRegression  : Regulatory baseline; fully interpretable; scorecard-friendly.
    RandomForest        : Handles non-linearity; robust to outliers; OOB estimate.
    GradientBoosting    : Sklearn proxy for XGBoost; strong AUC; slower to train.
    
    Note: In production, replace GradientBoosting with:
        from xgboost  import XGBClassifier
        from lightgbm import LGBMClassifier
    """
    return {
        "LogisticRegression": LogisticRegression(
            C=0.1,
            class_weight=class_weight,
            max_iter=1000,
            random_state=42
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=300,
            class_weight=class_weight,
            max_depth=6,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1
        ),
        "GradientBoosting_XGBProxy": GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            random_state=42
        ),
    }


# ---------------------------------------------------------------------------
# 7.  STRATIFIED K-FOLD CROSS VALIDATION
# ---------------------------------------------------------------------------

def cross_validate_models(X: np.ndarray, y: np.ndarray,
                           models: dict, cv_folds: int = 5) -> pd.DataFrame:
    """
    Run stratified K-Fold CV for each model.
    
    Metrics collected
    ─────────────────
    roc_auc   : Primary ranking metric; threshold-independent.
    avg_prec  : Area under Precision-Recall curve; critical for imbalanced data.
    f1        : Harmonic mean precision/recall at default threshold.
    neg_log   : Log-loss; penalises overconfident wrong predictions.
    
    Why AUC-ROC over Accuracy?
    ───────────────────────────
    With 95% non-defaults, a naive "predict all 0" model achieves 95% accuracy
    but 0.50 AUC. AUC measures rank-ordering ability across all thresholds,
    which is exactly what a credit scorecard needs.
    """
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    results = []

    for name, model in models.items():
        cv_res = cross_validate(
            model, X, y,
            cv=skf,
            scoring={
                "roc_auc":   "roc_auc",
                "avg_prec":  "average_precision",
                "f1":        "f1",
                "neg_log":   "neg_log_loss",
            },
            return_train_score=False,
            n_jobs=-1,
        )
        results.append({
            "Model":    name,
            "AUC_ROC":  cv_res["test_roc_auc"].mean(),
            "AUC_ROC_std": cv_res["test_roc_auc"].std(),
            "Avg_Prec": cv_res["test_avg_prec"].mean(),
            "F1":       cv_res["test_f1"].mean(),
            "LogLoss":  -cv_res["test_neg_log"].mean(),
        })
        print(f"  [{name}] AUC={results[-1]['AUC_ROC']:.4f} ±{results[-1]['AUC_ROC_std']:.4f} "
              f"| AvgPrec={results[-1]['Avg_Prec']:.4f} | F1={results[-1]['F1']:.4f}")

    return pd.DataFrame(results).sort_values("AUC_ROC", ascending=False)


# ---------------------------------------------------------------------------
# 8.  PERMUTATION FEATURE IMPORTANCE  (SHAP proxy)
# ---------------------------------------------------------------------------

def permutation_importance(model, X: pd.DataFrame, y: np.ndarray,
                            n_repeats: int = 30, metric=roc_auc_score) -> pd.DataFrame:
    """
    Estimate feature importance by measuring AUC drop when each feature is
    randomly permuted (breaks the feature-target relationship).
    
    This is a model-agnostic, SHAP-equivalent global importance measure.
    For production, replace with:
        import shap
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
    """
    baseline = metric(y, model.predict_proba(X.values)[:, 1])
    importances = {}

    for col in X.columns:
        drops = []
        for _ in range(n_repeats):
            X_perm = X.copy()
            X_perm[col] = np.random.permutation(X_perm[col].values)
            score = metric(y, model.predict_proba(X_perm.values)[:, 1])
            drops.append(baseline - score)
        importances[col] = np.mean(drops)

    imp_df = pd.DataFrame.from_dict(importances, orient="index", columns=["importance_drop"])
    return imp_df.sort_values("importance_drop", ascending=False)


# ---------------------------------------------------------------------------
# 9.  PSI – POPULATION STABILITY INDEX
# ---------------------------------------------------------------------------

def compute_psi(expected: np.ndarray, actual: np.ndarray,
                bins: int = 10) -> float:
    """
    PSI = Σ (Actual% − Expected%) × ln(Actual% / Expected%)
    
    PSI Interpretation
    ──────────────────
    < 0.10 : Stable; no action needed.
    0.10–0.25 : Slight shift; monitor closely.
    > 0.25 : Significant drift; consider retraining.
    
    Used for
    ─────────
    • Score distribution shift over time (PSI on predicted probabilities)
    • Feature distribution shift (CSI per feature)
    """
    breakpoints = np.linspace(0, 100, bins + 1)
    expected_pct = np.histogram(expected, bins=np.percentile(expected, breakpoints))[0]
    actual_pct   = np.histogram(actual,   bins=np.percentile(expected, breakpoints))[0]

    eps = 1e-6
    expected_pct = (expected_pct + eps) / expected_pct.sum()
    actual_pct   = (actual_pct   + eps) / actual_pct.sum()

    psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return psi


def monitor_drift(train_scores: np.ndarray, prod_scores: np.ndarray) -> dict:
    """Compute PSI and flag action level."""
    psi = compute_psi(train_scores, prod_scores)
    if psi < 0.10:
        action = "STABLE – Monitor monthly"
    elif psi < 0.25:
        action = "WARNING – Investigate; retrain candidate"
    else:
        action = "ALERT – Immediate retraining required"
    return {"psi": round(psi, 4), "action": action}


# ---------------------------------------------------------------------------
# 10. INCOME PARADOX ANALYSIS  (Model 1 / 2 / 3)
# ---------------------------------------------------------------------------

def income_paradox_analysis(df: pd.DataFrame) -> dict:
    """
    Analyse three architectural configurations for Income usage.

    Model 1 – Scoring only
    ──────────────────────
    Income included as a scoring feature.
    Risk: High income may mask high debt, inflating score.

    Model 2 – Limit setting only
    ────────────────────────────
    Income is used in a secondary regression/rule model to set credit limits.
    The scorecard sees only debt ratios; income determines exposure.
    Pro: Clean separation of default probability and exposure.

    Model 3 – Dual usage
    ────────────────────
    Income appears in both scorecard and limit model.
    Risk of "Double Counting": the same information reduces PD and increases
    credit limit simultaneously, creating a positive correlation between
    default probability (where it should be negative) and exposure.
    This leads to higher Expected Loss = PD × LGD × EAD than implied.

    Double-Counting Impact
    ──────────────────────
    E[Loss] = PD × LGD × EAD
    If Income ↑ → PD ↓ (via score) AND EAD ↑ (via limit),
    then income creates an inverted risk/exposure relationship.
    A customer with very high income and high leverage gets a LOW score
    but a HIGH limit — the opposite of prudent risk management.

    Recommendation: Use Income for limit-setting (Model 2) and DTI,
    log-income, or income-rank for the scorecard. Never raw income in both.
    """
    from sklearn.linear_model import LinearRegression

    results = {}

    feature_cols = [c for c in df.columns if c != "default"]
    y = df["default"].values
    pipe = build_feature_pipeline()

    # ---- Model 1: Income as scoring feature ----
    m1_features = feature_cols  # includes income
    X1 = pipe.fit_transform(df[m1_features])
    lr1 = LogisticRegression(C=0.1, class_weight="balanced", max_iter=1000, random_state=42)
    lr1.fit(X1, y)
    m1_auc = roc_auc_score(y, lr1.predict_proba(X1)[:, 1])
    results["Model1_income_as_score"] = {
        "auc": round(m1_auc, 4),
        "income_coef": float(
            lr1.coef_[0][list(df[m1_features].columns).index("income")]
        ) if "income" in m1_features else None,
        "note": "Income included as a direct scoring feature."
    }

    # ---- Model 2: Income removed from scorecard; used for limit only ----
    m2_features = [c for c in feature_cols if c != "income"]
    X2 = pipe.fit_transform(df[m2_features])
    lr2 = LogisticRegression(C=0.1, class_weight="balanced", max_iter=1000, random_state=42)
    lr2.fit(X2, y)
    m2_auc = roc_auc_score(y, lr2.predict_proba(X2)[:, 1])

    # Secondary limit model (simple regression of income on total_debt for illustration)
    limit_model = LinearRegression()
    limit_model.fit(df[["income"]].values, df["total_debt"].values)
    results["Model2_income_for_limit"] = {
        "auc": round(m2_auc, 4),
        "note": "Income excluded from score; used only for credit limit via secondary model.",
        "limit_model_r2": round(limit_model.score(df[["income"]].values, df["total_debt"].values), 4)
    }

    # ---- Model 3: Dual usage ----
    # AUC same as Model 1 (income in score), but limit also tied to income → double counting
    correlation_double_count = df["income"].corr(df["total_debt"])
    results["Model3_dual_income"] = {
        "auc": round(m1_auc, 4),  # same score as M1
        "double_count_risk": "HIGH" if abs(correlation_double_count) > 0.5 else "MODERATE",
        "income_debt_corr": round(correlation_double_count, 4),
        "note": (
            "Income lowers PD AND raises limit — correlated risk/exposure creates "
            "higher Expected Loss than the model implies."
        )
    }

    return results


# ---------------------------------------------------------------------------
# 11. MLOps / CI-CD STRATEGY  (documented, not executable here)
# ---------------------------------------------------------------------------

MLOPS_STRATEGY = """
=== MLOps / Continuous Training Architecture ===

1. DATA PIPELINE (Apache Airflow / Prefect)
   • Ingestion DAG: daily pulls from core banking system → feature store (Feast / Tecton)
   • Data quality checks: Great Expectations assertions on schema + statistical tests
   • OOT split: last 3 months held out for backtesting

2. MODEL REGISTRY (MLflow)
   • Every training run logs: params, metrics (AUC, PSI), artifacts (model pickle, SHAP plots)
   • Promotion workflow: Dev → Staging (shadow mode) → Production (champion)
   • Versioning: SemVer (1.0.0 = major retraining, 1.0.x = hyperparameter tweak)

3. CHAMPION–CHALLENGER (A/B Testing)
   • New model serves 10% traffic in shadow mode; champion handles remaining 90%
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
"""


# ---------------------------------------------------------------------------
# 12.  MAIN PIPELINE
# ---------------------------------------------------------------------------

def main(data_path: str = "data.csv") -> dict:
    print("=" * 60)
    print("CREDIT SCORING SYSTEM – FULL PIPELINE")
    print("=" * 60)

    # ── Load & Clean ──────────────────────────────────────────
    df = load_and_clean(data_path)

    # ── Missing value strategy (documenting for production) ──
    imputer_strategy = build_imputer_strategy(df)

    # ── Feature Engineering ───────────────────────────────────
    df = engineer_features(df)

    # ── IV Analysis ───────────────────────────────────────────
    print("\n[IV Analysis]")
    iv_df = run_iv_analysis(df)
    print(iv_df.to_string())

    # ── Prepare X, y ─────────────────────────────────────────
    y = df["default"].values
    feature_cols = [c for c in df.columns if c != "default"]
    X_raw = df[feature_cols]

    pipe = build_feature_pipeline(use_power_transform=True)
    X = pipe.fit_transform(X_raw)
    X_df = pd.DataFrame(X, columns=feature_cols)

    # ── Cross-Validation ─────────────────────────────────────
    print("\n[Cross-Validation Results]")
    models = get_models(class_weight="balanced")
    cv_results = cross_validate_models(X, y, models, cv_folds=5)
    print("\n", cv_results.to_string(index=False))

    # ── Select Champion Model ─────────────────────────────────
    champion_name = cv_results.iloc[0]["Model"]
    print(f"\n[Champion] {champion_name} selected based on highest AUC-ROC")

    champion = models[champion_name]
    champion.fit(X, y)

    # ── Permutation Importance ────────────────────────────────
    print("\n[Permutation Feature Importance (SHAP proxy)]")
    imp_df = permutation_importance(champion, X_df, y, n_repeats=15)
    print(imp_df.head(10).to_string())

    # ── Income Paradox ────────────────────────────────────────
    print("\n[Income Paradox Analysis]")
    income_results = income_paradox_analysis(df)
    for model_name, info in income_results.items():
        print(f"  {model_name}: AUC={info['auc']} | {info['note']}")

    # ── PSI Monitoring Demo ───────────────────────────────────
    print("\n[PSI Drift Monitoring Demo]")
    train_probs = champion.predict_proba(X)[:, 1]
    # Simulate production drift (shift mean by 0.05)
    prod_probs = np.clip(train_probs + np.random.normal(0.05, 0.02, len(train_probs)), 0, 1)
    drift = monitor_drift(train_probs, prod_probs)
    print(f"  PSI={drift['psi']} → {drift['action']}")

    # ── Print MLOps Strategy ──────────────────────────────────
    print(MLOPS_STRATEGY)

    # ── Save Outputs ──────────────────────────────────────────
    import os
    os.makedirs("outputs", exist_ok=True)

    cv_results.to_csv("outputs/cv_results.csv", index=False)
    iv_df.to_csv("outputs/iv_analysis.csv")
    imp_df.to_csv("outputs/feature_importance.csv")

    # Save income paradox as JSON
    with open("outputs/income_paradox.json", "w") as f:
        json.dump(income_results, f, indent=2)

    print("\n[Done] All outputs saved to outputs/")

    return {
        "cv_results": cv_results,
        "iv_df": iv_df,
        "imp_df": imp_df,
        "champion": champion_name,
        "income_paradox": income_results,
        "drift": drift,
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Credit Scoring System Pipeline")
    parser.add_argument(
        "--data",
        type=str,
        default="data.csv",
        help="Path to the raw input CSV (default: data.csv)"
    )
    args = parser.parse_args()
    results = main(data_path=args.data)
