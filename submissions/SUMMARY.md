## NHANES Age Prediction — Approach Summary

**Overview.** A clinically-grounded pipeline predicting Senior (65+) vs Adult from 7 NHANES health features (1,952 train / 312 test, 16% Senior, missing values throughout). The work prioritizes rigorous EDA and domain feature engineering, paired with a robust, cross-validated ensemble and careful F1 threshold tuning.

**EDA highlights.**
- Confirmed strong class imbalance (16% Senior) and light, near-uniform missingness across all features.
- Identified the **metabolic signature of ageing**: 2-hour oral glucose tolerance (`LBXGLT`) is the single strongest signal (Seniors ~141 vs Adults ~110 mg/dL), with seniors disproportionately crossing clinical *impaired* (≥140) and *diabetic* (≥200) thresholds.
- Key insight: insulin resistance (HOMA-IR) is **not** elevated in seniors — post-load glucose rises while insulin does not, consistent with age-related decline in insulin response. This directed feature design toward glucose-handling ratios.

**Feature engineering (7 → 26 features).** HOMA-IR; fasting-glucose & OGTT clinical categories (ADA); WHO BMI bands; glucose-handling ratios/gaps (`LBXGLT−LBXGLU`, `LBXGLT/LBXGLU`); an undiagnosed-diabetes flag; lifestyle × metabolism interactions; and informative-missingness indicators. The engineered glucose features ranked among the top predictors.

**Modeling.** Class-weighted Logistic Regression and HistGradientBoosting baselines → Optuna-tuned LightGBM / XGBoost / CatBoost → weighted / rank / stacking ensembles, all on stratified 10-fold out-of-fold predictions with decision-threshold tuning (the dominant F1 lever under imbalance). Final model: **rank-average ensemble**, selected for best, most CV-stable OOF F1. SHAP confirmed the model relies on the engineered glucose-handling features.

**Result.** 10-fold OOF **F1 ≈ 0.45** (ROC-AUC ≈ 0.76), well above the naive all-Senior baseline (0.28).

**Data-integrity note.** Diagnostic checks showed the ~0.98 leaderboard scores are attainable only by joining the `SEQN` identifier to externally published NHANES demographics (true age). We documented this but **deliberately restricted our solution to the provided features**, competing on legitimate predictive modeling.
