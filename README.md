# NHANES Age Prediction — Summer Analytics 2026 (IIT Guwahati)

Predict whether an NHANES respondent is a **Senior (65+ → `1`)** or an **Adult (<65 → `0`)** from 7 health / metabolic features. **Metric: F1** on the positive (Senior) class. Ties are broken on **EDA + Feature-Engineering quality**.

## Approach — legitimate ML, leak documented but NOT exploited

The single deliverable is [`notebooks/nhanes_age_prediction.ipynb`](notebooks/nhanes_age_prediction.ipynb): a fully-executed, narrated pipeline.

1. **EDA** of the metabolic signature of ageing — class imbalance (16% Senior), missingness, per-class distributions, correlations, and the glucose-tolerance / insulin-resistance / diabetes story.
2. **Domain feature engineering** grounded in clinical thresholds: `HOMA-IR`, fasting-glucose & OGTT categories (ADA), WHO BMI bands, glucose-handling ratios/gaps, an undiagnosed-diabetes flag, lifestyle interactions, and informative-missingness flags.
3. **Robust modeling**: class-weighted Logistic Regression & HistGradientBoosting baselines → **Optuna-tuned LightGBM / XGBoost / CatBoost** → weighted / rank / **stacking** ensembles, all on **stratified 10-fold OOF** with **decision-threshold tuning** (the biggest F1 lever under imbalance).
4. **SHAP** interpretability closing the EDA → FE → model loop.
5. **Data-leakage forensics (Section 9)** — see below.

## Honest result

| | Senior-F1 (10-fold OOF) | ROC-AUC |
|---|---|---|
| Naive (all-Senior) | 0.277 | — |
| Logistic Regression | 0.412 | 0.739 |
| HistGradientBoosting | 0.405 | 0.720 |
| LightGBM (Optuna) | 0.442 | 0.756 |
| CatBoost (Optuna) | 0.440 | 0.759 |
| **Rank-average ensemble (selected)** | **0.453** | **0.760** |

This sits at the top of the **legitimate "ranks 2–13" leaderboard cluster (~0.44–0.50)**.

## The 0.98 leaderboard scores are a data leak (we do not use it)

The public leaderboard has eight teams tied at exactly **`0.9796` (= 48/49)**. Our CV proves the 7 features give ROC-AUC ≈ 0.72–0.76, so **F1 = 0.98 is impossible honestly** (it needs AUC ≈ 0.999 on a 16%-minority class). The cause: `SEQN` is the **real NHANES 2013–2014 respondent ID**; joining it to the public CDC file `DEMO_H.XPT` (true age `RIDAGEYR`) recovers the labels directly. Section 9 reproduces the forensic evidence (SEQN-only AUC ≈ 0.51, disjoint train/test IDs, exact ID range). **We document this but deliberately decline to use external data** — this is a modeling exercise, not an answer-key lookup.

## How to run

```bash
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute --inplace notebooks/nhanes_age_prediction.ipynb
# or open the notebook in Jupyter / Cursor and Run All
```

> Note: use a Python 3.13 environment that has the libraries in `requirements.txt`. Running takes ~8 minutes (Optuna tuning of three boosters).

## Submitting on the platform

1. **Upload** `notebooks/nhanes_age_prediction.ipynb` for the EDA / Feature-Engineering tie-breaker.
2. **Submit** [`submissions/submission_final.csv`](submissions/submission_final.csv) (the selected, most CV-stable honest model) and **mark it as your final submission**.
   - The public LB is only ~156 rows (~25 Seniors) and is noisy; we trust **CV stability** over public-LB chasing to protect the private-half score.
   - Alternative candidates (`submission_lgbm.csv`, `submission_catboost.csv`, `submission_wavg.csv`, `submission_stack.csv`) are provided for comparison.

## Repository layout

```
Data/                         # provided train / test / sample-submission CSVs
notebooks/
  nhanes_age_prediction.ipynb # the deliverable (executed, with outputs)
submissions/
  submission_final.csv        # <- mark this one as final
  submission_{lgbm,catboost,wavg,stack}.csv
requirements.txt
README.md
```
