# COMPAS Recidivism Prediction & Algorithmic Fairness Audit

## Project Overview

This capstone project investigates **predictive performance, algorithmic fairness, and explainability** in criminal justice risk assessment. Using ProPublica's publicly available COMPAS dataset, I developed an independent **XGBoost binary classification model** to predict whether a person would recidivate within two years. I then evaluated the model's errors across demographic groups, investigated its most influential features, and assessed potential risks associated with using algorithmic predictions in high-stakes decisions.

The central research question is:

> **Does a recidivism prediction model treat racial groups equitably across multiple fairness metrics?**

**Important distinction:** This project trains and audits an independently developed model using the COMPAS dataset. It does **not** reproduce or directly audit the proprietary COMPAS scoring algorithm.

## Dataset

- **Source:** [ProPublica — COMPAS Analysis](https://github.com/propublica/compas-analysis)
- **File:** [`compas-scores-two-years.csv`](https://github.com/propublica/compas-analysis/blob/master/compas-scores-two-years.csv)
- **Original dataset:** 7,214 records and 53 columns
- **Target variable:** `two_year_recid` (binary two-year recidivism outcome)
- **Model features:** age, age category, sex, prior offense count, juvenile offense counts, and current charge degree

**Race was excluded from the model's input features** and retained separately for fairness evaluation. COMPAS risk scores and post-assessment outcome information were not used as predictors, helping keep the independently trained model distinct from the original scoring system and reduce information leakage.

## Methodology

1. **Exploratory data analysis (EDA):** Examined class balance, demographic distributions, COMPAS risk categories, prior offense histories, and data quality using Pandas, Matplotlib, and Seaborn.
2. **Preprocessing and feature engineering:** Built Scikit-learn `Pipeline` and `ColumnTransformer` workflows with median imputation for numerical variables, most-frequent imputation for categorical variables, and one-hot encoding. Used an 80/20 stratified train-test split.
3. **Machine learning:** Trained an `XGBClassifier` and tuned hyperparameters with `GridSearchCV` and cross-validation, optimizing probability quality using the Brier score. Applied sigmoid probability calibration with `CalibratedClassifierCV`.
4. **Model evaluation:** Assessed accuracy, precision, recall, F1-score, ROC-AUC, Brier score, and confusion matrices on the held-out test set.
5. **Fairness assessment:** Compared demographic parity, equality of opportunity, false-positive and false-negative rates, and predictive parity between the two largest racial groups in the test set.
6. **Model explainability:** Used **SHAP** to identify the key drivers of predictions and performed counterfactual scenario tests to examine how input changes affect an individual prediction.
7. **Responsible AI and governance:** Connected findings to relevant ethical and legal frameworks and proposed safeguards, including monitoring, human review, and procedures for challenging incorrect information.

## Explainability & Counterfactual Analysis

SHAP identified **prior offense count** and **age** as the strongest global prediction drivers, followed by sex, age category, and juvenile offense history.

In an illustrative false-positive case, changing the historical prior-offense count from eight to zero reduced the model's predicted recidivism probability from **69.8% to 25.9%**, reversing the predicted class. This was a **hypothetical sensitivity test**, not a realistic intervention: people cannot simply erase their criminal histories. The result highlights a gap between explaining a prediction and providing meaningful, actionable recourse.

## Governance Recommendations

The project also examined the implications of the **U.S. Fourteenth Amendment**, *State v. Loomis*, **GDPR Article 22**, and the **proposed Algorithmic Accountability Act of 2023**. Based on the audit, recommended safeguards include:

- Evaluate multiple fairness metrics and monitor group-level error rates rather than relying on aggregate accuracy alone.
- Review potential proxy variables and document model limitations.
- Provide understandable explanations, meaningful human oversight, and ways to correct or contest inaccurate input data.
- Do not use a model with these limitations as the sole basis for consequential criminal justice decisions.

