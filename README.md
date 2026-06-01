**FraudGuard AI — Transaction Fraud Detection
**

> Binary classification system for detecting fraudulent financial transactions on a severely imbalanced dataset (~0.17% fraud rate). Built as an end-to-end ML experimentation project covering baseline modeling, threshold optimization, imbalance handling, feature selection, and business-oriented evaluation.

**Project Overview
**

Financial fraud detection presents a core ML challenge: class imbalance so extreme that a model predicting "not fraud" on every transaction achieves >99% accuracy while catching zero fraud. This project addresses that challenge through iterative experimentation with model selection, probability thresholds, cost-sensitive weighting, and evaluation metrics appropriate for rare-event classification.

**Dataset:** [Credit Card Fraud Detection — Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- 284,807 transactions | 492 fraud cases | ~0.17% fraud rate
- 28 PCA-anonymized features (V1–V28) + `Time`, `Amount`, `Class`

**Results Summary
**

| Metric | Logistic Regression (baseline) | XGBoost (tuned) |
|---|---|---|
| ROC-AUC | N/A | ~0.97 |
| PR-AUC | N/A | ~0.81 |
| Fraud Recall | ~65% | ~88% |
| Threshold | 0.50 | 0.20 |

> **Key finding:** PR-AUC is the primary evaluation metric here — ROC-AUC inflates on imbalanced data. Improving fraud recall from 65% to 88% while maintaining a PR-AUC of 0.81 is the core result.

**Methodology**

1. Exploratory Data Analysis

Investigated class imbalance, feature distributions, and fraud-vs-legitimate transaction behavior.

- Identified severe imbalance: 99.83% legitimate / 0.17% fraud
- Compared feature distributions between fraud and non-fraud classes
- Correlation and distribution analysis highlighted V14, V17, V12, V10, and V7 as features with the largest separability between classes
- Confirmed that standard accuracy is not a valid optimization target for this problem

2. Baseline Model — Logistic Regression

Established a baseline using Logistic Regression with default parameters.

- Demonstrated the "accuracy paradox": high overall accuracy while underperforming on fraud recall (~65%)
- Used `predict_proba()` to experiment with custom decision thresholds (0.50 → 0.05)
- Showed that lowering the threshold from 0.5 to 0.2 improved fraud recall at the cost of additional false positives — a core precision-recall tradeoff present in real fraud policy decisions

3. XGBoost Model

Replaced the linear baseline with a gradient-boosted tree model.

- XGBoost outperformed Logistic Regression on PR-AUC (0.81 vs ~0.71)
- Better capture of nonlinear patterns in the PCA-transformed feature space
- Fraud recall improved to ~88% with threshold tuning

4. Feature Importance & Reduced-Feature Model

Extracted and ranked feature importance scores from the trained XGBoost model.

- V14 emerged as the dominant fraud-predictive feature by a significant margin
- Top 5 features: **V14, V7, V12, V17, V10**
- Built a reduced-feature model using only these 5 features
- Result: comparable PR-AUC to the full-feature model, suggesting fraud signal is concentrated in a small feature subset

5. Imbalance Handling — `scale_pos_weight`

Addressed the class imbalance directly via XGBoost's `scale_pos_weight` parameter.

- Tested multiple weight ratios reflecting the actual fraud/legitimate class ratio
- Higher weighting increased fraud recall but reduced precision — consistent with cost-sensitive classification theory
- Analyzed the business tradeoff: catching more fraud increases false alert volume, impacting operational costs

6. Evaluation Framework

Used metrics appropriate for imbalanced binary classification:

| Metric | Why it matters here |
|---|---|
| **PR-AUC** | Primary metric — measures quality across all thresholds on imbalanced data |
| **ROC-AUC** | Secondary — optimistic on imbalanced data, included for completeness |
| **Fraud Recall** | Core business KPI — what % of actual fraud does the model catch? |
| **Precision** | Operational cost — what % of flagged transactions are actual fraud? |
| **Confusion Matrix** | Visualizes false negatives (missed fraud) vs false positives (false alerts) |

**Key Technical Insights
**
**Threshold tuning is a business decision, not just a modeling one.** In fraud detection, the cost of missing fraud (false negative) typically exceeds the cost of a false alert (false positive). Lowering the classification threshold from 0.5 to 0.2 improved fraud recall by ~20 percentage points with a manageable increase in false positives — a tradeoff that maps directly to how fraud teams think about risk tolerance.

**Feature concentration.** Five features (V14, V7, V12, V17, V10) carried the majority of the fraud signal. The reduced-feature model achieved comparable performance to the full 28-feature model, suggesting these features likely correspond to behavioral patterns strongly associated with fraudulent activity in the original (anonymized) feature space.

**PR-AUC over ROC-AUC.** On datasets with extreme class imbalance, ROC-AUC can be misleadingly high because it considers the true negative rate — which is trivially high when 99.8% of data is negative. PR-AUC focuses on precision and recall among predicted positives, making it the more informative metric for this problem.

**Tech Stack
**
| Layer | Tools |
|---|---|
| Language | Python 3.x |
| Data manipulation | Pandas, NumPy |
| Modeling | Scikit-learn, XGBoost |
| Visualization | Matplotlib, Seaborn |
| Environment | Jupyter Notebook |

**How to Run
**
```bash
git clone https://github.com/olzhasmalik61/fraudguard-ai
cd fraudguard-ai
pip install -r requirements.txt
```

Download the dataset from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place `creditcard.csv` in the project root. Then run notebook.

**Dataset Credit
**
Kaggle — [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
ULB Machine Learning Group. Real anonymized transaction data from European cardholders (September 2013).
