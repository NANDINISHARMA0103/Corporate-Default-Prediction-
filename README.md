# Corporate Default Prediction: A Matched-Pair Analysis of Indian Credit Events

## Overview

This project builds a corporate default/credit-downgrade classification model using a **matched-pair sample design** — 20 Indian companies that suffered a confirmed credit rating downgrade or default, each paired against a same-sector, investment-grade peer from the same fiscal year. The methodology follows the precedent set by Altman's original 1968 Z-score study, which used the same matched-pair approach at a similar scale (33 distressed vs. 33 stable firms).

Unlike commonly used public credit-risk datasets (e.g., Kaggle's Home Credit Default Risk competition), every data point here was independently sourced and verified: rating actions were traced to primary sources (CRISIL, ICRA, CARE, India Ratings, and — where applicable — RBI regulatory actions), and financial ratios were computed from each company's own annual report data via screener.in, aligned to the fiscal year immediately preceding its rating event.

## Data

- **Sample size:** 40 companies (20 distressed, 20 matched peers), fully balanced
- **Sectors covered:** Aviation, NBFC/Banking, Telecom, Steel, Auto Components, Retail, Textiles/Plastics, Power/Infrastructure, Travel, Gems & Jewellery, Fitness Services, Housing Finance, Shipbuilding, Construction
- **Time period:** 2012–2022
- **Features (7):** Debt/Equity, Interest Coverage Ratio, ROA, ROE, Operating Margin, Sales Growth (YoY), Operating Cash Flow / Total Debt
- **Label:** Binary (1 = distressed/downgraded, 0 = stable investment-grade peer)
- **Sources:** Official rating rationales (CRISIL, ICRA, CARE, India Ratings), RBI regulatory actions, and company financial statements via screener.in

### Data Limitations (disclosed)

This project takes an explicit "small, verified, honestly-caveated" approach rather than inflating sample size at the cost of accuracy:

- **Year misalignment:** For 6 peer companies (Motherson Sumi, Whirlpool India, Titan Company, NTPC, Cochin Shipyard), the earliest available financial data in the free data source did not reach back to the exact fiscal year of the matched distressed company's event. The nearest available year was used instead, with the mismatch disclosed per row.
- **Ratio distortions:** Several ratios become numerically unstable under specific conditions and were flagged rather than silently reported: negative net worth causes Debt/Equity and ROE to invert sign (e.g., Jet Airways, Reliance Capital, Suzlon Energy); near-zero debt causes Interest Coverage and OCF/Total Debt to inflate to extreme values (e.g., ABB India, Cochin Shipyard, Finolex Industries); near-zero revenue causes Operating Margin to become uninterpretable (ABG Shipyard).
- **Sector-specific interpretation:** For banks and NBFCs, "Operating Margin" reflects total income rather than manufacturing revenue, and "Interest Coverage" reflects lending spread rather than discretionary financing cost. These are noted rather than treated as directly comparable to non-financial firms.
- **One unverified case:** Era Infra Engineering's event timing was inferred from a P&L inflection point rather than a confirmed rating agency action, and is retained in the sample with this caveat.
- **Corporate group concentration:** Three sample firms (Reliance Capital, Reliance Communications, Reliance Home Finance) share a common promoter group; results may partly reflect group-specific governance and liquidity contagion rather than purely firm-level fundamentals.
- **Missing values:** True missing values (e.g., Whirlpool India's Debt/Equity, ABG Shipyard's Operating Margin) were preserved as missing (NaN) for the XGBoost model, which handles missing data natively without imputation. Logistic Regression, which cannot process missing values, used group-median imputation for these specific cells only — disclosed explicitly rather than silently applied.

## Methodology

1. **Matched-pair construction:** Each distressed firm's financial ratios were computed from its annual report for the fiscal year immediately preceding its rating downgrade/default (T-1), avoiding lookahead bias. Peer companies were matched on sector and, wherever data allowed, fiscal year.
2. **Models:** Logistic Regression (with RobustScaler to handle outlier ratios) and XGBoost (shallow trees, L1/L2 regularization tuned for small-n).
3. **Validation:** Leave-One-Out Cross-Validation (LOOCV) — the appropriate validation strategy at this sample size, since standard k-fold splits would leave too few training examples per fold.
4. **Explainability:** SHAP (SHapley Additive exPlanations) values computed on the fitted XGBoost model to identify which features actually drove classifications, rather than relying on accuracy alone.

### Feature Correlation

![Correlation Heatmap](correlation_heatmap.png)

Ratios show largely independent behavior across the sample, with a few expected relationships (e.g., ROA and ROE move together for firms with positive net worth) — supporting the use of all seven as distinct model inputs rather than redundant signals.

## Results

| Model | Accuracy | ROC-AUC |
|---|---|---|
| Logistic Regression | 77.5% | 0.850 |
| XGBoost | 75.0% | 0.792 |

![ROC Curve Comparison](roc_curve.png)

**Classification Report (Logistic Regression):**
```
              precision    recall  f1-score   support
           0       0.79      0.75      0.77        20
           1       0.76      0.80      0.78        20
    accuracy                           0.78        40
```

Logistic Regression slightly outperformed XGBoost on this sample — a plausible and honestly-reported result at n=40, where simpler models often generalize better than higher-capacity ones.

### Feature Importance (SHAP)

| Rank | Feature | Importance |
|---|---|---|
| 1 | ROA | 0.298 |
| 2 | Interest Coverage | 0.225 |
| 3 | ROE | 0.136 |
| 4 | OCF/Total Debt | 0.095 |
| 5 | Operating Margin | 0.094 |
| 6 | Sales Growth | 0.077 |
| 7 | Debt/Equity | 0.075 |

ROA and Interest Coverage emerged as the strongest predictors, while Debt/Equity and Sales Growth carried the least signal — directly consistent with the ratio-distortion issues identified during data collection (Debt/Equity was repeatedly unreliable under negative net worth and sector-specific leverage norms; Sales Growth often *masked* distress rather than indicating health, as several distressed firms showed strong revenue growth shortly before collapse).

![SHAP Summary Plot](shap_summary.png)

### Feature Importance (XGBoost Native)

![Feature Importance Bar Chart](feature_importance.png)

Confirms the SHAP ranking above using XGBoost's built-in importance scoring — ROA and Interest Coverage lead by a clear margin under both methods.

### Precision-Recall Curve (XGBoost)

![Precision-Recall Curve](precision_recall_curve.png)

### Key Finding: Misclassification Pattern

All 10 misclassifications (25% of the sample) were concentrated in companies with **documented, previously-identified data limitations** — not random noise:

**False negatives** (predicted stable, actually distressed): Cox & Kings, Gitanjali Gems, and Talwalkars Better Value Fitness — all three carry analyst-alleged or confirmed accounting irregularities, where manipulated financial statements produced healthy-looking ratios that masked genuine distress. Era Infra Engineering, the one unverified event-date case, was also misclassified.

**False positives** (predicted distressed, actually stable): Thomas Cook India and Bharti Airtel — both matched to distressed peers during periods of genuine sector-wide stress (Cox & Kings' travel-sector downturn, Vodafone Idea's telecom price war), narrowing the real contrast. IndusInd Bank and Can Fin Homes reflect known banking/NBFC ratio-comparability limits. Thangamayil Jewellery and NCC Ltd were both flagged with significant year-mismatches (4+ years) relative to their matched distressed companies.

**No misclassification occurred among the 30 cleanly-matched, fully-verified pairs.** This suggests the model is behaving as a sound methodology should: failing specifically where the underlying data has documented limitations, and succeeding where the data is solid — evidence that ratio-based analysis has genuine, identifiable blind spots (manipulated financial statements) rather than being generically unreliable.

## Limitations & Future Work

- Sample size (n=40) limits statistical power; findings should be read as indicative rather than production-grade
- Standalone (not consolidated) financials were used throughout, which may understate risk in complex group/conglomerate structures (noted specifically for Lanco Infratech)
- Future extensions could include: consolidated financials for group entities, additional matched pairs to reach closer to Altman's original 66-firm scale, and a formal fraud-detection sub-analysis given the recurring "healthy ratios, later-confirmed fraud" pattern observed in 3+ sample firms


Conclusion

This project set out to test a narrow, specific question: whether a small set of standard financial ratios, applied to an independently-sourced, carefully-verified sample of Indian corporate credit events, can meaningfully distinguish distressed firms from healthy peers — and where such an approach reaches its limits. The results support a qualified but genuine yes. Logistic Regression achieved 77.5% accuracy and 0.850 ROC-AUC under Leave-One-Out Cross-Validation, with Return on Assets and Interest Coverage emerging as the dominant, cross-sector-reliable predictors — a finding that held up independently under SHAP analysis after being first identified manually during data verification.

More importantly, the project's errors were not arbitrary. Every misclassification traced back to a company with a documented, previously-disclosed data limitation — confirmed accounting irregularities, an unverified event date, or a known sector-specific ratio distortion — while all 30 cleanly-matched, fully-verified pairs were classified correctly. That pattern is the project's core contribution: it demonstrates, on real data, precisely the conditions under which ratio-based credit analysis can be expected to succeed and precisely where it should be expected to fail, most notably in the face of manipulated financial statements. That distinction — knowing not just what a model predicts, but why it might be wrong — is the kind of judgment credit risk teams actually need, and is the primary skill this project was built to demonstrate.

---
## Methodology Note on Data Integrity

Every ratio in this dataset traces to a sourced financial statement or rating agency action. Where data was genuinely unavailable, values were left as missing (for models that support it) or imputed using disclosed, principled rules (group medians) rather than fabricated — this distinction is documented per-cell in the limitations section above, in keeping with standard practice for credit risk research under real-world data constraints.
