# Evaluation Report

> Probability-based decision-support tool. **Not** a guarantee of any individual deal outcome.

**Best model:** Logistic Regression  
**Test-set size:** 1000  
**Class balance (completed):** 65.8%

## Model comparison (ranked by test ROC-AUC)

| model               |   cv_roc_auc_mean |   cv_roc_auc_std |   test_accuracy |   test_precision |   test_recall |   test_f1 |   test_roc_auc |
|:--------------------|------------------:|-----------------:|----------------:|-----------------:|--------------:|----------:|---------------:|
| Logistic Regression |             0.865 |            0.013 |           0.799 |            0.884 |         0.799 |     0.84  |          0.879 |
| Gradient Boosting   |             0.862 |            0.012 |           0.807 |            0.831 |         0.888 |     0.858 |          0.876 |
| Random Forest       |             0.847 |            0.017 |           0.813 |            0.855 |         0.862 |     0.858 |          0.868 |
| XGBoost             |             0.853 |            0.014 |           0.805 |            0.833 |         0.88  |     0.856 |          0.866 |
| KNN (baseline)      |             0.815 |            0.02  |           0.782 |            0.792 |         0.907 |     0.846 |          0.834 |
| Decision Tree       |             0.773 |            0.011 |           0.787 |            0.845 |         0.828 |     0.837 |          0.801 |

## Best-model test metrics

```
               precision    recall  f1-score   support

   Failed (0)      0.674     0.798     0.731       342
Completed (1)      0.884     0.799     0.840       658

     accuracy                          0.799      1000
    macro avg      0.779     0.799     0.785      1000
 weighted avg      0.812     0.799     0.802      1000

```

**ROC-AUC:** 0.879

## Confusion matrix & error analysis

|                       | Predicted Failed | Predicted Completed |
|-----------------------|------------------|---------------------|
| **Actually Failed**   | 273 (TN)        | 69 (FP)           |
| **Actually Completed**| 132 (FN)        | 526 (TP)           |

- **False Positives (predict *complete*, deal actually fails): 69** — rate 20.2% of truly-failed deals. This is the costliest error: a user could rely on a deal that ultimately collapses.
- **False Negatives (predict *fail*, deal actually completes): 132** — rate 20.1% of truly-completed deals.

Because false positives are the expensive error, the prediction interface lets you raise the completion-probability threshold above 0.5 to trade recall for precision on the 'will complete' call.

## Figures

- `figures/confusion_matrix.png`
- `figures/roc_curve.png`
- `figures/precision_recall.png`
- `figures/feature_importance.png` (from explainability stage)
- `figures/election_cancellation_rates.png`
- `figures/shap_summary.png` (if SHAP available)