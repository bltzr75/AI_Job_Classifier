# Threshold Analysis Results

## Optimal Threshold Selection

The default logistic regression threshold of 0.5 was optimized using the validation set to maximize F1-score for AI/ML job detection.

### Validation Set Analysis

| Threshold | Precision | Recall | F1-Score | AI Jobs Found |
|-----------|-----------|--------|----------|---------------|
| 0.300 | 0.589 | 0.951 | 0.727 | 197 |
| 0.400 | 0.643 | 0.902 | 0.751 | 171 |
| **0.471** | **0.718** | **0.877** | **0.790** | **149** |
| 0.500 | 0.710 | 0.803 | 0.754 | 138 |
| 0.600 | 0.754 | 0.754 | 0.754 | 122 |
| 0.700 | 0.788 | 0.639 | 0.706 | 99 |

**Optimal threshold: 0.471** (maximizes F1-score)

### Test Set Performance Comparison

| Metric | Default (0.5) | Optimal (0.471) |
|--------|---------------|-----------------|
| Precision | 0.736 | 0.725 |
| Recall | 0.869 | 0.910 |
| F1-Score | 0.797 | 0.807 |
| Jobs to Review | 144 | 153 |

## Business Impact

Using the optimal threshold of 0.471:
- **9 more jobs** to review (153 vs 144)
- **5 fewer missed AI jobs** (11 vs 16)
- **4.1% improvement** in recall (91.0% vs 86.9%)
- **1.0% improvement** in F1-score (80.7% vs 79.7%)

## Model Comparison Notes

**Model chosen: TF-IDF + Logistic Regression**
- **Probability calibration**: LR gives reliable probability scores for threshold tuning. LinearSVC doesn't have native probabilities.
- **1% gain not worth it**: 0.754 → 0.764 F1 (SVC) is negligible in practice
- **Proven on test set**: Already validated at 79.7% F1
- **Simpler deployment**: No need for calibration wrappers

## Recommendation

Use threshold of **0.471** for production deployment to maximize the detection of AI/ML jobs while maintaining reasonable precision.
