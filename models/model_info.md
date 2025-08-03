# Model Information

## Model Architecture
- **Algorithm**: TF-IDF + Logistic Regression
- **Pipeline**: sklearn Pipeline with TfidfVectorizer and LogisticRegression

## TF-IDF Parameters
- **max_features**: 1000 (Limit features)
- **ngram_range**: (1, 2) - Uni and Bigram
- **min_df**: 2 - Ignore terms in less than 2 docs
- **max_df**: 0.95 - Ignore terms that appear in 95% of docs
- **strip_accents**: unicode - Normalize to unicode
- **lowercase**: True
- **token_pattern**: r'\b[a-zA-Z]{2,}\b' - Only words with 2 or more letters

## Logistic Regression Parameters
- **class_weight**: balanced - Handle imbalance
- **random_state**: 42
- **max_iter**: 1000
- **C**: 1.0 - C is the inverse of regularization strength. Default is the l2 ridge regularization, minimizes feature without getting exactly zero

## Feature Engineering
- Title weight: 3x (Title appears 3 times to give it more weight)
- Text preprocessing: Remove prefixes, normalize whitespace, remove bullets
- Minimum description length: 50 characters (Reasonable minimum for a job description)

## Training Data
- Total samples: 2,794 (after filtering)
- Class distribution: 78.1% Other, 21.9% AI/ML
- Train/Val/Test split: 60%/20%/20%
- Stratified splitting to preserve class proportions

## Model Files
- **ai_job_classifier.pkl**: Contains dictionary with:
  - 'pipeline': Trained sklearn Pipeline
  - 'threshold': Optimal decision threshold (0.471)
  - 'performance': Test metrics

## Usage
```python
import joblib
model_package = joblib.load('ai_job_classifier.pkl')
pipeline = model_package['pipeline']
threshold = model_package['threshold']
```

## Notes
- Craete dataframes with all necessary cols (kept original typo)
- prepare combined text for feature engineering
- Model chosen over alternatives due to better probability calibration
