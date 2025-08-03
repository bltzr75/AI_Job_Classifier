# AI Job Classifier

A machine learning project to automatically classify job postings as AI/ML/GenAI-related or other tech roles.

## 🎯 Overview

This classifier helps identify AI/ML/GenAI engineering roles from general tech job postings with high accuracy:
- **Precision**: 72.5% (correctly identifies AI roles)
- **Recall**: 91.0% (catches most AI roles)
- **F1-Score**: 80.7% (balanced performance)

Built using TF-IDF + Logistic Regression on 2,794 manually labeled job postings.

## 📊 Performance Metrics

### Test Set Results (559 samples)
| Metric | Other Jobs | AI/ML Jobs |
|--------|-----------|------------|
| Precision | 0.973 | 0.725 |
| Recall | 0.904 | 0.910 |
| F1-Score | 0.937 | 0.807 |
| Support | 437 | 122 |

### Confusion Matrix
```
                 Predicted
                 Other  AI
Actual Other     395    42
Actual AI         11   111
```

## 🚀 Quick Start

### Installation
```bash
# Clone repository

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage
```python
import joblib

# Load the model
model_package = joblib.load('models/ai_job_classifier.pkl')

# Classify a job
from scripts.predict import predict_ai_job

result = predict_ai_job(
    title="Machine Learning Engineer",
    description="Build NLP models using transformers...",
    model_package=model_package
)

print(result)
# Output: {'is_ai_job': True, 'confidence': 0.847, 'decision': 'AI/ML/NLP Job'}
```

## 📁 Project Structure
```
├── data/                 # Sample data (5 rows)
│   ├── sample_data.csv
│   └── data_format.md
├── src/                  # Core modules
│   ├── text_cleaner.py
│   ├── feature_combiner.py
│   └── label_merger.py
├── scripts/              # Executable scripts
│   ├── train_classifier.py
│   └── predict.py
├── notebooks/            # Jupyter notebooks
├── prompts/              # LLM prompts used
├── models/               # Trained models
└── results/              # Performance metrics
```

## 🔧 Training Your Own Model

1. Prepare your data in CSV format with columns: `Job_Title`, `Job_Description`, `Label` (1=AI/ML, 0=Other)
2. Run the training script:
   ```bash
   python scripts/train_classifier.py --input data/your_data.csv --output models/classifier.pkl
   ```

## 📈 Model Development Process

1. **Data Collection**: Scraped 2,810 job postings
2. **Labeling**: Used Claude Sonnet to classify jobs in batches of 50
3. **Preprocessing**: Cleaned text, removed short descriptions (<50 chars)
4. **Feature Engineering**: Combined title (3x weight) + description, TF-IDF with bigrams
5. **Model Selection**: Tested Logistic Regression, Naive Bayes, and LinearSVC
6. **Optimization**: Tuned decision threshold from 0.5 to 0.471 for better F1

## 🛡️ Privacy & Ethics

- Sample data has been anonymized


## 📝 License

MIT License

Copyright (c) 2024

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## 🤝 Contributing

Contributions welcome! Please ensure any data contributions are properly anonymized.
