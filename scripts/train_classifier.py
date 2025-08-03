"""
Train the AI job classifier from labeled data
"""
import argparse
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
import time

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.text_cleaner import preprocess_dataframe
from src.feature_combiner import prepare_features

def train_classifier(data_path, output_path='models/ai_job_classifier.pkl', 
                    test_size=0.2, val_size=0.25, random_seed=42):
    """
    Train the job classifier
    
    Args:
        data_path: Path to CSV with labeled data
        output_path: Where to save the trained model
        test_size: Proportion for test set
        val_size: Proportion for validation from remaining data
        random_seed: Random seed for reproducibility
    """
    print("Loading data...")
    df = pd.read_csv(data_path)
    
    # Clean text
    print("Preprocessing text...")
    df = preprocess_dataframe(df)
    
    # Filter short descriptions
    MIN_DESC_LENGTH = 50  # Reasonable minimum for a job description
    df_clean = df[df['desc_len'] >= MIN_DESC_LENGTH].copy()
    print(f"Filtered to {len(df_clean)} samples (removed {len(df) - len(df_clean)} short descriptions)")
    
    # Prepare features
    X = df_clean[['Job_Title_Clean', 'Job_Description_Clean']]
    y = df_clean['Label']
    
    # Split data
    print("Splitting data...")
    # First split: 80% train+val, 20% test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_seed, stratify=y  # Stratified splitting means preserving the same proportion of each class in all splits. Takes the labels as input, stratify=y works by grouping data by class labels before splitting
    )
    
    # Second split: From the 80%, make 75% train and 25% val
    # This gives us 60% train, 20% val, 20% test overall
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size, random_state=random_seed, stratify=y_temp
    )
    
    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # prepare combined text
    X_train_combined = prepare_features(X_train)
    X_val_combined = prepare_features(X_val)
    X_test_combined = prepare_features(X_test)
    
    # Create pipeline
    print("Training TF-IDF + Logistic Regression...")
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            max_features=1000,  # Limit features
            ngram_range=(1,2),  #Uni and Bigram
            min_df=2,  # Ignore terms in less than 2 docs
            max_df=0.95,  # Ignore terms that appear in 95% of docs
            strip_accents='unicode',  #Normalize to unicode
            lowercase=True,
            token_pattern=r'\b[a-zA-Z]{2,}\b'  # Only words with 2 or more letters
        )),
        ('classifier', LogisticRegression(
            class_weight='balanced',  #Handle imbalance
            random_state=random_seed,
            max_iter=1000,
            C=1.0  #C is the inverse of regularization strength.  Default is the l2 ridge regularization, minimizes feature without getting exactly zero
        ))
    ])
    
    # Train the model
    start_time = time.time()
    pipeline.fit(X_train_combined, y_train)
    train_time = time.time() - start_time
    print(f"Training complete! (took {train_time:.2f} seconds)")
    
    # Make predictions on validation set
    y_val_pred = pipeline.predict(X_val_combined)
    y_val_proba = pipeline.predict_proba(X_val_combined)
    
    # Evaluate on validation set
    print("\n" + "="*50)
    print("VALIDATION SET PERFORMANCE")
    print("="*50)
    print("\nClassification Report:")
    print(classification_report(y_val, y_val_pred,
                              target_names=['Other Jobs (0)', 'AI/ML/DS Jobs (1)'],  # Correct order
                              digits=3))
    
    # Find optimal threshold
    from sklearn.metrics import precision_recall_curve
    
    # Get probabilities for positive class (AI jobs)
    y_val_proba_ai = y_val_proba[:,1]
    
    # Calculate precision-recall curve
    precision, recall, thresholds = precision_recall_curve(y_val, y_val_proba_ai)
    
    # Find threshold that maximizes the F1 score
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
    
    best_threshold_idx = np.argmax(f1_scores[:-1])  # Exclude the last point
    best_threshold = thresholds[best_threshold_idx]
    best_f1 = f1_scores[best_threshold_idx]
    
    print(f"\nCurrent threshold: 0.5")
    print(f"Optimal threshold: {best_threshold:.3f}")
    print(f"F1 at optimal threshold: {best_f1:.3f}")
    
    # Test set evaluation
    print("\n" + "="*60)
    print("FINAL TEST SET PERFORMANCE")
    print("="*60)
    
    # Prepare test data
    y_test_pred = pipeline.predict(X_test_combined)
    y_test_proba = pipeline.predict_proba(X_test_combined)[:, 1]  # Probability of AI job
    
    # Apply optimal threshold
    y_test_pred_optimal = (y_test_proba >= best_threshold).astype(int)
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_test_pred_optimal,
                              target_names=['Other Jobs (0)', 'AI/ML/DS Jobs (1)'],  # Correct order
                              digits=3))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_test_pred_optimal)
    print("\nConfusion Matrix:")
    print("                   Predicted")
    print("                   Other  AI")
    print(f"Actual Other    {cm[0][0]:4d}  {cm[0][1]:4d}")
    print(f"Actual AI       {cm[1][0]:4d}  {cm[1][1]:4d}")
    
    # Calculate key metrics for AI jobs
    precision_ai = cm[1][1] / (cm[0][1] + cm[1][1]) if (cm[0][1] + cm[1][1]) > 0 else 0
    recall_ai = cm[1][1] / (cm[1][0] + cm[1][1]) if (cm[1][0] + cm[1][1]) > 0 else 0
    f1_ai = 2 * (precision_ai * recall_ai) / (precision_ai + recall_ai) if (precision_ai + recall_ai) > 0 else 0
    
    print(f"\nKey Metrics for AI/ML/DS Jobs:")
    print(f"Precision: {precision_ai:.3f}")
    print(f"Recall: {recall_ai:.3f}")
    print(f"F1-Score: {f1_ai:.3f}")
    
    # Save model and threshold
    model_package = {
        'pipeline': pipeline,
        'threshold': best_threshold,
        'performance': {
            'test_precision': precision_ai,
            'test_recall': recall_ai,
            'test_f1': f1_ai
        }
    }
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    joblib.dump(model_package, output_path)
    print(f"\nModel saved as '{output_path}'")
    print(f"Optimal threshold: {model_package['threshold']}")
    print(f"Expected performance: {model_package['performance']}")
    
    return model_package

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train AI job classifier')
    parser.add_argument('--input', required=True, help='Path to labeled CSV data')
    parser.add_argument('--output', default='models/ai_job_classifier.pkl', 
                       help='Output path for trained model')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    train_classifier(args.input, args.output, random_seed=args.seed)
