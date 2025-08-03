"""
Make predictions using the trained classifier
"""
import argparse
import pandas as pd
import joblib
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.text_cleaner import clean_text_for_classification
from src.feature_combiner import combine_text

def predict_ai_job(title, description, model_package):
    """
    Predict if a job is AI/ML/NLP related
    
    Args:
        title: Job title
        description: Job description
        model_package: Loaded model package with pipeline and threshold
    
    Returns:
        Dictionary with prediction results
    """
    # Clean inputs
    title_clean = clean_text_for_classification(title, is_title=True)
    desc_clean = clean_text_for_classification(description, is_title=False)
    
    # Combine text
    row = {'Job_Title_Clean': title_clean, 'Job_Description_Clean': desc_clean}
    text = combine_text(row)
    
    # Get probability
    prob = model_package['pipeline'].predict_proba([text])[0, 1]
    
    # Apply threshold
    threshold = model_package.get('threshold', 0.5)
    is_ai_job = prob >= threshold
    
    return {
        'is_ai_job': bool(is_ai_job),
        'confidence': float(prob),
        'decision': 'AI/ML/NLP Job' if is_ai_job else 'Other Job'
    }

def predict_batch(csv_path, model_path, output_path=None):
    """
    Make predictions on a batch of jobs from CSV
    
    Args:
        csv_path: Path to CSV with job data
        model_path: Path to saved model
        output_path: Where to save predictions (optional)
    """
    # Load model
    print(f"Loading model from {model_path}...")
    model_package = joblib.load(model_path)
    
    # Load data
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Make predictions
    print("Making predictions...")
    predictions = []
    
    for idx, row in df.iterrows():
        title = row.get('Job_Title', '')
        description = row.get('Job_Description', '')
        
        result = predict_ai_job(title, description, model_package)
        predictions.append(result)
    
    # Add predictions to dataframe
    df['Predicted_Label'] = [p['is_ai_job'] for p in predictions]
    df['Confidence'] = [p['confidence'] for p in predictions]
    df['Decision'] = [p['decision'] for p in predictions]
    
    # Save results
    if output_path:
        df.to_csv(output_path, index=False)
        print(f"Predictions saved to {output_path}")
    
    # Print summary
    ai_jobs = sum(1 for p in predictions if p['is_ai_job'])
    print(f"\nSummary:")
    print(f"Total jobs: {len(predictions)}")
    print(f"AI/ML jobs: {ai_jobs} ({ai_jobs/len(predictions)*100:.1f}%)")
    print(f"Other jobs: {len(predictions) - ai_jobs} ({(len(predictions) - ai_jobs)/len(predictions)*100:.1f}%)")
    
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict job classifications')
    
    subparsers = parser.add_subparsers(dest='mode', help='Prediction mode')
    
    # Single prediction
    single_parser = subparsers.add_parser('single', help='Predict single job')
    single_parser.add_argument('--title', required=True, help='Job title')
    single_parser.add_argument('--description', required=True, help='Job description')
    single_parser.add_argument('--model', default='models/ai_job_classifier.pkl', 
                              help='Path to model')
    
    # Batch prediction
    batch_parser = subparsers.add_parser('batch', help='Predict from CSV')
    batch_parser.add_argument('--input', required=True, help='Input CSV path')
    batch_parser.add_argument('--model', default='models/ai_job_classifier.pkl',
                             help='Path to model')
    batch_parser.add_argument('--output', help='Output CSV path (optional)')
    
    args = parser.parse_args()
    
    if args.mode == 'single':
        # Load the model
        model_package = joblib.load(args.model)
        
        # Test it
        test_job = predict_ai_job(
            args.title,
            args.description,
            model_package
        )
        print(test_job)
    
    elif args.mode == 'batch':
        predict_batch(args.input, args.model, args.output)
    
    else:
        parser.print_help()
