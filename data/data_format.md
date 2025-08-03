# Data Format Specification

## Expected Input Format

The classifier expects a CSV file with the following columns:

| Column | Type | Description | Required |
|--------|------|-------------|----------|
| Job_Title | string | Job title/position name | Yes |
| Job_Description | string | Full job description text | Yes |
| Label | integer | Classification label (1=AI/ML, 0=Other) | For training only |

## Label Definitions
- **1**: AI/ML/GenAI/LLM engineering roles
- **0**: Other tech roles (including general software engineering, DevOps, etc.)

## Data Preprocessing

Before classification, the system will:
1. Clean job titles and descriptions (remove "About the job" prefixes)
2. Normalize whitespace and punctuation
3. Remove bullet points while preserving text
4. Combine title (3x weight) with description for feature extraction
5. Apply TF-IDF vectorization with unigrams and bigrams

## Minimum Requirements
- Job descriptions should be at least 50 characters
- Text should be in English
- UTF-8 encoding recommended

## Output Format

The classifier returns:
```json
{
    "is_ai_job": boolean,
    "confidence": float (0-1),
    "decision": string ("AI/ML/NLP Job" or "Other Job")
}
```

## Sample Data Structure
```csv
Job_Title,Job_Description,Label
"Machine Learning Engineer","Build and deploy ML models...",1
"Software Engineer","Develop web applications...",0
```
