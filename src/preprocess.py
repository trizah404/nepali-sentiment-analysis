import pandas as pd
import re

def load_data(file_path: str) -> pd.DataFrame:
    """Load the dataset from CSV."""
    df = pd.read_csv(file_path)
    print(f"Loaded {len(df)} rows")
    print(f"Columns: {df.columns.tolist()}")
    print(f"\nFirst 5 rows:")
    print(df.head())
    print(f"\nLabel distribution:")
    print(df.iloc[:, -1].value_counts())
    return df

def clean_text(text: str) -> str:
    """Clean Nepali text."""
    if not isinstance(text, str):
        return ""
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def preprocess(file_path: str):
    """Full preprocessing pipeline."""
    df = load_data(file_path)
    
    # Get column names
    text_col = 'Sentences'
    label_col = 'Sentiment'
    
    # Clean text
    df[text_col] = df[text_col].apply(clean_text)
    
    # Drop empty rows
    df = df[df[text_col].str.len() > 0]
    
    print(f"\nAfter cleaning: {len(df)} rows")
    return df, text_col, label_col

if __name__ == "__main__":
    df, text_col, label_col = preprocess("data/sentiment_analysis_nepali_final.csv")