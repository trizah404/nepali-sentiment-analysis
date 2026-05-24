import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle
import sys
sys.path.append('src')
from preprocess import preprocess

def train(file_path: str):
    """Full training pipeline."""
    
    # Step 1 — Load and preprocess data
    print("Loading data...")
    df, text_col, label_col = preprocess(file_path)
    
    # Step 2 — Split into train and test
    X = df[text_col].tolist()
    y = df[label_col].tolist()
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\nTrain size: {len(X_train)}")
    print(f"Test size: {len(X_test)}")
    
    # Step 3 — Embed text using multilingual model
    print("\nLoading embedding model...")
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    
    print("Embedding training data...")
    X_train_embeddings = model.encode(X_train, show_progress_bar=True)
    
    print("Embedding test data...")
    X_test_embeddings = model.encode(X_test, show_progress_bar=True)
    
    # Step 4 — Train classifier
    print("\nTraining classifier...")
    classifier = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
    classifier.fit(X_train_embeddings, y_train)
    
    # Step 5 — Save model and embeddings
    print("\nSaving model...")
    with open('src/classifier.pkl', 'wb') as f:
        pickle.dump(classifier, f)
    
    np.save('src/X_test_embeddings.npy', X_test_embeddings)
    np.save('src/y_test.npy', np.array(y_test))
    
    print("Training complete. Model saved to src/classifier.pkl")
    return classifier, X_test_embeddings, y_test

if __name__ == "__main__":
    train("data/sentiment_analysis_nepali_final.csv")