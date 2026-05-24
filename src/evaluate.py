import numpy as np
import pickle
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate():
    """Evaluate the trained classifier."""
    
    # Load saved model and test data
    with open('src/classifier.pkl', 'rb') as f:
        classifier = pickle.load(f)
    
    X_test = np.load('src/X_test_embeddings.npy')
    y_test = np.load('src/y_test.npy')
    
    # Make predictions
    y_pred = classifier.predict(X_test)
    
    # Accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Detailed report
    print("\nClassification Report:")
    print(classification_report(
        y_test, y_pred,
        target_names=['Negative (-1)', 'Neutral (0)', 'Positive (1)']
    ))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=['Negative', 'Neutral', 'Positive'],
        yticklabels=['Negative', 'Neutral', 'Positive']
    )
    plt.title('Confusion Matrix — Nepali Sentiment Analysis')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig('src/confusion_matrix.png')
    print("\nConfusion matrix saved to src/confusion_matrix.png")

if __name__ == "__main__":
    evaluate()