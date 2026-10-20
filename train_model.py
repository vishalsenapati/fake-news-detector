"""
Phase 1: Baseline ML Classifier for Fake News Detection
TF-IDF + Logistic Regression with evaluation metrics and model export

This version works with sample data or local CSV file.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, 
    recall_score, f1_score, classification_report
)
import joblib
import os

# Set style for plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 8)


def create_sample_dataset(n_samples=5000):
    """Create a realistic synthetic dataset for demo purposes"""
    np.random.seed(42)
    
    # Real news templates
    real_templates = [
        "Government announces new policy to {action} {topic} affecting millions of citizens",
        "Scientists discover breakthrough in {topic} research that could {action}",
        "Stock market {action} as {topic} reports quarterly earnings",
        "New study reveals {topic} linked to {action} according to experts",
        "International summit addresses {topic} with {action} agreement",
        "Technology company launches innovative {topic} that can {action}",
        "Health officials recommend {action} to prevent {topic} outbreak",
        "Economic indicators show {topic} is expected to {action} next quarter",
    ]
    
    # Fake news templates (sensational, clickbait, conspiratorial)
    fake_templates = [
        "SHOCKING: {topic} secretly {action} - mainstream media HIDES the truth!",
        "You won't BELIEVE what {topic} is doing to {action} your children!",
        "BREAKING: {topic} exposed in massive conspiracy to {action} - government covers it up!",
        "MIRACLE CURE: {topic} can {action} overnight - doctors HATE this!",
        "Secret documents reveal {topic} plans to {action} - share before DELETED!",
        "EXCLUSIVE: {topic} caught on camera {action} - authorities SILENCED the witness!",
        "This ONE weird trick {topic} uses to {action} - Big Pharma FURIOUS!",
        "URGENT: {topic} will {action} tomorrow - prepare NOW or regret forever!",
    ]
    
    topics = [
        "COVID-19 vaccines", "climate change", "cryptocurrency", "artificial intelligence",
        "election results", "5G networks", "pharmaceutical companies", "government surveillance",
        "food supply", "education system", "social media platforms", "renewable energy"
    ]
    
    actions_real = [
        "improve", "reduce", "increase", "address", "investigate", "support", "regulate", "fund"
    ]
    
    actions_fake = [
        "control", "destroy", "manipulate", "poison", "enslave", "brainwash", "eliminate", "weaponize"
    ]
    
    real_texts = []
    fake_texts = []
    
    for _ in range(n_samples // 2):
        # Real news
        template = np.random.choice(real_templates)
        topic = np.random.choice(topics)
        action = np.random.choice(actions_real)
        text = template.format(topic=topic, action=action)
        # Add some realistic variation
        text += f" According to {np.random.choice(['Reuters', 'AP News', 'BBC', 'The Guardian', 'NYTimes'])} sources."
        real_texts.append(text)
        
        # Fake news
        template = np.random.choice(fake_templates)
        topic = np.random.choice(topics)
        action = np.random.choice(actions_fake)
        text = template.format(topic=topic, action=action)
        fake_texts.append(text)
    
    # Combine
    texts = real_texts + fake_texts
    labels = [0] * (n_samples // 2) + [1] * (n_samples // 2)
    
    # Shuffle
    indices = np.random.permutation(n_samples)
    texts = [texts[i] for i in indices]
    labels = [labels[i] for i in indices]
    
    df = pd.DataFrame({'text': texts, 'label': labels})
    return df


def load_dataset(csv_path=None):
    """Load dataset from CSV or create sample"""
    if csv_path and os.path.exists(csv_path):
        print(f"Loading dataset from {csv_path}...")
        df = pd.read_csv(csv_path)
        return preprocess_data(df)
    else:
        print("Creating sample dataset for demonstration...")
        df = create_sample_dataset(10000)
        return preprocess_data(df)


def preprocess_data(df):
    """Clean and prepare the dataset"""
    print("\nPreprocessing data...")
    print(f"Original shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Find text and label columns
    text_cols = [c for c in df.columns if c.lower() in ['text', 'title', 'content', 'article', 'news', 'headline']]
    label_cols = [c for c in df.columns if c.lower() in ['label', 'class', 'target', 'category', 'fake', 'real']]
    
    print(f"Text columns found: {text_cols}")
    print(f"Label columns found: {label_cols}")
    
    if len(text_cols) > 0:
        text_col = text_cols[0]
    else:
        # Assume first object column is text
        obj_cols = df.select_dtypes(include=['object']).columns
        if len(obj_cols) > 0:
            text_col = obj_cols[0]
        else:
            raise ValueError("No text column found")
    
    if len(label_cols) > 0:
        label_col = label_cols[0]
    else:
        # Assume last column is label
        label_col = df.columns[-1]
    
    print(f"Using text column: {text_col}")
    print(f"Using label column: {label_col}")
    
    # Clean labels - map to binary (0 = real, 1 = fake)
    unique_labels = df[label_col].unique()
    print(f"Unique labels: {unique_labels}")
    
    label_map = {}
    for label in unique_labels:
        label_lower = str(label).lower()
        if ('fake' in label_lower or 'unreliable' in label_lower or 
            'false' in label_lower or label == 1 or label == '1' or
            label == 'FAKE' or label == 'unreliable'):
            label_map[label] = 1
        else:
            label_map[label] = 0
    
    df['binary_label'] = df[label_col].map(label_map)
    
    # Remove rows with NaN
    df = df.dropna(subset=[text_col, 'binary_label'])
    df['binary_label'] = df['binary_label'].astype(int)
    
    print(f"Final shape: {df.shape}")
    print(f"Label distribution:\n{df['binary_label'].value_counts()}")
    
    return df[text_col], df['binary_label']


def train_and_evaluate(X, y):
    """Train TF-IDF + Logistic Regression and evaluate"""
    print("\nSplitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    
    print("\nTraining TF-IDF + Logistic Regression...")
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 2),
            stop_words='english',
            min_df=2,
            max_df=0.95
        )),
        ('clf', LogisticRegression(
            max_iter=1000,
            random_state=42,
            class_weight='balanced'
        ))
    ])
    
    pipeline.fit(X_train, y_train)
    
    print("\nEvaluating on test set...")
    y_pred = pipeline.predict(X_test)
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"\n{'='*50}")
    print(f"MODEL PERFORMANCE METRICS")
    print(f"{'='*50}")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-Score : {f1:.4f}")
    print(f"{'='*50}")
    
    # Classification report
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Real', 'Fake']))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"\nConfusion Matrix:")
    print(cm)
    print(f"\nTN: {cm[0,0]}, FP: {cm[0,1]}")
    print(f"FN: {cm[1,0]}, TP: {cm[1,1]}")
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'],
                cbar_kws={'label': 'Count'})
    plt.title('Confusion Matrix - Fake News Detection (TF-IDF + Logistic Regression)', fontsize=14)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    print("\nConfusion matrix saved as: confusion_matrix.png")
    plt.close()
    
    return pipeline


def save_model(pipeline, filepath='fake_news_model.joblib'):
    """Export the trained model using joblib"""
    print(f"\nSaving model to {filepath}...")
    joblib.dump(pipeline, filepath)
    print(f"Model saved successfully!")
    
    # Also save the vectorizer and classifier separately for inspection
    joblib.dump(pipeline.named_steps['tfidf'], 'tfidf_vectorizer.joblib')
    joblib.dump(pipeline.named_steps['clf'], 'logistic_regression.joblib')
    print("Individual components saved as well.")


def main(csv_path=None):
    print("="*60)
    print("FAKE NEWS DETECTOR - PHASE 1: BASELINE ML CLASSIFIER")
    print("TF-IDF + Logistic Regression")
    print("="*60)
    
    # Load dataset
    X, y = load_dataset(csv_path)
    
    # Train and evaluate
    pipeline = train_and_evaluate(X, y)
    
    # Save model
    save_model(pipeline, 'fake_news_model.joblib')
    
    print("\n" + "="*60)
    print("PHASE 1 COMPLETE!")
    print("Model exported as: fake_news_model.joblib")
    print("Vectorizer exported as: tfidf_vectorizer.joblib")
    print("Classifier exported as: logistic_regression.joblib")
    print("Confusion matrix saved as: confusion_matrix.png")
    print("="*60)


if __name__ == "__main__":
    import sys
    csv_path = sys.argv[1] if len(sys.argv) > 1 else None
    main(csv_path)