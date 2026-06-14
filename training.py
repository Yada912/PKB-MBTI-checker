import csv
import re
import os
import sys
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, f1_score, roc_auc_score

# Paths
DATASET_PATH = "dataset/mbti_2.csv"
MODEL_SAVE_PATH = "mbti_models.joblib"
VECTORIZER_SAVE_PATH = "tfidf_vectorizer.joblib"

# 16 MBTI Types and their plurals for masking
MBTI_TYPES = [
    'infj', 'enfp', 'intj', 'entp', 'intp', 'infp', 'entj', 'enfj',
    'isfj', 'istj', 'isfp', 'istp', 'estp', 'esfp', 'estj', 'esfj'
]
MASK_REGEX = re.compile(r'\b(' + '|'.join(MBTI_TYPES) + r')s?\b', re.IGNORECASE)

def clean_and_mask_text(text):
    """
    Cleans text by removing URLs, MBTI type mentions (to avoid data leakage),
    extra whitespaces, and standardizing formatting.
    """
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    # Remove MBTI type mentions (e.g. INFJ, ENFPs, intj)
    text = MASK_REGEX.sub('[MASK]', text)
    
    # Remove HTML-like tags or noise
    text = re.sub(r'<[^>]+>', '', text)
    
    # Standardize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text.lower()

def chunk_text(posts, target_word_count=256):
    """
    Splits user posts (separated by |||) into chunks of approximately target_word_count words.
    This mimics the paper's methodology of 256-word fragments.
    """
    chunks = []
    current_chunk = []
    current_words = 0
    
    for post in posts:
        cleaned_post = clean_and_mask_text(post)
        if not cleaned_post:
            continue
            
        post_words = cleaned_post.split()
        post_word_count = len(post_words)
        
        if current_words + post_word_count > target_word_count and current_chunk:
            # Save current chunk and start a new one
            chunks.append(" ".join(current_chunk))
            current_chunk = post_words
            current_words = post_word_count
        else:
            current_chunk.extend(post_words)
            current_words += post_word_count
            
    # Append any remaining words as the last chunk
    if current_chunk:
        chunks.append(" ".join(current_chunk))
        
    return chunks

def load_data():
    """
    Loads dataset/mbti_2.csv, cleans and fragments the text,
    and returns processed lists of texts and binary labels.
    """
    print("Loading dataset...")
    if not os.path.exists(DATASET_PATH):
        print(f"Error: Dataset not found at {DATASET_PATH}")
        sys.exit(1)
        
    # Read the raw users and labels first
    users_data = []
    
    with open(DATASET_PATH, mode='r', encoding='utf-8') as f:
        # Increase field size limit for large CSV entries, using 32-bit max to avoid Windows overflow
        csv.field_size_limit(2147483647)
        reader = csv.DictReader(f)
        for row in reader:
            # Split posts by |||
            posts = row['text'].split('|||')
            
            # Binary labels mapping: I/E, N/S, F/T, J/P
            # We convert them to integers
            labels = {
                'I/E': int(row['I/E']),
                'N/S': int(row['N/S']),
                'F/T': int(row['F/T']),
                'J/P': int(row['J/P'])
            }
            users_data.append((posts, labels))
            
    print(f"Loaded {len(users_data)} users. Splitting into train and test sets to avoid data leakage...")
    
    # Split users first to avoid data leakage between chunks of the same user
    train_users, test_users = train_test_split(users_data, test_size=0.2, random_state=42)
    
    # Process training and test sets separately
    X_train_list, y_train_list = [], {'I/E': [], 'N/S': [], 'F/T': [], 'J/P': []}
    X_test_list, y_test_list = [], {'I/E': [], 'N/S': [], 'F/T': [], 'J/P': []}
    
    print("Processing and chunking training set...")
    for posts, labels in train_users:
        chunks = chunk_text(posts, target_word_count=256)
        for chunk in chunks:
            if len(chunk.split()) >= 10:  # Ignore very short chunks
                X_train_list.append(chunk)
                for trait in labels:
                    y_train_list[trait].append(labels[trait])
                    
    print("Processing and chunking test set...")
    for posts, labels in test_users:
        chunks = chunk_text(posts, target_word_count=256)
        for chunk in chunks:
            if len(chunk.split()) >= 10:
                X_test_list.append(chunk)
                for trait in labels:
                    y_test_list[trait].append(labels[trait])
                    
    # Convert labels to numpy arrays
    y_train = {trait: np.array(y_train_list[trait]) for trait in y_train_list}
    y_test = {trait: np.array(y_test_list[trait]) for trait in y_test_list}
    
    print(f"Created {len(X_train_list)} training chunks and {len(X_test_list)} test chunks.")
    return X_train_list, X_test_list, y_train, y_test

def main():
    X_train, X_test, y_train, y_test = load_data()
    
    # 2. Vectorization
    print("Fitting TF-IDF Vectorizer...")
    vectorizer = TfidfVectorizer(
        max_features=12000,
        ngram_range=(1, 2),
        min_df=3,
        stop_words='english',
        sublinear_tf=True
    )
    
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    # Save Vectorizer
    print("Saving Vectorizer...")
    joblib.dump(vectorizer, VECTORIZER_SAVE_PATH)
    
    # 3. Model Training & Evaluation
    models = {}
    traits = ['I/E', 'N/S', 'F/T', 'J/P']
    
    # Explain mapping
    trait_names = {
        'I/E': 'Introvert (1) vs Extrovert (0)',
        'N/S': 'Intuitive (1) vs Sensing (0)',
        'F/T': 'Feeling (1) vs Thinking (0)',
        'J/P': 'Judging (1) vs Perceiving (0)'
    }
    
    print("\n=== Training 4 Trait Classifiers ===")
    for trait in traits:
        print(f"\nTraining classifier for {trait_names[trait]}...")
        
        # We use class_weight='balanced' to handle any dataset imbalance (especially for N/S)
        clf = LogisticRegression(
            C=1.0, 
            max_iter=1000, 
            class_weight='balanced', 
            random_state=42, 
            n_jobs=-1
        )
        clf.fit(X_train_tfidf, y_train[trait])
        
        # Evaluate
        preds = clf.predict(X_test_tfidf)
        probs = clf.predict_proba(X_test_tfidf)[:, 1]
        
        acc = accuracy_score(y_test[trait], preds)
        f1 = f1_score(y_test[trait], preds)
        auc = roc_auc_score(y_test[trait], probs)
        
        print(f"Evaluation for {trait}:")
        print(f"  Accuracy : {acc:.4f}")
        print(f"  F1 Score : {f1:.4f}")
        print(f"  AUC ROC  : {auc:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test[trait], preds, target_names=["0 (Ext/Sens/Think/Perc)", "1 (Int/Intuit/Feel/Judg)"]))
        
        models[trait] = clf
        
        # Display top 10 positive and negative words (Feature Coefficients)
        feature_names = np.array(vectorizer.get_feature_names_out())
        coefs = clf.coef_[0]
        sorted_indices = np.argsort(coefs)
        
        neg_label = trait_names[trait].split(' vs ')[1].split(' ')[0]
        pos_label = trait_names[trait].split(' vs ')[0].split(' ')[0]
        
        print(f"Top 10 words associated with {neg_label} (negative coefficients):")
        print(", ".join(feature_names[sorted_indices[:10]]))
        print(f"Top 10 words associated with {pos_label} (positive coefficients):")
        print(", ".join(feature_names[sorted_indices[-10:][::-1]]))
        
    # Save Models
    print("\nSaving Models...")
    joblib.dump(models, MODEL_SAVE_PATH)
    print("Training pipeline finished successfully!")

if __name__ == "__main__":
    main()
