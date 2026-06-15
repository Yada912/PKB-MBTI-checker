import csv
import re
import os
import sys
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, accuracy_score, f1_score, roc_auc_score,
    roc_curve, precision_recall_curve, confusion_matrix, ConfusionMatrixDisplay
)

import matplotlib
matplotlib.use('Agg')  # Headless mode for Matplotlib
import matplotlib.pyplot as plt

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

def generate_plots(X_train, X_test, y_train, y_test, vectorizer, models):
    """
    Generates scientific and functional evaluation plots and matching raw CSV data files
    for the NLP MBTI classifiers:
    1. Word count distribution of preprocessed text chunks.
    2. Class distributions showing the data imbalance across the 4 axes.
    3. ROC curves for all 4 dimensions on a single figure.
    4. Precision-Recall curves for all 4 dimensions.
    5. A 2x2 grid of normalized Confusion Matrices.
    6. Feature importance subplots showing key linguistic markers.
    """
    PLOTS_DIR = "static/plots"
    os.makedirs(PLOTS_DIR, exist_ok=True)
    
    traits = ['I/E', 'N/S', 'F/T', 'J/P']
    trait_names = {
        'I/E': 'Introvert (1) vs Extrovert (0)',
        'N/S': 'Intuitive (1) vs Sensing (0)',
        'F/T': 'Feeling (1) vs Thinking (0)',
        'J/P': 'Judging (1) vs Perceiving (0)'
    }
    
    # Modern color palette (matching the PsychoNLP theme)
    colors = {
        'I/E': '#3b82f6',  # Blue
        'N/S': '#10b981',  # Emerald Green
        'F/T': '#ec4899',  # Pink
        'J/P': '#8b5cf6',  # Purple
        'Class0': '#f43f5e', # Rose Red
        'Class1': '#06b6d4'  # Cyan
    }
    
    # Prepare test data tf-idf
    X_test_tfidf = vectorizer.transform(X_test)
    
    print("\nGenerating analysis plots and raw CSV data files...")
    
    # ------------------ Plot 1: Word Count Distribution ------------------
    print("  Generating Plot & CSV 1: Word Count Distribution...")
    plt.figure(figsize=(10, 6))
    train_word_counts = [len(chunk.split()) for chunk in X_train]
    test_word_counts = [len(chunk.split()) for chunk in X_test]
    
    plt.hist(train_word_counts, bins=30, alpha=0.6, label='Train Chunks', color='#4f46e5', edgecolor='black')
    plt.hist(test_word_counts, bins=30, alpha=0.6, label='Test Chunks', color='#f43f5e', edgecolor='black')
    plt.title('Distribution of Words per Text Chunk (Fragmentation Verification)', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Word Count', fontsize=12)
    plt.ylabel('Frequency (Chunks)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'mbti_chunk_word_counts.png'), dpi=150)
    plt.close()
    
    # Save raw CSV
    with open(os.path.join(PLOTS_DIR, 'mbti_chunk_word_counts.csv'), mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['chunk_index', 'split', 'word_count'])
        for idx, wc in enumerate(train_word_counts):
            writer.writerow([idx, 'train', wc])
        for idx, wc in enumerate(test_word_counts):
            writer.writerow([idx + len(train_word_counts), 'test', wc])
            
    # ------------------ Plot 2: Class Distribution (Imbalance) ------------------
    print("  Generating Plot & CSV 2: Class Distribution...")
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()
    
    # Prep CSV list
    class_dist_data = []
    
    for i, trait in enumerate(traits):
        y_vals = y_test[trait]
        labels = [trait_names[trait].split(' vs ')[1].split(' ')[0], trait_names[trait].split(' vs ')[0].split(' ')[0]]
        counts = [np.sum(y_vals == 0), np.sum(y_vals == 1)]
        
        bars = axes[i].bar(labels, counts, color=[colors['Class0'], colors['Class1']], edgecolor='black', alpha=0.85, width=0.6)
        axes[i].set_title(f'Axis Distribution: {trait}', fontsize=12, fontweight='bold')
        axes[i].set_ylabel('Number of Samples')
        axes[i].grid(True, axis='y', linestyle='--', alpha=0.5)
        
        # Add labels on top of the bars
        for bar in bars:
            height = bar.get_height()
            axes[i].annotate(f'{height}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=10, fontweight='semibold')
                        
        class_dist_data.append([trait, labels[0], counts[0], labels[1], counts[1]])
                        
    fig.suptitle('Dataset Class Distribution (Validation Support)', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(PLOTS_DIR, 'mbti_class_distribution.png'), dpi=150)
    plt.close()
    
    # Save raw CSV
    with open(os.path.join(PLOTS_DIR, 'mbti_class_distribution.csv'), mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['axis', 'class_0_label', 'class_0_count', 'class_1_label', 'class_1_count'])
        writer.writerows(class_dist_data)
        
    # ------------------ Plot 3: ROC Curves ------------------
    print("  Generating Plot & CSV 3: ROC Curves...")
    plt.figure(figsize=(10, 8))
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random Guess (AUC = 0.50)')
    
    # Prep CSV list
    roc_data = []
    
    for trait in traits:
        clf = models[trait]
        probs = clf.predict_proba(X_test_tfidf)[:, 1]
        fpr, tpr, thresholds = roc_curve(y_test[trait], probs)
        auc_score = roc_auc_score(y_test[trait], probs)
        plt.plot(fpr, tpr, color=colors[trait], label=f'{trait} (AUC = {auc_score:.4f})', linewidth=2)
        
        for fp, tp, th in zip(fpr, tpr, thresholds):
            roc_data.append([trait, fp, tp, th])
        
    plt.title('Receiver Operating Characteristic (ROC) Curves', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
    plt.ylabel('True Positive Rate (Sensitivity)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(fontsize=11, loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'mbti_roc_curves.png'), dpi=150)
    plt.close()
    
    # Save raw CSV
    with open(os.path.join(PLOTS_DIR, 'mbti_roc_curves.csv'), mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['axis', 'fpr', 'tpr', 'threshold'])
        writer.writerows(roc_data)
        
    # ------------------ Plot 4: Precision-Recall Curves ------------------
    print("  Generating Plot & CSV 4: Precision-Recall Curves...")
    plt.figure(figsize=(10, 8))
    
    # Prep CSV list
    pr_data = []
    
    for trait in traits:
        clf = models[trait]
        probs = clf.predict_proba(X_test_tfidf)[:, 1]
        precision, recall, thresholds = precision_recall_curve(y_test[trait], probs)
        pos_ratio = np.sum(y_test[trait] == 1) / len(y_test[trait])
        
        plt.plot(recall, precision, color=colors[trait], label=f'{trait} (Baseline = {pos_ratio:.2f})', linewidth=2)
        
        for idx in range(len(precision)):
            th = thresholds[idx] if idx < len(thresholds) else ''
            pr_data.append([trait, recall[idx], precision[idx], th])
        
    plt.title('Precision-Recall Curves (NLP Performance Metric)', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'mbti_precision_recall_curves.png'), dpi=150)
    plt.close()
    
    # Save raw CSV
    with open(os.path.join(PLOTS_DIR, 'mbti_precision_recall_curves.csv'), mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['axis', 'recall', 'precision', 'threshold'])
        writer.writerows(pr_data)
        
    # ------------------ Plot 5: Confusion Matrices ------------------
    print("  Generating Plot & CSV 5: Confusion Matrices...")
    fig, axes = plt.subplots(2, 2, figsize=(12, 11))
    axes = axes.ravel()
    
    # Prep CSV list
    cm_data = []
    
    for i, trait in enumerate(traits):
        clf = models[trait]
        preds = clf.predict(X_test_tfidf)
        cm = confusion_matrix(y_test[trait], preds, normalize='true')
        
        neg_label = trait_names[trait].split(' vs ')[1].split(' ')[0]
        pos_label = trait_names[trait].split(' vs ')[0].split(' ')[0]
        
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[neg_label, pos_label])
        disp.plot(ax=axes[i], cmap='Blues', values_format='.2f', colorbar=False)
        axes[i].set_title(f'Normalized Confusion Matrix: {trait}', fontsize=12, fontweight='bold')
        axes[i].grid(False)
        
        cm_data.append([trait, neg_label, neg_label, cm[0, 0]])
        cm_data.append([trait, neg_label, pos_label, cm[0, 1]])
        cm_data.append([trait, pos_label, neg_label, cm[1, 0]])
        cm_data.append([trait, pos_label, pos_label, cm[1, 1]])
        
    fig.suptitle('Normalized Confusion Matrices per Dimension', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(PLOTS_DIR, 'mbti_confusion_matrices.png'), dpi=150)
    plt.close()
    
    # Save raw CSV
    with open(os.path.join(PLOTS_DIR, 'mbti_confusion_matrices.csv'), mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['axis', 'true_label', 'predicted_label', 'normalized_percentage'])
        writer.writerows(cm_data)
        
    # ------------------ Plot 6: Feature Importance ------------------
    print("  Generating Plot & CSV 6: Feature Coefficient Importance...")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.ravel()
    
    feature_names = np.array(vectorizer.get_feature_names_out())
    
    # Prep CSV list
    feat_data = []
    
    for i, trait in enumerate(traits):
        clf = models[trait]
        coefs = clf.coef_[0]
        sorted_indices = np.argsort(coefs)
        
        neg_indices = sorted_indices[:10]
        pos_indices = sorted_indices[-10:]
        
        combined_indices = np.concatenate([neg_indices, pos_indices])
        combined_coefs = coefs[combined_indices]
        combined_words = feature_names[combined_indices]
        
        neg_label = trait_names[trait].split(' vs ')[1].split(' ')[0]
        pos_label = trait_names[trait].split(' vs ')[0].split(' ')[0]
        
        bar_colors = [colors['Class0']] * 10 + [colors['Class1']] * 10
        
        y_pos = np.arange(20)
        axes[i].barh(y_pos, combined_coefs, color=bar_colors, edgecolor='black', alpha=0.8)
        axes[i].set_yticks(y_pos)
        axes[i].set_yticklabels(combined_words, fontsize=10)
        axes[i].axvline(0, color='black', linewidth=1, linestyle='-')
        axes[i].set_title(f'Top Predictors for {trait} Axis\n(← {neg_label}  |  {pos_label} →)', fontsize=12, fontweight='bold')
        axes[i].grid(True, axis='x', linestyle='--', alpha=0.5)
        
        # Add details to feature CSV
        for idx in sorted_indices[:10]:
            feat_data.append([trait, feature_names[idx], coefs[idx], 'negative', neg_label])
        for idx in sorted_indices[-10:][::-1]:
            feat_data.append([trait, feature_names[idx], coefs[idx], 'positive', pos_label])
        
    fig.suptitle('Linguistic Feature Coefficients (Local Predictor Tokens)', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(PLOTS_DIR, 'mbti_feature_importance.png'), dpi=150)
    plt.close()
    
    # Save raw CSV
    with open(os.path.join(PLOTS_DIR, 'mbti_feature_importance.csv'), mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['axis', 'word', 'coefficient', 'direction', 'associated_class'])
        writer.writerows(feat_data)
        
    print(f"All graphs and raw CSV datasets generated successfully and saved to: '{PLOTS_DIR}/'")

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
    
    # Generate Analysis Plots
    generate_plots(X_train, X_test, y_train, y_test, vectorizer, models)
    
    print("Training pipeline finished successfully!")

if __name__ == "__main__":
    main()
