"""
data/eda_visualize.py
Generates graphs for the midterm report based on the processed dataset.
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from config import TRAIN_PATH

# Ensure the results folder exists
RESULTS_DIR = os.path.join(BASE_DIR, "results", "figures")
os.makedirs(RESULTS_DIR, exist_ok=True)

def generate_graphs():
    print(f"Loading data from: {TRAIN_PATH}")
    try:
        df = pd.read_csv(TRAIN_PATH)
    except FileNotFoundError:
        print("Error: train.csv not found! Run preprocess.py first.")
        return

    print(f"Dataset loaded: {len(df)} rows.")

    # Set professional visual style
    sns.set_theme(style="whitegrid")
    colors = ['#2ecc71', '#e74c3c'] # Green for Correct, Red for Hallucinated

    # ──────────────────────────────────────────────
    # Plot 1: Overall Distribution
    # ──────────────────────────────────────────────
    plt.figure(figsize=(8, 6))
    ax = sns.countplot(data=df, x='label', palette=colors)
    plt.title('Overall Dataset Balance: Correct vs Hallucinated', fontsize=14, pad=15)
    plt.xticks([0, 1], ['Correct (0)', 'Hallucinated (1)'])
    plt.ylabel('Number of Samples')
    plt.xlabel('')
    
    # Add number labels on top of the bars
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='bottom', fontsize=12, color='black', xytext=(0, 5),
                    textcoords='offset points')
    
    plt.tight_layout()
    dist_path = os.path.join(RESULTS_DIR, 'label_distribution.png')
    plt.savefig(dist_path, dpi=300)
    print(f"  -> Saved: {dist_path}")

    # ──────────────────────────────────────────────
    # Plot 2: Distribution by Source
    # ──────────────────────────────────────────────
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x='source', hue='label', palette=colors)
    plt.title('Label Distribution by Source Dataset', fontsize=14, pad=15)
    plt.legend(['Correct (0)', 'Hallucinated (1)'], title='Label')
    plt.ylabel('Number of Samples')
    plt.xlabel('Dataset Source')
    
    plt.tight_layout()
    source_path = os.path.join(RESULTS_DIR, 'source_distribution.png')
    plt.savefig(source_path, dpi=300)
    print(f"  -> Saved: {source_path}")

    # ──────────────────────────────────────────────
    # Plot 3: Answer Length Analysis
    # ──────────────────────────────────────────────
    # Calculate how many words are in each answer
    df['answer_length'] = df['answer'].astype(str).apply(lambda x: len(x.split()))
    
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x='label', y='answer_length', palette=colors)
    plt.title('Answer Length: Correct vs Hallucinated', fontsize=14, pad=15)
    plt.xticks([0, 1], ['Correct (0)', 'Hallucinated (1)'])
    plt.ylabel('Word Count')
    plt.xlabel('')
    
    # Cap the Y-axis to ignore extreme outliers so the boxplot is readable
    y_max = df['answer_length'].quantile(0.95)
    plt.ylim(0, y_max)
    
    plt.tight_layout()
    len_path = os.path.join(RESULTS_DIR, 'answer_lengths.png')
    plt.savefig(len_path, dpi=300)
    print(f"  -> Saved: {len_path}")

    print("\n✅ All visualizations generated successfully!")

if __name__ == "__main__":
    generate_graphs()