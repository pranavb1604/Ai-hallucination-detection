"""
data/eda_visualize.py
Generates graphs for the midterm report based on the processed dataset.

Run from project root:
    python data/eda_visualize.py
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from config import TRAIN_PATH, TEST_PATH

RESULTS_DIR = os.path.join(BASE_DIR, "results", "figures")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Style ─────────────────────────────────────────
sns.set_theme(style="whitegrid", font_scale=1.1)
COLORS      = ['#2ecc71', '#e74c3c']
PALETTE     = {0: '#2ecc71', 1: '#e74c3c'}
SOURCE_PAL  = {'halueval': '#3498db', 'truthfulqa': '#9b59b6'}


def _save(fig, name: str):
    path = os.path.join(RESULTS_DIR, name)
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  → Saved: {path}")


# ──────────────────────────────────────────────
# Plot 1
# ──────────────────────────────────────────────
def plot_label_distribution(df):
    fig, ax = plt.subplots(figsize=(7, 5))
    counts = df['label'].value_counts().sort_index()

    bars = ax.bar(
        ['Correct (0)', 'Hallucinated (1)'],
        counts.values,
        color=COLORS, edgecolor='white', linewidth=0.8, width=0.5
    )

    for bar in bars:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f'{int(bar.get_height()):,}',
            ha='center', va='bottom'
        )

    fig.tight_layout()
    _save(fig, 'label_distribution.png')


# ──────────────────────────────────────────────
# Plot 2
# ──────────────────────────────────────────────
def plot_source_distribution(df):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    src_label = df.groupby(['source', 'label']).size().unstack(fill_value=0)
    src_label.columns = ['Correct', 'Hallucinated']

    src_label.plot(kind='bar', ax=axes[0], color=COLORS)

    src_counts = df['source'].value_counts()
    axes[1].pie(src_counts.values, labels=src_counts.index, autopct='%1.1f%%')

    fig.tight_layout()
    _save(fig, 'source_distribution.png')


# ──────────────────────────────────────────────
# Plot 3 (FIXED)
# ──────────────────────────────────────────────
def plot_answer_lengths(df):
    df = df.copy()

    # ✅ FIX: ensure correct dtype
    df['label'] = df['label'].astype(int)

    df['answer_length'] = df['answer'].astype(str).apply(lambda x: len(x.split()))
    cap = df['answer_length'].quantile(0.95)
    df_capped = df[df['answer_length'] <= cap]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # ✅ FIX: added hue
    sns.boxplot(
        data=df_capped,
        x='label',
        y='answer_length',
        hue='label',
        palette=PALETTE,
        ax=axes[0],
        width=0.4,
        legend=False
    )

    # histogram
    for lbl, color in PALETTE.items():
        subset = df_capped[df_capped['label'] == lbl]['answer_length']
        axes[1].hist(subset, bins=30, alpha=0.5, color=color)

    fig.tight_layout()
    _save(fig, 'answer_lengths.png')


# ──────────────────────────────────────────────
# Plot 4
# ──────────────────────────────────────────────
def plot_train_test_split(train_df, test_df):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, split_df in zip(axes, [train_df, test_df]):
        counts = split_df['label'].value_counts().sort_index()

        ax.bar(['Correct', 'Hallucinated'], counts.values, color=COLORS)

    fig.tight_layout()
    _save(fig, 'train_test_split.png')


# ──────────────────────────────────────────────
# Plot 5
# ──────────────────────────────────────────────
def plot_question_lengths(df):
    df = df.copy()
    df['label'] = df['label'].astype(int)

    df['q_length'] = df['question'].astype(str).apply(lambda x: len(x.split()))

    fig, ax = plt.subplots()

    for lbl, color in PALETTE.items():
        subset = df[df['label'] == lbl]['q_length']
        ax.hist(subset, bins=25, alpha=0.6, color=color)

    fig.tight_layout()
    _save(fig, 'question_lengths.png')


# ──────────────────────────────────────────────
# MAIN (FIXED)
# ──────────────────────────────────────────────
def main():
    train_df = pd.read_csv(TRAIN_PATH)
    test_df  = pd.read_csv(TEST_PATH)

    combined = pd.concat([train_df, test_df], ignore_index=True)

    # ✅ GLOBAL FIX
    train_df['label'] = train_df['label'].astype(int)
    test_df['label'] = test_df['label'].astype(int)
    combined['label'] = combined['label'].astype(int)

    plot_label_distribution(combined)
    plot_source_distribution(combined)
    plot_answer_lengths(combined)
    plot_train_test_split(train_df, test_df)
    plot_question_lengths(combined)


if __name__ == "__main__":
    main()