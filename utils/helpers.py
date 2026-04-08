def clean_text(text):
    """
    Clean the input text before processing.

    - Removes leading/trailing spaces
    - Converts everything to lowercase
    This helps make comparisons consistent.
    """
    return text.strip().lower()


def normalize_score(score):
    """
    Normalize a score to ensure it lies between 0 and 1.

    Why?
    - Similarity values or calculations might go slightly out of range
    - We clamp them to [0, 1] for stability

    If conversion fails (invalid input), return 0.0 as safe fallback.
    """
    try:
        score = float(score)          # convert to float
        return max(0.0, min(1.0, score))  # clamp between 0 and 1
    except:
        return 0.0  # fallback if score is invalid