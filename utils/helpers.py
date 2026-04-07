# - get_wikipedia_evidence(query) → fetches Wiki passage for M2 & M4
# - split_into_claims(text) → sentence tokenizer for M4
# - normalize_score(val) → clamps any float to [0.0, 1.0]
# - save_scores(scores_dict, path) → saves JSON to results/scores/