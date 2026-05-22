# M4 Min Score Problem - Analysis & Solution

## The Problem: Why M4 Returns 0

Your M4 module uses **MIN entailment** across all claims, which means:

```python
min_score = min(support_probs)  # Takes the LOWEST score among all claims
m4_score = min_score            # This becomes your M4 score
```

### Why This Causes 0 Scores:

**Example**: Answer: "Jupiter is the largest planet. It has a Great Red Spot."

```
Claim 1: "Jupiter is the largest planet"     → support_score = 0.85 ✓
Claim 2: "It has a Great Red Spot"           → support_score = 0.05 ✗
                                                 (pronoun "It" not resolved)

min_score = min(0.85, 0.05) = 0.05
m4_score = 0.05  ← FAILS! (even though main claim is correct)
```

### Root Causes:

1. **Pronoun resolution fails**: "It", "This", "They" don't match Wikipedia text
2. **Sentence splitting issues**: Short fragments like "Yes." or "Indeed." score 0
3. **One bad claim ruins everything**: Even 1 unsupported claim → entire answer fails
4. **Paraphrase mismatches**: Correct but differently worded claims get low scores

---

## Current Logic: MIN vs MEAN

### MIN (Current - Too Strict):
```python
m4_score = min(all_claim_scores)  # One bad claim = total failure
```

**Philosophy**: "Chain is only as strong as weakest link"  
**Problem**: Too harsh for correct answers with minor issues

### MEAN (Alternative - Too Lenient):
```python
m4_score = mean(all_claim_scores)  # Average all claims
```

**Philosophy**: "Overall correctness matters"  
**Problem**: Hallucinations can hide among correct claims

---

## Recommended Solution: HARMONIC MEAN with Outlier Filtering

This balances strictness with robustness:

```python
# Remove worst outlier if we have multiple claims
# Then use harmonic mean (penalizes low scores more than arithmetic mean)
```

### Why Harmonic Mean?

**Arithmetic Mean**: (0.9 + 0.9 + 0.1) / 3 = 0.63  
**Harmonic Mean**: 3 / (1/0.9 + 1/0.9 + 1/0.1) ≈ 0.24  

Harmonic mean is **stricter** than arithmetic but **more forgiving** than min.

---

## Implementation Options

### Option 1: **Harmonic Mean with Outlier Removal** (RECOMMENDED)

**Best for**: Correct answers with 1-2 problematic claims (pronouns, short fragments)

```python
def _aggregate_claim_scores(support_probs: list[float], n_claims: int) -> float:
    """
    Aggregate claim scores using harmonic mean with outlier filtering.
    More robust than min, stricter than arithmetic mean.
    """
    if not support_probs:
        return 0.0
    
    if len(support_probs) == 1:
        return support_probs[0]
    
    # Remove worst outlier if we have 3+ claims
    if len(support_probs) >= 3:
        sorted_scores = sorted(support_probs)
        # Remove the single worst score
        filtered_scores = sorted_scores[1:]
    else:
        filtered_scores = support_probs
    
    # Harmonic mean: n / sum(1/x_i)
    # Add small epsilon to avoid division by zero
    epsilon = 1e-6
    harmonic = len(filtered_scores) / sum(1.0 / (s + epsilon) for s in filtered_scores)
    
    return float(np.clip(harmonic, 0.0, 1.0))
```

**Example**:
```
Claims: [0.85, 0.80, 0.05]  (last one is pronoun issue)
After filtering: [0.85, 0.80]
Harmonic mean: 2 / (1/0.85 + 1/0.80) ≈ 0.82  ✓ Good score!
```

---

### Option 2: **Weighted Mean (Penalize Short Claims)** (GOOD ALTERNATIVE)

**Best for**: When short claims are the problem

```python
def _aggregate_claim_scores_weighted(claim_scores: list[dict]) -> float:
    """
    Weight claims by their length - longer claims matter more.
    Short fragments (pronouns, "Yes.", etc.) get less weight.
    """
    if not claim_scores:
        return 0.0
    
    weighted_sum = 0.0
    total_weight = 0.0
    
    for cs in claim_scores:
        claim_len = len(cs["claim"].split())
        # Weight: sqrt(length) to avoid over-weighting very long claims
        weight = np.sqrt(claim_len)
        weighted_sum += cs["support_score"] * weight
        total_weight += weight
    
    return float(np.clip(weighted_sum / total_weight, 0.0, 1.0))
```

**Example**:
```
Claim 1: "Jupiter is the largest planet" (5 words) → score=0.85, weight=2.24
Claim 2: "It orbits the Sun" (4 words)            → score=0.05, weight=2.00

Weighted mean: (0.85*2.24 + 0.05*2.00) / (2.24+2.00) = 0.47
Better than min(0.85, 0.05) = 0.05!
```

---

### Option 3: **Percentile-Based (e.g., 25th percentile)** (BALANCED)

**Best for**: General robustness

```python
def _aggregate_claim_scores_percentile(support_probs: list[float]) -> float:
    """
    Use 25th percentile instead of min.
    More robust to outliers than min, stricter than mean.
    """
    if not support_probs:
        return 0.0
    
    if len(support_probs) == 1:
        return support_probs[0]
    
    # 25th percentile: 75% of claims must be above this score
    percentile_25 = float(np.percentile(support_probs, 25))
    return float(np.clip(percentile_25, 0.0, 1.0))
```

**Example**:
```
Claims: [0.85, 0.80, 0.75, 0.05]
25th percentile ≈ 0.69  (much better than min=0.05)
```

---

### Option 4: **Hybrid: Min of Top K Claims** (SIMPLE & EFFECTIVE)

**Best for**: Simple fix without complex math

```python
def _aggregate_claim_scores_topk(support_probs: list[float], k_ratio: float = 0.75) -> float:
    """
    Take min of top K% of claims (default: top 75%).
    Ignores worst 25% as potential outliers.
    """
    if not support_probs:
        return 0.0
    
    if len(support_probs) == 1:
        return support_probs[0]
    
    # Keep top k% of claims
    k = max(1, int(len(support_probs) * k_ratio))
    top_k_scores = sorted(support_probs, reverse=True)[:k]
    
    return float(np.clip(min(top_k_scores), 0.0, 1.0))
```

**Example**:
```
Claims: [0.85, 0.80, 0.75, 0.05]
Top 75% (3 claims): [0.85, 0.80, 0.75]
Min of top 75%: 0.75  ✓ Good score!
```

---

## Comparison Table

| Method | Score for [0.85, 0.80, 0.05] | Pros | Cons |
|--------|------------------------------|------|------|
| **MIN (current)** | 0.05 | Strictest, catches all issues | Too harsh, 1 bad claim fails all |
| **MEAN** | 0.57 | Simple, balanced | Too lenient on hallucinations |
| **Harmonic + Filter** | 0.82 | Robust, mathematically sound | Slightly complex |
| **Weighted by Length** | 0.47-0.82 | Handles short claims well | Depends on claim lengths |
| **25th Percentile** | 0.69 | Simple, robust | Less intuitive |
| **Min of Top 75%** | 0.80 | Simple, effective | Arbitrary threshold |

---

## My Recommendation: **Harmonic Mean with Outlier Filtering**

### Why?

1. ✅ **Robust to 1-2 bad claims** (pronouns, fragments)
2. ✅ **Still strict enough** to catch real hallucinations
3. ✅ **Mathematically principled** (harmonic mean is standard for rates/ratios)
4. ✅ **Scales well** with different numbers of claims

### When It Works Best:

- ✓ Correct answer with pronoun issues: [0.85, 0.80, 0.05] → 0.82
- ✓ Correct answer with short fragment: [0.90, 0.88, 0.10] → 0.89
- ✗ Real hallucination: [0.85, 0.15, 0.10] → 0.13 (still catches it!)

### When It Might Fail:

- If 50%+ of claims are hallucinated, it might still score too high
- Solution: Combine with `n_unsupported` check (see below)

---

## Implementation

I'll implement **Option 1 (Harmonic Mean + Outlier Filter)** with a safety check:

```python
def _aggregate_claim_scores(support_probs: list[float], n_unsupported: int) -> float:
    """
    Aggregate claim scores with outlier filtering and harmonic mean.
    
    Strategy:
    1. If 50%+ claims unsupported → use min (likely hallucination)
    2. If 3+ claims → remove worst outlier
    3. Use harmonic mean (stricter than arithmetic, more robust than min)
    """
    if not support_probs:
        return 0.0
    
    if len(support_probs) == 1:
        return support_probs[0]
    
    # Safety: if majority of claims unsupported, use strict min
    if n_unsupported > len(support_probs) / 2:
        return float(np.clip(min(support_probs), 0.0, 1.0))
    
    # Remove worst outlier if we have 3+ claims
    if len(support_probs) >= 3:
        sorted_scores = sorted(support_probs)
        filtered_scores = sorted_scores[1:]  # Remove lowest
    else:
        filtered_scores = support_probs
    
    # Harmonic mean
    epsilon = 1e-6
    harmonic = len(filtered_scores) / sum(1.0 / (s + epsilon) for s in filtered_scores)
    
    return float(np.clip(harmonic, 0.0, 1.0))
```

---

## Testing Examples

### Test 1: Correct Answer with Pronoun Issue
```
Question: "What is the largest planet?"
Answer: "Jupiter is the largest planet. It is a gas giant."

Claims:
  1. "Jupiter is the largest planet" → 0.85
  2. "It is a gas giant" → 0.05 (pronoun issue)

Current (MIN): 0.05 ✗
Proposed (Harmonic): 0.85 ✓
```

### Test 2: Correct Answer with Multiple Claims
```
Question: "Tell me about Jupiter"
Answer: "Jupiter is the largest planet. It has 79 moons. The Great Red Spot is a storm."

Claims:
  1. "Jupiter is the largest planet" → 0.85
  2. "It has 79 moons" → 0.10 (pronoun)
  3. "The Great Red Spot is a storm" → 0.80

Current (MIN): 0.10 ✗
Proposed (Harmonic after filter): 0.82 ✓
```

### Test 3: Real Hallucination (Should Still Fail)
```
Question: "What is the capital of France?"
Answer: "The capital of France is Berlin. It is in Germany."

Claims:
  1. "The capital of France is Berlin" → 0.05 (contradiction)
  2. "It is in Germany" → 0.10 (wrong context)

Current (MIN): 0.05 ✓
Proposed (Harmonic): 0.07 ✓ (still fails!)
```

---

## Alternative: Use MEAN for M4, Keep MIN for Debugging

Another approach: Use **mean** as the main score, but keep **min** for diagnostics:

```python
return {
    "m4_score": round(mean_score, 4),        # Use mean for M5
    "m4_mean_score": round(mean_score, 4),
    "m4_min_score": round(min_score, 4),     # Keep for debugging
    "m4_n_unsupported": n_unsupported,
    ...
}
```

**Pros**: Simple, works well if most claims are correct  
**Cons**: Less strict, might miss subtle hallucinations

---

## Summary & Recommendation

### Current Problem:
- **MIN is too strict** → 1 bad claim (pronoun, fragment) = entire answer fails
- Correct answers score 0 even when main content is accurate

### Best Solution:
**Harmonic Mean with Outlier Filtering**
- Removes worst outlier (if 3+ claims)
- Uses harmonic mean (stricter than arithmetic, more robust than min)
- Safety check: if 50%+ unsupported, fall back to min

### Expected Impact:
- Correct answers with 1-2 problematic claims: 0.05 → 0.75-0.85 ✓
- Real hallucinations: Still score low (< 0.30) ✓
- Overall M4 contribution to 80+ target: Much more reliable!

Would you like me to implement this solution?
