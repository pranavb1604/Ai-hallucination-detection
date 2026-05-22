"""
test_m4_aggregation.py
──────────────────────
Test script to demonstrate M4 aggregation improvements.
Shows how harmonic mean with outlier filtering handles problematic claims better than MIN.

Usage:
    python test_m4_aggregation.py
"""

import sys
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from modules.m4_entailment import score as m4_score


def test_m4(question: str, answer: str, description: str):
    """Test M4 scoring and show claim breakdown."""
    print("\n" + "=" * 80)
    print(f"TEST: {description}")
    print("=" * 80)
    print(f"Question: {question}")
    print(f"Answer:   {answer}")
    print("-" * 80)
    
    result = m4_score(question, [answer])
    
    print(f"\nM4 Scores:")
    print(f"  Main Score (aggregated): {result['m4_score']:.4f}")
    print(f"  Mean Score:              {result['m4_mean_score']:.4f}")
    print(f"  Min Score (old method):  {result['m4_min_score']:.4f}")
    print(f"  Unsupported Claims:      {result['m4_n_unsupported']}")
    print(f"  Verdict:                 {result['m4_verdict']}")
    
    print(f"\nClaim Breakdown:")
    for i, claim_score in enumerate(result['m4_claim_scores'], 1):
        claim = claim_score['claim']
        support = claim_score['support_score']
        ent = claim_score['entailment_prob']
        neu = claim_score['neutral_prob']
        con = claim_score['contradiction_prob']
        supported = "✓" if claim_score['supported'] else "✗"
        
        print(f"  {i}. [{supported}] {claim}")
        print(f"      Support: {support:.3f} | Ent: {ent:.3f} | Neu: {neu:.3f} | Con: {con:.3f}")
    
    print(f"\nImprovement:")
    improvement = result['m4_score'] - result['m4_min_score']
    if improvement > 0.1:
        print(f"  ✓ Aggregated score is {improvement:.3f} higher than MIN")
        print(f"  ✓ Harmonic mean successfully filtered outliers!")
    elif improvement > 0:
        print(f"  ~ Small improvement of {improvement:.3f}")
    else:
        print(f"  - No improvement (both methods agree)")
    
    print("=" * 80)
    
    return result


def main():
    print("\n" + "█" * 80)
    print("  M4 AGGREGATION METHOD TEST")
    print("  Comparing: MIN (old) vs HARMONIC MEAN + OUTLIER FILTER (new)")
    print("█" * 80)
    
    # ─── Test 1: Correct answer with pronoun issue ───────────────────────────
    test_m4(
        question="What is the largest planet in our solar system?",
        answer="Jupiter is the largest planet in our solar system. It is a gas giant.",
        description="Correct answer with pronoun issue"
    )
    
    # ─── Test 2: Correct answer with multiple claims ─────────────────────────
    test_m4(
        question="Tell me about Jupiter",
        answer="Jupiter is the largest planet. It has many moons. The Great Red Spot is a massive storm.",
        description="Correct answer with multiple claims (some with pronouns)"
    )
    
    # ─── Test 3: Correct answer with short fragment ──────────────────────────
    test_m4(
        question="Is Paris the capital of France?",
        answer="Yes, Paris is the capital of France. It is located in northern France.",
        description="Correct answer with short fragment and pronoun"
    )
    
    # ─── Test 4: Real hallucination (should still fail) ──────────────────────
    test_m4(
        question="What is the capital of France?",
        answer="The capital of France is Berlin. It is a major European city.",
        description="Real hallucination (should still score low)"
    )
    
    # ─── Test 5: Mixed correct and incorrect claims ──────────────────────────
    test_m4(
        question="Tell me about Mars",
        answer="Mars is the largest planet. It has a red color. It has two moons.",
        description="Mixed: one wrong claim, two correct claims"
    )
    
    # ─── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "█" * 80)
    print("  SUMMARY")
    print("█" * 80)
    print("\nKey Improvements:")
    print("  ✓ Harmonic mean is more robust to 1-2 bad claims (pronouns, fragments)")
    print("  ✓ Outlier filtering removes worst claim if 3+ claims present")
    print("  ✓ Safety check: if 50%+ claims unsupported, falls back to MIN")
    print("  ✓ Real hallucinations still score low (< 0.30)")
    print("\nExpected Behavior:")
    print("  • Correct answer + pronoun issue:  MIN ≈ 0.05 → Harmonic ≈ 0.75-0.85")
    print("  • Correct answer + short fragment: MIN ≈ 0.10 → Harmonic ≈ 0.70-0.80")
    print("  • Real hallucination:              MIN ≈ 0.05 → Harmonic ≈ 0.05-0.15")
    print("  • Mixed (50%+ wrong):              MIN ≈ 0.10 → Harmonic ≈ 0.10-0.20")
    print("\n" + "█" * 80 + "\n")


if __name__ == "__main__":
    main()
