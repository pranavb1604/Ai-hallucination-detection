"""
test_scoring.py
───────────────
Quick test script to verify scoring improvements for correct answers.
Tests both correct answers (should score 80+) and hallucinations (should score low).

Usage:
    python test_scoring.py
"""

import sys
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from pipeline import run_pipeline


def print_result(question: str, responses: list[str], expected: str):
    """Run pipeline and print formatted results."""
    print("\n" + "=" * 70)
    print(f"Question: {question}")
    print(f"Answer:   {responses[0]}")
    print(f"Expected: {expected}")
    print("-" * 70)
    
    result = run_pipeline(question, responses)
    
    trust_score = result['trust_score']
    trust_label = result['trust_label']
    
    # Color coding for terminal (optional)
    if trust_score >= 0.80:
        status = "✓ PASS" if expected == "CORRECT" else "✗ FAIL"
    elif trust_score < 0.50:
        status = "✓ PASS" if expected == "HALLUCINATION" else "✗ FAIL"
    else:
        status = "~ UNCERTAIN"
    
    print(f"\nTrust Score: {trust_score:.4f} ({trust_score*100:.1f}/100)")
    print(f"Trust Label: {trust_label}")
    print(f"Status:      {status}")
    
    print(f"\nModule Breakdown:")
    print(f"  M1 (Consistency): {result['m1']['m1_score']:.4f} - {result['m1']['m1_verdict']}")
    print(f"  M2 (Grounding):   {result['m2']['m2_score']:.4f} - {result['m2']['m2_verdict']}")
    print(f"  M3 (Uncertainty): {result['m3']['m3_score']:.4f} - {result['m3']['m3_verdict']}")
    print(f"  M4 (Entailment):  {result['m4']['m4_score']:.4f} - {result['m4']['m4_verdict']}")
    
    print(f"\nScorer Used: {result['scorer_used']}")
    print("=" * 70)
    
    return trust_score


def main():
    print("\n" + "█" * 70)
    print("  HALLUCINATION DETECTION - SCORING TEST")
    print("█" * 70)
    print("\nTesting scoring improvements...")
    print("Target: Correct answers should score 80+ (0.80)")
    print("        Hallucinations should score <50 (0.50)")
    
    # ─── Test 1: Simple factual question (CORRECT) ───────────────────────────
    score1 = print_result(
        question="What is the capital of France?",
        responses=["The capital of France is Paris."],
        expected="CORRECT"
    )
    
    # ─── Test 2: Scientific fact (CORRECT) ───────────────────────────────────
    score2 = print_result(
        question="What is the largest planet in our solar system?",
        responses=["Jupiter is the largest planet in our solar system."],
        expected="CORRECT"
    )
    
    # ─── Test 3: Historical fact (CORRECT) ───────────────────────────────────
    score3 = print_result(
        question="Who was the first president of the United States?",
        responses=["George Washington was the first president of the United States."],
        expected="CORRECT"
    )
    
    # ─── Test 4: Obvious hallucination (WRONG) ───────────────────────────────
    score4 = print_result(
        question="What is the capital of France?",
        responses=["The capital of France is Berlin."],
        expected="HALLUCINATION"
    )
    
    # ─── Test 5: Subtle hallucination (WRONG) ────────────────────────────────
    score5 = print_result(
        question="What is the largest planet in our solar system?",
        responses=["Saturn is the largest planet in our solar system."],
        expected="HALLUCINATION"
    )
    
    # ─── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "█" * 70)
    print("  TEST SUMMARY")
    print("█" * 70)
    
    correct_scores = [score1, score2, score3]
    halluc_scores = [score4, score5]
    
    avg_correct = sum(correct_scores) / len(correct_scores)
    avg_halluc = sum(halluc_scores) / len(halluc_scores)
    
    print(f"\nCorrect Answers:")
    print(f"  Average Score: {avg_correct:.4f} ({avg_correct*100:.1f}/100)")
    print(f"  Individual:    {', '.join(f'{s:.2f}' for s in correct_scores)}")
    print(f"  Target:        ≥ 0.80 (80/100)")
    
    if avg_correct >= 0.80:
        print(f"  Result:        ✓ PASS - Correct answers score well!")
    else:
        print(f"  Result:        ✗ FAIL - Need further tuning")
        print(f"  Gap:           {(0.80 - avg_correct)*100:.1f} points below target")
    
    print(f"\nHallucinations:")
    print(f"  Average Score: {avg_halluc:.4f} ({avg_halluc*100:.1f}/100)")
    print(f"  Individual:    {', '.join(f'{s:.2f}' for s in halluc_scores)}")
    print(f"  Target:        < 0.50 (50/100)")
    
    if avg_halluc < 0.50:
        print(f"  Result:        ✓ PASS - Hallucinations detected!")
    else:
        print(f"  Result:        ✗ FAIL - Too lenient on hallucinations")
        print(f"  Gap:           {(avg_halluc - 0.50)*100:.1f} points above target")
    
    print("\n" + "█" * 70)
    
    # Overall assessment
    if avg_correct >= 0.80 and avg_halluc < 0.50:
        print("  ✓✓✓ ALL TESTS PASSED - System is well-calibrated!")
    elif avg_correct >= 0.80:
        print("  ⚠ PARTIAL PASS - Good for correct answers, but check hallucination detection")
    elif avg_halluc < 0.50:
        print("  ⚠ PARTIAL PASS - Good hallucination detection, but correct answers score too low")
    else:
        print("  ✗ NEEDS TUNING - See SCORING_IMPROVEMENTS.md for adjustment guidelines")
    
    print("█" * 70 + "\n")


if __name__ == "__main__":
    main()
