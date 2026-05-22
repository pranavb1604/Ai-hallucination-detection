"""
Quick test to verify M4 entailment scoring is working.
"""

import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

print("Testing M4 Entailment Module...")
print("=" * 60)

# Test case 1: Simple factual statement
question1 = "What is the biggest planet in our solar system?"
responses1 = ["Jupiter is the biggest planet in our solar system."]
evidence1 = "Jupiter is the largest planet in the Solar System. It is a gas giant with a mass more than two and a half times that of all the other planets in the Solar System combined."

print("\n[Test 1] Simple factual statement")
print(f"Question: {question1}")
print(f"Response: {responses1[0]}")
print(f"Evidence: {evidence1[:100]}...")

from modules.m4_entailment import score as m4_score

result1 = m4_score(question1, responses1, evidence=evidence1)
print(f"\n✓ M4 Score: {result1['m4_score']}")
print(f"✓ Verdict: {result1['m4_verdict']}")
print(f"✓ Claims:")
for claim in result1['m4_claim_scores']:
    print(f"  - '{claim['claim']}' → {claim['entailment_prob']}")

# Test case 2: Multiple claims
question2 = "Tell me about Jupiter"
responses2 = ["Jupiter is the biggest planet. It's known for being the largest terrestrial body. Jupiter is also a gas giant."]
evidence2 = "Jupiter is the largest planet in the Solar System. It is a gas giant with a mass more than two and a half times that of all the other planets combined."

print("\n" + "=" * 60)
print("\n[Test 2] Multiple claims")
print(f"Question: {question2}")
print(f"Response: {responses2[0]}")
print(f"Evidence: {evidence2[:100]}...")

result2 = m4_score(question2, responses2, evidence=evidence2)
print(f"\n✓ M4 Score: {result2['m4_score']}")
print(f"✓ Verdict: {result2['m4_verdict']}")
print(f"✓ Claims:")
for claim in result2['m4_claim_scores']:
    print(f"  - '{claim['claim']}' → {claim['entailment_prob']}")

print("\n" + "=" * 60)
print("\n✅ M4 module test complete!")
print("\nIf you see non-zero entailment scores above, M4 is working correctly.")
print("If all scores are still 0.00, there may be an issue with the NLI model.")
