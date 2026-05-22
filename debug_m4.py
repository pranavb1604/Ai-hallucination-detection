"""
Debug script to test M4 directly and show what's happening.
Run this to verify M4 is working outside of Streamlit.
"""

import sys
import os

# Clear any cached imports
if 'modules.m4_entailment' in sys.modules:
    del sys.modules['modules.m4_entailment']
if 'modules.m2_grounding' in sys.modules:
    del sys.modules['modules.m2_grounding']

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

print("=" * 70)
print("M4 ENTAILMENT DEBUG TEST")
print("=" * 70)

# Test the actual question from your screenshot
question = "What is the biggest planet in our solar system?"
responses = [
    "The biggest planet in our solar system is Jupiter.",
    "It's known for being the largest terrestrial body in the solar system and has the highest volume of all the planets.",
    "Jupiter is also a gas giant, along with Saturn, Uranus, and Neptune.",
    "Its diameter is about 139,820 kilometers (86,900 miles).",
    "However, if you're considering only terrestrial planets like Earth, Mars, Venus, and Mercury, then Earth is the biggest among them."
]

print(f"\nQuestion: {question}")
print(f"Number of responses: {len(responses)}")
print(f"\nFirst response: {responses[0][:100]}...")

# First get M2 evidence
print("\n" + "-" * 70)
print("STEP 1: Getting Wikipedia evidence (M2)...")
print("-" * 70)

from modules.m2_grounding import score as m2_score
m2_result = m2_score(question, responses)

print(f"✓ M2 Found: {m2_result['found']}")
print(f"✓ M2 Score: {m2_result['m2_score']}")
print(f"✓ M2 Source: {m2_result['source']}")
print(f"✓ Evidence length: {len(m2_result.get('context', ''))} characters")
if m2_result.get('context'):
    print(f"✓ Evidence preview: {m2_result['context'][:150]}...")

# Now test M4 with that evidence
print("\n" + "-" * 70)
print("STEP 2: Running M4 entailment with evidence...")
print("-" * 70)

from modules.m4_entailment import score as m4_score
m4_result = m4_score(question, responses, evidence=m2_result.get('context', ''))

print(f"\n✓ M4 Score: {m4_result['m4_score']}")
print(f"✓ M4 Verdict: {m4_result['m4_verdict']}")
print(f"✓ Number of claims: {len(m4_result.get('m4_claim_scores', []))}")

print("\n" + "-" * 70)
print("CLAIM-BY-CLAIM BREAKDOWN:")
print("-" * 70)

for i, claim_score in enumerate(m4_result.get('m4_claim_scores', []), 1):
    claim = claim_score['claim']
    prob = claim_score['entailment_prob']
    status = "✓" if prob > 0 else "✗"
    print(f"\n{status} Claim {i}: {claim}")
    print(f"   Entailment: {prob:.4f}")

print("\n" + "=" * 70)
if m4_result['m4_score'] > 0:
    print("✅ SUCCESS! M4 is working correctly!")
    print(f"   Overall M4 score: {m4_result['m4_score']}")
else:
    print("❌ PROBLEM! M4 is still returning 0.00")
    print("   This means the NLI model is not working properly.")
print("=" * 70)
