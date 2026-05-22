"""
Test script to verify M2 and M4 improvements.
Compares retrieval success before and after enhancements.
"""

import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from modules.m2_grounding import score as m2_score
from modules.m4_entailment import score as m4_score

# Test cases with known Wikipedia pages
TEST_CASES = [
    {
        "question": "Which American actor that founded Thunderant.com starred in Late Night?",
        "responses": ["Fred Armisen"],
        "expected_entity": "Fred Armisen"
    },
    {
        "question": "Who is the current president of the United States?",
        "responses": ["Joe Biden"],
        "expected_entity": "Joe Biden"
    },
    {
        "question": "Where is the Taj Mahal located?",
        "responses": ["The Taj Mahal is located in Agra, India"],
        "expected_entity": "Taj Mahal"
    },
    {
        "question": "What is the capital of France?",
        "responses": ["Paris"],
        "expected_entity": "Paris"
    },
    {
        "question": "Who wrote Harry Potter?",
        "responses": ["J.K. Rowling"],
        "expected_entity": "J.K. Rowling"
    },
    {
        "question": "What is the tallest mountain in the world?",
        "responses": ["Mount Everest"],
        "expected_entity": "Mount Everest"
    },
]


def test_m2_retrieval():
    """Test M2 Wikipedia retrieval success rate."""
    print("=" * 70)
    print("Testing M2 (Grounding) Module - Wikipedia Retrieval")
    print("=" * 70)
    
    success_count = 0
    total_count = len(TEST_CASES)
    
    for i, test in enumerate(TEST_CASES, 1):
        print(f"\n[Test {i}/{total_count}]")
        print(f"Question: {test['question']}")
        print(f"Expected entity: {test['expected_entity']}")
        
        result = m2_score(test['question'], test['responses'])
        
        found = result['found']
        score = result['m2_score']
        source = result['source']
        verdict = result['m2_verdict']
        
        print(f"✓ Found: {found}")
        print(f"✓ Score: {score}")
        print(f"✓ Source: {source}")
        print(f"✓ Verdict: {verdict}")
        
        if found:
            success_count += 1
            print("✓ SUCCESS: Evidence retrieved")
        else:
            print("✗ FAILED: No evidence found")
    
    print("\n" + "=" * 70)
    print(f"M2 Retrieval Success Rate: {success_count}/{total_count} ({100*success_count/total_count:.1f}%)")
    print("=" * 70)
    
    return success_count, total_count


def test_m4_entailment():
    """Test M4 entailment scoring with retrieved evidence."""
    print("\n" + "=" * 70)
    print("Testing M4 (Entailment) Module - Evidence-Based Scoring")
    print("=" * 70)
    
    success_count = 0
    total_count = len(TEST_CASES)
    
    for i, test in enumerate(TEST_CASES, 1):
        print(f"\n[Test {i}/{total_count}]")
        print(f"Question: {test['question']}")
        print(f"Answer: {test['responses'][0]}")
        
        # First get M2 evidence
        m2_result = m2_score(test['question'], test['responses'])
        evidence = m2_result.get('context', '')
        
        # Then run M4 with that evidence
        result = m4_score(test['question'], test['responses'], evidence=evidence)
        
        score = result['m4_score']
        verdict = result['m4_verdict']
        has_evidence = bool(result['m4_evidence_used'])
        
        print(f"✓ Evidence available: {has_evidence}")
        print(f"✓ Score: {score}")
        print(f"✓ Verdict: {verdict}")
        
        if has_evidence and score > 0:
            success_count += 1
            print("✓ SUCCESS: Entailment computed")
        else:
            print("✗ FAILED: No entailment score")
    
    print("\n" + "=" * 70)
    print(f"M4 Scoring Success Rate: {success_count}/{total_count} ({100*success_count/total_count:.1f}%)")
    print("=" * 70)
    
    return success_count, total_count


def test_entity_extraction():
    """Test entity extraction functionality."""
    print("\n" + "=" * 70)
    print("Testing Entity Extraction")
    print("=" * 70)
    
    try:
        from modules.m2_grounding import _extract_entities, _extract_key_phrases
        
        for i, test in enumerate(TEST_CASES, 1):
            print(f"\n[Test {i}] {test['question']}")
            
            entities = _extract_entities(test['question'])
            phrases = _extract_key_phrases(test['question'])
            
            print(f"  Entities: {entities}")
            print(f"  Key phrases: {phrases}")
            
            if test['expected_entity'] in entities or any(test['expected_entity'] in p for p in phrases):
                print("  ✓ Expected entity found")
            else:
                print(f"  ⚠ Expected entity '{test['expected_entity']}' not found")
        
        print("\n" + "=" * 70)
        
    except Exception as e:
        print(f"⚠ Entity extraction test skipped: {e}")
        print("Make sure spaCy is installed: python setup_spacy.py")


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("M2 & M4 IMPROVEMENT VERIFICATION TEST")
    print("=" * 70)
    
    # Test entity extraction
    test_entity_extraction()
    
    # Test M2 retrieval
    m2_success, m2_total = test_m2_retrieval()
    
    # Test M4 entailment
    m4_success, m4_total = test_m4_entailment()
    
    # Summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"M2 Retrieval: {m2_success}/{m2_total} ({100*m2_success/m2_total:.1f}%)")
    print(f"M4 Entailment: {m4_success}/{m4_total} ({100*m4_success/m4_total:.1f}%)")
    print("=" * 70)
    
    if m2_success >= m2_total * 0.8:
        print("✓ M2 improvements working well (≥80% success)")
    else:
        print("⚠ M2 needs further tuning (<80% success)")
    
    if m4_success >= m4_total * 0.8:
        print("✓ M4 improvements working well (≥80% success)")
    else:
        print("⚠ M4 needs further tuning (<80% success)")
    
    print("\nNote: Success rates depend on Wikipedia availability and network connection.")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
