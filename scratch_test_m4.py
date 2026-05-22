"""Quick check: Wikipedia retrieval + M4 entailment."""
from modules.m2_grounding import fetch_evidence_for_qa
from modules.m4_entailment import score as m4_score

question = "Who is the prime minister of India?"
answer = "Narendra Modi is the prime minister of India."

retrieval = fetch_evidence_for_qa(question, answer)
print("Wiki:", retrieval.get("title"), "| rel=", retrieval.get("relevance"))

m4 = m4_score(question, [answer])
print("M4 score:", m4["m4_score"], "|", m4["m4_verdict"])
