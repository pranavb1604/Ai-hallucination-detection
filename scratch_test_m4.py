import sys
sys.stdout.reconfigure(line_buffering=True, encoding='utf-8')
from modules.m4_entailment import score as m4_score
from modules.m2_grounding import fetch_best_context, _build_query

q = 'Who is the Prime Minister of India'
ans = 'As of my last update, the Prime Minister of India is Narendra Modi. He has been in office since May 2014.'

query = _build_query(q)
print('Query to Wikipedia:', repr(query))

retrieval = fetch_best_context(query)
print('Source:', retrieval['source'])
print('Context fetched.')
# Check if Narendra Modi is mentioned in the fetched context
context = retrieval['context']
if context:
    print('Does context mention Modi?:', 'Modi' in context)

r4 = m4_score(q, [ans])
print('\nM4 claims and scores:')
for claim in r4['m4_claim_scores']:
    print(f"- Claim: {claim['claim']}")
    print(f"  Score: {claim['entailment_prob']}")
print('Overall M4 score:', r4['m4_score'])
