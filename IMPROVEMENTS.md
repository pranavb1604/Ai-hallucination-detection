# M2 & M4 Wikipedia Extraction Improvements

## Problem Statement
The original M2 (Grounding) and M4 (Entailment) modules had low accuracy due to poor Wikipedia extraction. The main issues were:

1. **Simplistic query building** - Only stripped question prefixes, didn't extract actual entities
2. **No entity recognition** - Couldn't identify people, places, organizations in questions
3. **Single search attempt** - Failed if first query didn't work
4. **Poor disambiguation handling** - Couldn't handle Wikipedia disambiguation pages
5. **Limited context** - Only 20 sentences, often missing relevant information

## Solutions Implemented

### 1. Named Entity Recognition (NER)
- **Added spaCy integration** for intelligent entity extraction
- Extracts: PERSON, ORG, GPE, LOC, WORK_OF_ART, EVENT, PRODUCT, FAC, NORP
- Example: "Who is Fred Armisen?" → extracts "Fred Armisen" directly

### 2. Multi-Query Strategy
- **Query variants generation**: Creates 4 different query variations
  - Entity-based queries (primary entities from NER)
  - Noun phrase queries (key phrases from question)
  - Cleaned queries (original prefix-stripping logic)
  - Simple queries (just remove question words)
- Tries each variant until finding good results

### 3. Disambiguation Handling
- **Catches `DisambiguationError`** from Wikipedia API
- Automatically tries the first disambiguation option
- Prevents complete failure on ambiguous topics

### 4. Better Error Recovery
- **Multiple fallback strategies**:
  - Try multiple query variants
  - Handle PageError gracefully
  - Skip failed candidates, continue with others
  - Return best match from all successful retrievals

### 5. Increased Context
- **Increased from 20 to 30 sentences** for summaries
- **Full page content** (up to 3000 chars) instead of just summary
- Better coverage of relevant information

### 6. Semantic Ranking Enhancement
- Collects summaries from **multiple query variants**
- Ranks all candidates using **sentence transformers**
- Selects the most semantically relevant to the question

### 7. Evidence Sharing Between Modules
- **M2 passes retrieved context to M4** via pipeline
- Eliminates redundant Wikipedia API calls
- Ensures both modules use the same evidence

## Installation

### Step 1: Install dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Install spaCy language model
```bash
python setup_spacy.py
```

Or manually:
```bash
python -m spacy download en_core_web_sm
```

## Code Changes Summary

### M2 (modules/m2_grounding.py)
- Added spaCy NER integration
- Implemented `_extract_entities()` and `_extract_key_phrases()`
- Created `_build_query_variants()` for multi-query strategy
- Enhanced `fetch_best_context()` with disambiguation handling
- Increased context size to 3000 characters

### M4 (modules/m4_entailment.py)
- Added same NER improvements as M2
- Enhanced `_fetch_evidence()` with multi-query support
- Added disambiguation handling
- Modified `score()` to accept question parameter for better extraction

### Pipeline (pipeline.py)
- Modified to pass M2's context to M4
- Eliminates redundant Wikipedia calls

## Expected Improvements

### Before:
- **Query**: "Who is Fred Armisen?"
- **Wikipedia search**: "Fred Armisen?" (fails or gets wrong page)
- **Success rate**: ~40-50%

### After:
- **Query variants**: 
  1. "Fred Armisen" (entity extraction)
  2. "Fred Armisen" (noun phrase)
  3. "Fred Armisen" (cleaned)
  4. "fred armisen" (simple)
- **Success rate**: ~80-90%

## Testing the Improvements

Run a quick test:
```python
from modules.m2_grounding import score as m2_score
from modules.m4_entailment import score as m4_score

question = "Which American actor that founded Thunderant.com starred in Late Night?"
responses = ["Fred Armisen"]

# Test M2
m2_result = m2_score(question, responses)
print(f"M2 Score: {m2_result['m2_score']}")
print(f"Found: {m2_result['found']}")
print(f"Source: {m2_result['source']}")

# Test M4
m4_result = m4_score(question, responses, evidence=m2_result['context'])
print(f"M4 Score: {m4_result['m4_score']}")
print(f"Verdict: {m4_result['m4_verdict']}")
```

## Additional Recommendations

### 1. Use Alternative Knowledge Sources
If Wikipedia still fails for certain domains:
- **Wikidata API** - More structured data
- **DBpedia** - Semantic knowledge base
- **Google Knowledge Graph API** - Broader coverage
- **Custom knowledge bases** - Domain-specific sources

### 2. Implement Caching
```python
from functools import lru_cache

@lru_cache(maxsize=1000)
def fetch_best_context_cached(query: str, question: str):
    return fetch_best_context(query, question)
```

### 3. Add Query Expansion
Use WordNet or word embeddings to expand queries:
```python
# Example: "actor" → ["actor", "actress", "performer", "artist"]
```

### 4. Use Wikipedia's Full Text Search
Instead of just `wikipedia.search()`, use the MediaWiki API directly:
```python
import requests

def advanced_search(query):
    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "format": "json",
        "srlimit": 10
    }
    response = requests.get(url, params=params)
    return response.json()
```

### 5. Implement Relevance Scoring
Score each retrieved document by:
- Entity overlap with question
- Keyword match
- Semantic similarity
- Recency (for time-sensitive questions)

## Troubleshooting

### Issue: spaCy model not found
**Solution**: Run `python -m spacy download en_core_web_sm`

### Issue: Still getting "No evidence found"
**Possible causes**:
1. Very obscure topics not in Wikipedia
2. Misspelled entities in questions
3. Questions requiring multi-hop reasoning

**Solutions**:
- Add more query variants
- Use fuzzy matching for entity names
- Implement multi-hop retrieval

### Issue: Wrong Wikipedia page retrieved
**Solution**: 
- Increase MAX_CANDIDATES to 10
- Improve semantic ranking with better embeddings
- Add entity type filtering (e.g., only search for PERSON entities for "who" questions)

## Performance Metrics to Track

Monitor these metrics to measure improvement:
1. **Retrieval success rate**: % of questions where evidence is found
2. **Retrieval accuracy**: % of questions where correct page is retrieved
3. **M2 score distribution**: Should shift higher
4. **M4 score distribution**: Should shift higher
5. **End-to-end accuracy**: Overall hallucination detection accuracy

## Next Steps

1. **Install spaCy model**: `python setup_spacy.py`
2. **Test on your dataset**: Run evaluation on test.csv
3. **Compare metrics**: Before vs after improvements
4. **Fine-tune thresholds**: Adjust similarity thresholds based on results
5. **Consider additional sources**: If Wikipedia coverage is still insufficient
