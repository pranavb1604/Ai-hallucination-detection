"""
precompute_features.py
──────────────────────
Runs the M2 (Grounding) and M4 (Entailment) modules to extract 8 factual features.
Balances the dataset with equal numbers of correct and hallucinated rows.

Speed optimisation: Wikipedia evidence is pre-fetched in parallel using a
ThreadPoolExecutor so the GPU is never idle waiting for network I/O.
"""
import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from config import TRAIN_PATH, PROCESSED_DATA_DIR
from modules.m2_grounding import score as m2_score, fetch_evidence_for_qa
from modules.m4_entailment import score as m4_score

CACHE_PATH = os.path.join(PROCESSED_DATA_DIR, "cached_m2_m4_features.csv")

# Number of Wikipedia requests to fire in parallel.
# 16 is a sweet spot: enough to saturate the network without rate-limiting.
PREFETCH_WORKERS = 16


def _prefetch_evidence(rows):
    """
    Pre-fetch Wikipedia evidence for a list of (idx, question, answer, label) tuples
    using a thread pool so all network requests happen concurrently.
    Returns a dict: question -> evidence_dict
    """
    evidence_map = {}

    def fetch(item):
        _, q, a, _ = item
        return q, fetch_evidence_for_qa(q, a)

    with ThreadPoolExecutor(max_workers=PREFETCH_WORKERS) as pool:
        futures = {pool.submit(fetch, item): item for item in rows}
        for future in as_completed(futures):
            try:
                q, ev = future.result()
                evidence_map[q] = ev
            except Exception:
                _, q, _, _ = futures[future]
                evidence_map[q] = {"context": "", "source": "", "found": False,
                                   "relevance": 0.0, "title": ""}
    return evidence_map


def _compute_row(idx, q, a, lbl, evidence):
    """Run M2 grounding + M4 entailment for a single row given pre-fetched evidence."""
    context = evidence.get("context", "")

    # M2 grounding score (re-use pre-fetched evidence)
    from modules.m2_grounding import compute_grounding
    grounding = compute_grounding(a, context, question=q)
    if not evidence.get("found"):
        m2_verdict = "No evidence found"
    elif grounding >= 0.75:
        m2_verdict = "Well-supported by evidence"
    elif grounding >= 0.50:
        m2_verdict = "Partially supported"
    elif grounding >= 0.30:
        m2_verdict = "Weakly supported"
    else:
        m2_verdict = "Not supported — possible hallucination"

    m2_res = {
        "m2_score": round(grounding, 4),
        "m2_verdict": m2_verdict,
        "context": context,
        "source": evidence.get("source", ""),
        "found": evidence.get("found", False),
        "wiki_title": evidence.get("title", ""),
        "wiki_relevance": evidence.get("relevance", 0.0),
    }

    # M4 entailment score
    m4_res = m4_score(q, [a], evidence=context)

    claim_scores = m4_res.get("m4_claim_scores", [])
    if claim_scores:
        avg_nli      = float(np.mean([c["nli_support"]    for c in claim_scores]))
        avg_semantic = float(np.mean([c["semantic_score"] for c in claim_scores]))
        avg_lexical  = float(np.mean([c["lexical_score"]  for c in claim_scores]))
    else:
        avg_nli = avg_semantic = avg_lexical = 0.0

    return {
        "idx":             idx,
        "question":        q,
        "label":           lbl,
        "m2_score":        m2_res["m2_score"],
        "m4_score":        m4_res["m4_score"],
        "m4_mean_score":   m4_res.get("m4_mean_score", 0.0),
        "m4_min_score":    m4_res.get("m4_min_score", 0.0),
        "m4_n_unsupported":float(m4_res.get("m4_n_unsupported", 0)),
        "m4_avg_nli":      avg_nli,
        "m4_avg_semantic": avg_semantic,
        "m4_avg_lexical":  avg_lexical,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=800,
                        help="Total balanced rows to precompute")
    parser.add_argument("--all", action="store_true", default=False,
                        help="Process ALL rows in train.csv (imbalanced)")
    parser.add_argument("--workers", type=int, default=PREFETCH_WORKERS,
                        help="Number of parallel Wikipedia fetch threads (default: 16)")
    args = parser.parse_args()

    df = pd.read_csv(TRAIN_PATH)

    if args.all:
        balanced_df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        n_correct     = len(df[df['label'] == 0])
        n_hallucinated = len(df[df['label'] == 1])
        print(f"Processing ALL rows: {n_correct} correct (label=0) and "
              f"{n_hallucinated} hallucinated (label=1).")
    else:
        df_correct     = df[df['label'] == 0]
        df_hallucinated = df[df['label'] == 1]
        max_half  = min(len(df_correct), len(df_hallucinated))
        half_size = min(args.rows // 2, max_half)
        df_c_sampled = df_correct.sample(n=half_size, random_state=42)
        df_h_sampled = df_hallucinated.sample(n=half_size, random_state=42)
        balanced_df  = pd.concat([df_c_sampled, df_h_sampled]) \
                         .sample(frac=1, random_state=42).reset_index(drop=True)
        print(f"Dataset balanced: {half_size} correct (label=0) and "
              f"{half_size} hallucinated (label=1) rows.")

    print(f"Total rows to process: {len(balanced_df)}")
    print(f"Parallel Wikipedia workers: {args.workers}")

    # Delete old cache if format is wrong
    if os.path.exists(CACHE_PATH):
        try:
            old_df = pd.read_csv(CACHE_PATH)
            if "m4_min_score" not in old_df.columns:
                print("Deleting old cache file (wrong format)...")
                os.remove(CACHE_PATH)
        except Exception:
            os.remove(CACHE_PATH)

    # Load existing cache to resume
    if os.path.exists(CACHE_PATH):
        cached_df = pd.read_csv(CACHE_PATH)
        processed_questions = set(cached_df['question'].tolist())
        results = cached_df.to_dict('records')
        print(f"Resuming from {len(processed_questions)} cached rows...")
    else:
        processed_questions = set()
        results = []

    # Build list of rows that still need processing
    pending = [
        (idx, str(row["question"]).strip(), str(row["answer"]).strip(), int(row["label"]))
        for idx, row in balanced_df.iterrows()
        if str(row["question"]).strip() not in processed_questions
    ]
    print(f"Rows left to process: {len(pending)}")

    BATCH = args.workers * 2   # pre-fetch 2× the worker count at a time

    try:
        with tqdm(total=len(pending), desc="Precomputing features") as pbar:
            for batch_start in range(0, len(pending), BATCH):
                batch = pending[batch_start: batch_start + BATCH]

                # ── Step 1: pre-fetch Wikipedia in parallel ────────────────
                evidence_map = _prefetch_evidence(batch)

                # ── Step 2: run AI inference sequentially (GPU stays busy) ─
                for idx, q, a, lbl in batch:
                    ev = evidence_map.get(q, {})
                    try:
                        record = _compute_row(idx, q, a, lbl, ev)
                        results.append(record)
                    except Exception as e:
                        print(f"\n[WARN] Row {idx} skipped: {e}")

                    pbar.update(1)

                    # Save every 10 new results
                    if len(results) % 10 == 0:
                        pd.DataFrame(results).to_csv(CACHE_PATH, index=False)

    except KeyboardInterrupt:
        print("\nProcess interrupted. Saving progress...")
    except Exception as e:
        print(f"\nError: {e}. Saving progress...")

    # Final save
    if results:
        pd.DataFrame(results).to_csv(CACHE_PATH, index=False)
        print(f"\nDone! Features saved to {CACHE_PATH}")
    else:
        print("No new features to save.")


if __name__ == "__main__":
    main()
