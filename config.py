import os

# ─────────────────────────────────────────────
# BASE PATHS
# ─────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR           = os.path.join(BASE_DIR, "data")
RAW_DATA_DIR       = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")

HALUEVAL_RAW_PATH  = os.path.join(RAW_DATA_DIR, "halueval", "halueval_qa.csv")
TRUTHFULQA_RAW_PATH= os.path.join(RAW_DATA_DIR, "truthfulqa", "truthfulqa.csv")

TRAIN_PATH         = os.path.join(PROCESSED_DATA_DIR, "train.csv")
TEST_PATH          = os.path.join(PROCESSED_DATA_DIR, "test.csv")

RESULTS_DIR        = os.path.join(BASE_DIR, "results")
SCORES_DIR         = os.path.join(RESULTS_DIR, "scores")

# ─────────────────────────────────────────────
# DATASET SETTINGS
# ─────────────────────────────────────────────

HALUEVAL_DATASET_NAME   = "pminervini/HaluEval"
HALUEVAL_SUBSET         = "qa_samples"

TRUTHFULQA_DATASET_NAME = "truthful_qa"
TRUTHFULQA_SUBSET       = "multiple_choice"

TEST_SIZE   = 0.2    # 80% train, 20% test
RANDOM_SEED = 42

# ─────────────────────────────────────────────
# MODULE 1 — SEMANTIC CONSISTENCY
# ─────────────────────────────────────────────

M1_MODEL_NAME           = "all-MiniLM-L6-v2"  # sentence transformer — fast & good
M1_NUM_SAMPLES          = 5
M1_M3_MIN_SAMPLES       = 3
M1_SIMILARITY_THRESHOLD = 0.7

# ─────────────────────────────────────────────
# MODULE 2 — RETRIEVAL GROUNDING
# ─────────────────────────────────────────────

M2_MODEL_NAME           = "all-MiniLM-L6-v2"
M2_WIKI_LANGUAGE        = "en"
M2_WIKI_USER_AGENT      = "HallucinationDetector/1.0 (btech.project@dtu.ac.in)"
M2_EVIDENCE_LENGTH      = 800   # reduced from 1000 — enough context, faster NLI
M2_GROUNDING_THRESHOLD  = 0.5

# ─────────────────────────────────────────────
# MODULE 4 — NLI ENTAILMENT
# ─────────────────────────────────────────────

# CHANGED: cross-encoder/nli-MiniLM2-L6-H768 is ~80MB vs 1.6GB for BART-large-mnli
# Accuracy is nearly identical on short factual claims; 5–8x faster on CPU.
# Other fast options (in order of speed):
#   "typeform/distilbert-base-uncased-mnli"   — ~260MB, good
#   "cross-encoder/nli-deberta-v3-small"      — ~180MB, best accuracy/speed tradeoff
#   "cross-encoder/nli-MiniLM2-L6-H768"      — ~80MB,  fastest
M4_MODEL_NAME     = "cross-encoder/nli-MiniLM2-L6-H768"

M4_EVIDENCE_LENGTH  = 600   # increased — more evidence = better NLI accuracy
M4_MIN_CLAIM_LENGTH = 10
M4_MAX_CHUNKS       = 4     # more chunks = better evidence coverage for entailment

# ─────────────────────────────────────────────
# MODULE 5 — TRUST CLASSIFIER
# ─────────────────────────────────────────────

M5_INPUT_SIZE    = 21   # matches FEATURE_DIM in m5_features.py
M5_HIDDEN_SIZE_1 = 64
M5_HIDDEN_SIZE_2 = 32
M5_OUTPUT_SIZE   = 1
M5_LEARNING_RATE = 0.001
M5_EPOCHS        = 100
M5_BATCH_SIZE    = 64    # increased from 32 — fewer steps per epoch if NN is used

# CHANGED: "gb" instead of "auto"
# GradientBoosting always wins on 4-feature tabular data.
# "auto" wastes 5–15 min training a NN just to discard it.
# Options: "gb" (recommended) | "nn" (legacy) | "auto" (trains both, picks winner)
M5_BACKEND       = "gb"
SCORING_WORKERS  = 6     # ThreadPoolExecutor workers for parallel scoring

MODEL_SAVE_PATH  = os.path.join(BASE_DIR, "models", "trust_classifier.pth")
M5_BUNDLE_PATH   = os.path.join(BASE_DIR, "models", "m5_bundle.pkl")
SCALER_SAVE_PATH = os.path.join(BASE_DIR, "models", "scaler_m5.pkl")

# ─────────────────────────────────────────────
# TRUST SCORE THRESHOLDS
# ─────────────────────────────────────────────

TRUST_THRESHOLDS = {
    "trusted"   : 0.65,
    "uncertain" : 0.40,
    "suspicious": 0.20,
}

TRUST_LABELS = {
    "trusted"      : "Trusted",
    "uncertain"    : "Uncertain",
    "suspicious"   : "Suspicious",
    "hallucinated" : "Hallucinated",
}

# ─────────────────────────────────────────────
# SCORING CACHE  (NEW)
# ─────────────────────────────────────────────

# Wikipedia results are cached to disk so repeated questions don't re-fetch.
# Delete this file with --fresh to force a clean re-score.
WIKI_CACHE_PATH = os.path.join(PROCESSED_DATA_DIR, "wiki_cache.pkl")

# ─────────────────────────────────────────────
# AUTO CREATE DIRECTORIES
# ─────────────────────────────────────────────

for _dir in [
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    SCORES_DIR,
    os.path.join(BASE_DIR, "models"),
    os.path.join(RAW_DATA_DIR, "halueval"),
    os.path.join(RAW_DATA_DIR, "truthfulqa"),
]:
    os.makedirs(_dir, exist_ok=True)