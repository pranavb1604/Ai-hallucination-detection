import os

# ─────────────────────────────────────────────
# BASE PATHS
# ─────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR              = os.path.join(BASE_DIR, "data")
RAW_DATA_DIR          = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR    = os.path.join(DATA_DIR, "processed")

HALUEVAL_RAW_PATH     = os.path.join(RAW_DATA_DIR, "halueval", "halueval_qa.csv")
TRUTHFULQA_RAW_PATH   = os.path.join(RAW_DATA_DIR, "truthfulqa", "truthfulqa.csv")

TRAIN_PATH            = os.path.join(PROCESSED_DATA_DIR, "train.csv")
TEST_PATH             = os.path.join(PROCESSED_DATA_DIR, "test.csv")

RESULTS_DIR           = os.path.join(BASE_DIR, "results")
SCORES_DIR            = os.path.join(RESULTS_DIR, "scores")

# ─────────────────────────────────────────────
# DATASET SETTINGS
# ─────────────────────────────────────────────

HALUEVAL_DATASET_NAME    = "pminervini/HaluEval"
HALUEVAL_SUBSET          = "qa_samples"

TRUTHFULQA_DATASET_NAME  = "truthful_qa"
TRUTHFULQA_SUBSET        = "multiple_choice"

TEST_SIZE                = 0.2       # 80% train, 20% test
RANDOM_SEED              = 42

# ─────────────────────────────────────────────
# MODULE 1 — SEMANTIC CONSISTENCY
# ─────────────────────────────────────────────

M1_MODEL_NAME        = "all-MiniLM-L6-v2"   # sentence transformer model
M1_NUM_SAMPLES       = 5                     # how many LLM responses to compare
M1_SIMILARITY_THRESHOLD = 0.7               # below this = inconsistent

# ─────────────────────────────────────────────
# MODULE 2 — RETRIEVAL GROUNDING
# ─────────────────────────────────────────────

M2_MODEL_NAME        = "all-MiniLM-L6-v2"   # same sentence transformer
M2_WIKI_LANGUAGE     = "en"
M2_WIKI_USER_AGENT   = "HallucinationDetector/1.0 (btech.project@dtu.ac.in)"
M2_EVIDENCE_LENGTH   = 1000                  # characters to take from Wikipedia
M2_GROUNDING_THRESHOLD = 0.5                # below this = not grounded

# ─────────────────────────────────────────────
# MODULE 4 — NLI ENTAILMENT
# ─────────────────────────────────────────────

M4_MODEL_NAME        =  "facebook/bart-large-mnli"
M4_EVIDENCE_LENGTH   = 500
M4_MIN_CLAIM_LENGTH  = 10                    # ignore sentences shorter than this

# ─────────────────────────────────────────────
# MODULE 5 — TRUST CLASSIFIER
# ─────────────────────────────────────────────

M5_INPUT_SIZE        = 10                    # M1, M2, M3, M4 scores
M5_HIDDEN_SIZE_1     = 64
M5_HIDDEN_SIZE_2     = 32
M5_OUTPUT_SIZE       = 1
M5_LEARNING_RATE     = 0.001
M5_EPOCHS            = 100
M5_BATCH_SIZE        = 32

MODEL_SAVE_PATH      = os.path.join(BASE_DIR, "models", "trust_classifier.pth")
SCALER_SAVE_PATH     = os.path.join(BASE_DIR, "models", "scaler_m5.pkl")

# ─────────────────────────────────────────────
# TRUST SCORE THRESHOLDS
# ─────────────────────────────────────────────

TRUST_THRESHOLDS = {
    "trusted"      : 0.75,
    "uncertain"    : 0.50,
    "suspicious"   : 0.25,
}

TRUST_LABELS = {
    "trusted"      : "Trusted",
    "uncertain"    : "Uncertain",
    "suspicious"   : "Suspicious",
    "hallucinated" : "Hallucinated",
}

# ─────────────────────────────────────────────
# AUTO CREATE DIRECTORIES IF THEY DONT EXIST
# ─────────────────────────────────────────────

for _dir in [RAW_DATA_DIR, PROCESSED_DATA_DIR, SCORES_DIR,
             os.path.join(BASE_DIR, "models"),
             os.path.join(RAW_DATA_DIR, "halueval"),
             os.path.join(RAW_DATA_DIR, "truthfulqa")]:
    os.makedirs(_dir, exist_ok=True)