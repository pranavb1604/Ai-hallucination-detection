

import os
import sys
import requests
import numpy as np
import streamlit as st
from dotenv import load_dotenv

# ── project root on path ──────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

load_dotenv(os.path.join(BASE_DIR, ".env"))

from pipeline import run_pipeline
from config import MODEL_SAVE_PATH, TRUST_THRESHOLDS

# ── Ollama config ─────────────────────────────────────────────────────────────
OLLAMA_URL   = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral")

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Trust Evaluator",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .trust-badge {
        display: inline-block;
        padding: 6px 20px;
        border-radius: 20px;
        font-size: 15px;
        font-weight: 600;
        letter-spacing: 0.3px;
    }
    .badge-trusted      { background: #d1fae5; color: #065f46; }
    .badge-uncertain    { background: #fef3c7; color: #78350f; }
    .badge-suspicious   { background: #ffedd5; color: #7c2d12; }
    .badge-hallucinated { background: #fee2e2; color: #7f1d1d; }

    .score-ring {
        font-size: 56px;
        font-weight: 700;
        text-align: center;
        line-height: 1;
    }
    .module-card {
        background: #f9fafb;
        border: 1px solid #e5e7eb;
        border-radius: 10px;
        padding: 14px 18px;
        margin-bottom: 10px;
    }
    .pair-row {
        font-size: 13px;
        padding: 4px 0;
        border-bottom: 1px solid #f0f0f0;
    }
    .evidence-box {
        background: #f0fdf4;
        border-left: 4px solid #10b981;
        border-radius: 6px;
        padding: 12px 16px;
        font-size: 14px;
        color: #065f46;
    }
    .no-evidence-box {
        background: #fef9c3;
        border-left: 4px solid #f59e0b;
        border-radius: 6px;
        padding: 12px 16px;
        font-size: 14px;
        color: #78350f;
    }
    .claim-row {
        font-size: 13px;
        padding: 5px 0;
        border-bottom: 1px solid #f0f0f0;
    }
    .wiki-answer-box {
        background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%);
        border-left: 5px solid #3b82f6;
        border-radius: 8px;
        padding: 16px 20px;
        margin: 12px 0;
        font-size: 15px;
        color: #1e3a5f;
    }
    .wiki-answer-box .answer-label {
        font-size: 12px;
        font-weight: 600;
        color: #3b82f6;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 6px;
    }
    .wiki-answer-box .answer-text {
        font-size: 22px;
        font-weight: 700;
        color: #1e40af;
        margin: 4px 0 8px;
    }
    .wiki-answer-box .answer-confidence {
        font-size: 12px;
        color: #6b7280;
    }
    .wiki-answer-box .answer-confidence .conf-badge {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 10px;
        font-weight: 600;
        font-size: 11px;
    }
    .conf-high   { background: #d1fae5; color: #065f46; }
    .conf-medium { background: #fef3c7; color: #78350f; }
    .conf-low    { background: #fee2e2; color: #7f1d1d; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def check_ollama():
    """Returns (is_running: bool, model_list: list)."""
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=2)
        models = [m["name"] for m in r.json().get("models", [])]
        return True, models
    except Exception:
        return False, []


def generate_ollama_responses(question: str, n: int, model: str, temp: float):
    """Ask Ollama the same question n times and return responses."""
    samples = []
    for i in range(n):
        try:
            r = requests.post(
                f"{OLLAMA_URL}/api/generate",
                json={
                    "model": model,
                    "prompt": question,
                    "stream": False,
                    "options": {
                        "temperature": temp,
                        "num_predict": 200,
                    },
                },
                timeout=60,
            )
            text = r.json().get("response", "").strip()
            if text:
                samples.append(text)
        except Exception as e:
            st.error(f"Ollama error on sample {i + 1}: {e}")
    return samples


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")

    model_exists = os.path.exists(MODEL_SAVE_PATH)
    if model_exists:
        st.success("✅ Trained M5 model loaded")
    else:
        st.warning("⚠️ No trained model — using weighted fallback")

    st.divider()
    st.subheader("Trust thresholds")
    thresh_trusted    = st.slider("Trusted ≥",    0.0, 1.0,
                                  TRUST_THRESHOLDS["trusted"],    0.05)
    thresh_uncertain  = st.slider("Uncertain ≥",  0.0, 1.0,
                                  TRUST_THRESHOLDS["uncertain"],  0.05)
    thresh_suspicious = st.slider("Suspicious ≥", 0.0, 1.0,
                                  TRUST_THRESHOLDS["suspicious"], 0.05)

    st.divider()
    st.subheader("🤖 LLM Settings (Ollama)")

    ollama_ok, ollama_models = check_ollama()

    if ollama_ok:
        st.success("🟢 Ollama connected")
        model_options  = ollama_models if ollama_models else [OLLAMA_MODEL]
        selected_model = st.selectbox("Model", model_options)
    else:
        st.error("🔴 Ollama offline")
        st.caption("Run `ollama serve` in a terminal, then refresh.")
        selected_model = OLLAMA_MODEL

    num_samples = st.slider("Samples to generate (M1)", 3, 7, 3)
    temperature  = st.slider("Temperature", 0.1, 1.0, 0.7, 0.05)

    st.divider()
    st.subheader("Demo examples")
    demo_topic = st.selectbox(
        "Pre-fill example",
        ["(none)", "Eiffel Tower", "Black holes",
         "Aspirin mechanism", "Hallucinated example"],
    )

    _demos = {
        "Eiffel Tower": {
            "q": "Where is the Eiffel Tower located?",
            "r": [
                "The Eiffel Tower is located in Paris, France, on the Champ de Mars.",
                "It stands in Paris, near the Seine river.",
                "The Eiffel Tower is in Paris, France.",
            ],
        },
        "Black holes": {
            "q": "What is a black hole?",
            "r": [
                "A black hole is a region of spacetime where gravity is so strong that nothing, not even light, can escape.",
                "Black holes are formed when massive stars collapse and create an extremely dense point called a singularity.",
                "A black hole is a cosmic object with such intense gravity that light cannot escape its event horizon.",
            ],
        },
        "Aspirin mechanism": {
            "q": "How does aspirin reduce fever?",
            "r": [
                "Aspirin inhibits COX enzymes, reducing prostaglandin synthesis, which lowers fever and inflammation.",
                "It works by blocking cyclooxygenase enzymes in the prostaglandin pathway.",
                "Aspirin reduces fever by preventing prostaglandin production via COX inhibition.",
            ],
        },
        "Hallucinated example": {
            "q": "Who invented the telephone?",
            "r": [
                "The telephone was invented by Nikola Tesla in 1892 in New York.",
                "Thomas Edison created the first telephone in his Menlo Park laboratory in 1878.",
                "The telephone was invented by Albert Einstein as part of his work on electromagnetic waves.",
            ],
        },
    }

    st.divider()
    st.caption("AI Hallucination Detection · DTU IT 401")


# ─────────────────────────────────────────────────────────────────────────────
# Main area
# ─────────────────────────────────────────────────────────────────────────────
st.title("🔍 AI Hallucination Detection & Trust Calibration")
st.caption(
    "Multi-signal system: M1 Consistency · M2 Grounding · "
    "M3 Uncertainty · M4 Entailment · M5 Neural Classifier"
)

# ─── Pre-fill from demo ───────────────────────────────────────────────────────
prefill_q, prefill_r = "", ""
if demo_topic != "(none)" and demo_topic in _demos:
    d = _demos[demo_topic]
    prefill_q = d["q"]
    prefill_r = "\n".join(d["r"])

# ─── Input section ────────────────────────────────────────────────────────────
st.subheader("Input")

mode = st.radio(
    "Input mode",
    ["🤖 Auto — Ask Ollama", "✏️ Manual — Paste responses"],
    horizontal=True,
)

question = st.text_area(
    "Question asked to the LLM",
    value=prefill_q,
    height=90,
    placeholder="e.g. Where is the Eiffel Tower located?",
)

responses = []

# ── Manual mode ───────────────────────────────────────────────────────────────
if mode == "✏️ Manual — Paste responses":
    raw_responses = st.text_area(
        "LLM responses — one per line (≥ 3 for best results)",
        value=prefill_r,
        height=110,
        placeholder="Response 1\nResponse 2\nResponse 3",
    )
    responses = [r.strip() for r in raw_responses.strip().splitlines() if r.strip()]

# ── Auto Ollama mode ──────────────────────────────────────────────────────────
else:
    st.caption(
        f"Ollama will generate **{num_samples}** responses using `{selected_model}` "
        f"at temperature **{temperature}**"
    )

    # Session state to persist generated responses across reruns
    if "ollama_responses" not in st.session_state:
        st.session_state.ollama_responses = []
    if "ollama_question" not in st.session_state:
        st.session_state.ollama_question = ""

    col_gen, col_clear = st.columns([2, 1])
    with col_gen:
        generate = st.button(
            "⚡ Generate responses from Ollama",
            disabled=(not ollama_ok or not question.strip()),
            use_container_width=True,
        )
    with col_clear:
        if st.button("🗑 Clear", use_container_width=True):
            st.session_state.ollama_responses = []
            st.session_state.ollama_question  = ""
            st.rerun()

    if generate and question.strip():
        with st.spinner(f"Asking {selected_model} {num_samples} times…"):
            st.session_state.ollama_responses = generate_ollama_responses(
                question, num_samples, selected_model, temperature
            )
            st.session_state.ollama_question = question

    if st.session_state.ollama_responses:
        st.success(f"✅ {len(st.session_state.ollama_responses)} responses generated")
        st.markdown("**Generated responses:**")
        for i, r in enumerate(st.session_state.ollama_responses, 1):
            st.info(f"**{i}.** {r}")
        responses = st.session_state.ollama_responses
    else:
        if not ollama_ok:
            st.info("Start Ollama first: `ollama serve` → then refresh the page.")
        elif question.strip():
            st.info("Click ⚡ Generate to get responses from Ollama.")

# ── Evaluate button ───────────────────────────────────────────────────────────
col_btn, col_info = st.columns([1, 3])
with col_btn:
    run = st.button("▶ Evaluate", type="primary", use_container_width=True)
with col_info:
    if responses:
        st.caption(f"✅ {len(responses)} response(s) ready to evaluate")
    else:
        st.caption("Generate or paste responses first, then click Evaluate")

st.divider()


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────
if run:
    if not question.strip():
        st.warning("Please enter a question.")
        st.stop()
    if not responses:
        st.warning("Please generate or paste at least one LLM response.")
        st.stop()

    # ── Run M1–M4 ─────────────────────────────────────────────────────────────
    progress = st.progress(0, text="Running M1 — Semantic Consistency …")

    with st.spinner("Running M1 — Semantic Consistency …"):
        import importlib
        import modules.m1_consistency
        importlib.reload(modules.m1_consistency)
        from modules.m1_consistency import score as _m1
        m1 = _m1(question, responses)
    progress.progress(25, text="Running M2 — Retrieval Grounding …")

    with st.spinner("Running M2 — Retrieval Grounding …"):
        import importlib
        import modules.m2_grounding
        importlib.reload(modules.m2_grounding)
        from modules.m2_grounding import score as _m2
        m2 = _m2(question, responses)
    progress.progress(50, text="Running M3 — Uncertainty Estimation …")

    with st.spinner("Running M3 — Uncertainty Estimation …"):
        import importlib
        import modules.m3_uncertainty
        importlib.reload(modules.m3_uncertainty)
        from modules.m3_uncertainty import score as _m3
        m3 = _m3(question, responses)
    progress.progress(75, text="Running M4 — NLI Entailment …")

    with st.spinner("Running M4 — NLI Entailment …"):
        import importlib
        import modules.m4_entailment
        importlib.reload(modules.m4_entailment)
        from modules.m4_entailment import score as _m4
        m4 = _m4(question, responses)
    progress.progress(90, text="Fusing scores via M5 …")

    # ── Fuse via M5 ───────────────────────────────────────────────────────────
    from modules.m5_classifier import load_model, predict_trust, weighted_trust_score
    from config import M5_INPUT_SIZE, M5_HIDDEN_SIZE_1, M5_HIDDEN_SIZE_2

    s1, s2, s3, s4 = (m1["m1_score"], m2["m2_score"],
                      m3["m3_score"], m4["m4_score"])

    model_exists = os.path.exists(MODEL_SAVE_PATH)
    if model_exists:
        nn_model    = load_model(MODEL_SAVE_PATH, M5_INPUT_SIZE,
                                 M5_HIDDEN_SIZE_1, M5_HIDDEN_SIZE_2)
        result      = predict_trust(nn_model, s1, s2, s3, s4)
        trust_score = result["trust_score"]
        scorer_used = "Neural M5 classifier"
    else:
        trust_score = weighted_trust_score(s1, s2, s3, s4)
        scorer_used = "Weighted fallback (no trained model)"

    progress.progress(100, text="Done.")
    progress.empty()

    # ── Trust label ───────────────────────────────────────────────────────────
    if trust_score >= thresh_trusted:
        badge_class  = "badge-trusted"
        verdict_text = "✅ Trusted"
    elif trust_score >= thresh_uncertain:
        badge_class  = "badge-uncertain"
        verdict_text = "⚠️ Uncertain"
    elif trust_score >= thresh_suspicious:
        badge_class  = "badge-suspicious"
        verdict_text = "🟠 Suspicious"
    else:
        badge_class  = "badge-hallucinated"
        verdict_text = "🚨 Hallucinated"

    ring_color = (
        "#10b981" if trust_score >= thresh_trusted else
        "#f59e0b" if trust_score >= thresh_uncertain else
        "#f97316" if trust_score >= thresh_suspicious else
        "#ef4444"
    )

    # ─────────────────────────────────────────────────────────────────────────
    # Results Layout
    # ─────────────────────────────────────────────────────────────────────────
    left, right = st.columns([3, 2], gap="large")

    # ── LEFT COLUMN ───────────────────────────────────────────────────────────
    with left:
        st.subheader("Primary answer")
        st.info(responses[0])

        # Show which LLM generated this (if auto mode)
        if mode == "🤖 Auto — Ask Ollama":
            st.caption(f"Generated by: `{selected_model}` via Ollama")

        with st.expander(f"All {len(responses)} sampled responses"):
            for idx, r in enumerate(responses, 1):
                st.markdown(f"**{idx}.** {r}")

        # M1 pairwise breakdown
        with st.expander("M1 — Pairwise similarity breakdown"):
            pairs = m1.get("m1_pairs", [])
            if pairs:
                for p in pairs:
                    sim   = p["similarity"]
                    color = ("#10b981" if sim >= 0.7 else
                             "#f59e0b" if sim >= 0.45 else "#ef4444")
                    bar_w = int(sim * 100)
                    st.markdown(
                        f"""<div class="pair-row">
                            Response {p['i']+1} ↔ Response {p['j']+1} &nbsp;
                            <span style="color:{color};font-weight:600">{sim:.2f}</span>
                            <div style="background:#e5e7eb;border-radius:4px;height:6px;margin-top:4px">
                              <div style="width:{bar_w}%;background:{color};height:6px;border-radius:4px"></div>
                            </div>
                        </div>""",
                        unsafe_allow_html=True,
                    )
            else:
                st.caption("Need ≥ 2 responses for pairwise comparison.")
            st.markdown("")
            st.caption(
                f"Min similarity: **{m1['m1_min']:.2f}** &nbsp;|&nbsp; "
                f"Std dev: **{m1['m1_std']:.3f}**"
            )

        # M4 claim-level breakdown
        with st.expander("M4 — Claim-level entailment breakdown"):
            claim_scores = m4.get("m4_claim_scores", [])
            if claim_scores:
                for cs in claim_scores:
                    prob  = cs["entailment_prob"]
                    color = ("#10b981" if prob >= 0.6 else
                             "#f59e0b" if prob >= 0.35 else "#ef4444")
                    bar_w = int(prob * 100)
                    st.markdown(
                        f"""<div class="claim-row">
                            <em>{cs['claim']}</em><br>
                            Entailment: <span style="color:{color};font-weight:600">{prob:.2f}</span>
                            <div style="background:#e5e7eb;border-radius:4px;height:5px;margin-top:3px">
                              <div style="width:{bar_w}%;background:{color};height:5px;border-radius:4px"></div>
                            </div>
                        </div>""",
                        unsafe_allow_html=True,
                    )
            else:
                st.caption(m4.get("m4_verdict", "No claims extracted."))

        # Evidence
        ev_text = m2["context"] or m4.get("m4_evidence_used", "")
        ev_src  = m2.get("source", "Wikipedia")
        st.subheader("Supporting evidence (M2 / M4)")
        if ev_text:
            st.markdown(
                f"""<div class="evidence-box">
                    <strong>{ev_src}</strong><br><br>{ev_text}
                </div>""",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """<div class="no-evidence-box">
                    No Wikipedia evidence could be retrieved for this query.
                </div>""",
                unsafe_allow_html=True,
            )

    # ── RIGHT COLUMN ──────────────────────────────────────────────────────────
    with right:
        st.subheader("Trust score")
        pct = int(trust_score * 100)
        st.markdown(
            f"""<div style="text-align:center;padding:20px 0 10px;">
                <div class="score-ring" style="color:{ring_color}">{pct}%</div>
                <div style="margin-top:12px">
                  <span class="trust-badge {badge_class}">{verdict_text}</span>
                </div>
                <div style="margin-top:8px;font-size:12px;color:#6b7280">
                  Scorer: {scorer_used}
                </div>
            </div>""",
            unsafe_allow_html=True,
        )
        st.progress(trust_score)
        st.divider()

        # Module breakdown cards
        st.subheader("Module breakdown")

        def _module_card(name, score_val, verdict):
            bar_w = int(score_val * 100)
            color = ("#10b981" if score_val >= 0.65 else
                     "#f59e0b" if score_val >= 0.40 else "#ef4444")
            st.markdown(
                f"""<div class="module-card">
                    <div style="display:flex;justify-content:space-between;align-items:baseline">
                        <strong>{name}</strong>
                        <span style="font-size:20px;font-weight:700;color:{color}">{score_val:.2f}</span>
                    </div>
                    <div style="font-size:12px;color:#6b7280;margin-bottom:6px">{verdict}</div>
                    <div style="background:#e5e7eb;border-radius:4px;height:8px">
                      <div style="width:{bar_w}%;background:{color};height:8px;border-radius:4px"></div>
                    </div>
                </div>""",
                unsafe_allow_html=True,
            )

        _module_card("M1 — Semantic Consistency", s1, m1["m1_verdict"])
        _module_card("M2 — Factual Grounding",    s2, m2["m2_verdict"])
        _module_card("M3 — Uncertainty",          s3, m3["m3_verdict"])
        _module_card("M4 — NLI Entailment",       s4, m4["m4_verdict"])

        st.divider()

        # SHAP-style bar chart
        st.subheader("Signal contribution (SHAP-style)")
        scores_arr = np.array([s1, s2, s3, s4])
        weights    = np.array([0.25, 0.30, 0.20, 0.25])
        contribs   = scores_arr * weights
        labels_bar = ["M1 Consistency", "M2 Grounding",
                      "M3 Uncertainty", "M4 Entailment"]
        colors_bar = ["#6366f1", "#10b981", "#f59e0b", "#ef4444"]

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(4.5, 2.8))
            bars = ax.barh(labels_bar, contribs, color=colors_bar)
            ax.set_xlabel("Weighted contribution to trust score")
            ax.set_xlim(0, max(contribs.max() * 1.3, 0.1))
            for bar, val in zip(bars, contribs):
                ax.text(val + 0.002, bar.get_y() + bar.get_height() / 2,
                        f"{val:.3f}", va="center", fontsize=9)
            ax.set_title("Per-module contribution", fontsize=10)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        except Exception:
            for lbl, val in zip(labels_bar, contribs):
                st.caption(f"{lbl}: {val:.3f}")

        st.divider()

        # Formula
        st.caption(
            f"Trust = Σ(module × weight)  \n"
            f"= {s1:.2f}×0.25 + {s2:.2f}×0.30 + {s3:.2f}×0.20 + {s4:.2f}×0.25  \n"
            f"= **{trust_score:.4f}**"
        )

        with st.expander("Raw module output (debug)"):
            st.json({
                "M1": m1, "M2": m2, "M3": m3, "M4": m4,
                "trust_score": trust_score,
                "scorer_used": scorer_used,
                "llm_used": selected_model if mode == "🤖 Auto — Ask Ollama" else "manual",
            })