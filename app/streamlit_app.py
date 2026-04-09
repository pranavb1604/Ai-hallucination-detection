"""
AI Hallucination Detection & Trust Calibration System
B.Tech Project-I  |  Delhi Technological University
"""

import streamlit as st

from modules.m1_consistency import score as m1_score
from modules.m2_grounding import score as m2_score

st.set_page_config(
    page_title="AI Trust Evaluator",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .trust-badge {
        display: inline-block;
        padding: 6px 18px;
        border-radius: 20px;
        font-size: 15px;
        font-weight: 600;
        letter-spacing: 0.3px;
    }
    .badge-trusted   { background: #d1fae5; color: #065f46; }
    .badge-uncertain { background: #fef3c7; color: #78350f; }
    .badge-suspicious{ background: #fee2e2; color: #7f1d1d; }

    .score-ring {
        font-size: 52px;
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
</style>
""", unsafe_allow_html=True)


# ─── Sidebar — Settings ───────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")

    st.subheader("Module weights")
    w1 = st.slider("M1 — Consistency weight", 0.0, 1.0, 0.5, 0.05)
    w2 = st.slider("M2 — Grounding weight",   0.0, 1.0, 0.5, 0.05)

    total_w = w1 + w2
    if total_w > 0:
        w1_norm = w1 / total_w
        w2_norm = w2 / total_w
    else:
        w1_norm = w2_norm = 0.5

    st.caption(f"Normalised → M1: {w1_norm:.2f}  |  M2: {w2_norm:.2f}")

    st.divider()

    st.subheader("Thresholds")
    thresh_trusted    = st.slider("Trusted ≥",    0.0, 1.0, 0.75, 0.05)
    thresh_uncertain  = st.slider("Uncertain ≥",  0.0, 1.0, 0.50, 0.05)

    st.divider()

    st.subheader("Demo mode")
    demo_topic = st.selectbox(
        "Pre-fill example",
        ["(none)", "Eiffel Tower", "Black holes", "Aspirin mechanism"],
    )

    demo_responses_map = {
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
    }


# ─── Main area ────────────────────────────────────────────────────────────────
st.title("🔍 AI Hallucination Detection")
st.caption("Multi-signal Trust Calibration System  ·  M1 Consistency + M2 Grounding")

# ─── Input ────────────────────────────────────────────────────────────────────
prefill_q = ""
prefill_r = ""

if demo_topic != "(none)" and demo_topic in demo_responses_map:
    demo = demo_responses_map[demo_topic]
    prefill_q = demo["q"]
    prefill_r = "\n".join(demo["r"])

st.subheader("Input")
col_q, col_r = st.columns([1, 1], gap="large")

with col_q:
    question = st.text_area(
        "Question asked to the LLM",
        value=prefill_q,
        height=100,
        placeholder="e.g. Where is the Eiffel Tower located?",
    )

with col_r:
    raw_responses = st.text_area(
        "LLM responses (one per line — sample the same question multiple times)",
        value=prefill_r,
        height=100,
        placeholder="Response 1\nResponse 2\nResponse 3",
    )

responses = [r.strip() for r in raw_responses.strip().splitlines() if r.strip()]

col_btn, col_info = st.columns([1, 3])
with col_btn:
    run = st.button("▶ Evaluate", type="primary", use_container_width=True)
with col_info:
    if responses:
        st.caption(f"✅ {len(responses)} response(s) loaded")
    else:
        st.caption("Enter at least 1 response to evaluate")

st.divider()

# ─── Evaluation ───────────────────────────────────────────────────────────────
if run:
    if not question.strip():
        st.warning("Please enter a question.")
        st.stop()
    if not responses:
        st.warning("Please enter at least one LLM response.")
        st.stop()

    with st.spinner("Running M1 consistency check…"):
        m1 = m1_score(question, responses)

    with st.spinner("Running M2 grounding check…"):
        m2 = m2_score(question, responses)

    final_score = round(m1["m1_score"] * w1_norm + m2["m2_score"] * w2_norm, 4)

    # ── Verdict ──────────────────────────────────────────────────────────────
    if final_score >= thresh_trusted:
        badge_class = "badge-trusted"
        verdict_text = "✅ Trusted"
        verdict_color = "success"
    elif final_score >= thresh_uncertain:
        badge_class = "badge-uncertain"
        verdict_text = "⚠️ Uncertain"
        verdict_color = "warning"
    else:
        badge_class = "badge-suspicious"
        verdict_text = "🚨 Hallucination Risk"
        verdict_color = "error"

    # ── Layout ───────────────────────────────────────────────────────────────
    left, right = st.columns([3, 2], gap="large")

    with left:
        # ── Primary answer ───────────────────────────────────────────────────
        st.subheader("Primary answer")
        st.info(responses[0])

        # ── All sampled responses ─────────────────────────────────────────────
        with st.expander(f"All {len(responses)} sampled responses"):
            for idx, r in enumerate(responses, 1):
                st.markdown(f"**{idx}.** {r}")

        # ── M1 pair breakdown ─────────────────────────────────────────────────
        with st.expander("M1 — Pairwise similarity breakdown"):
            pairs = m1.get("m1_pairs", [])
            if pairs:
                for p in pairs:
                    sim = p["similarity"]
                    color = "#10b981" if sim >= 0.7 else ("#f59e0b" if sim >= 0.45 else "#ef4444")
                    bar_w = int(sim * 100)
                    st.markdown(
                        f"""<div class="pair-row">
                            Response {p['i']+1} ↔ Response {p['j']+1} &nbsp;
                            <span style="color:{color}; font-weight:600">{sim:.2f}</span>
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

        # ── Evidence ──────────────────────────────────────────────────────────
        st.subheader("Supporting evidence")
        if m2["found"]:
            st.markdown(
                f"""<div class="evidence-box">
                    <strong>{m2['source']}</strong><br><br>{m2['context']}
                </div>""",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """<div class="no-evidence-box">
                    No Wikipedia evidence could be retrieved for this query.
                    M2 grounding score defaults to 0.
                </div>""",
                unsafe_allow_html=True,
            )

    with right:
        # ── Trust score ───────────────────────────────────────────────────────
        st.subheader("Trust score")

        pct = int(final_score * 100)
        ring_color = "#10b981" if final_score >= thresh_trusted else (
            "#f59e0b" if final_score >= thresh_uncertain else "#ef4444"
        )

        st.markdown(
            f"""<div style="text-align:center; padding: 20px 0 10px;">
                <div class="score-ring" style="color:{ring_color}">{pct}%</div>
                <div style="margin-top:12px">
                  <span class="trust-badge {badge_class}">{verdict_text}</span>
                </div>
            </div>""",
            unsafe_allow_html=True,
        )

        st.progress(final_score)

        st.divider()

        # ── Module breakdown ──────────────────────────────────────────────────
        st.subheader("Module breakdown")

        def module_card(name, score_val, verdict, weight):
            bar_w = int(score_val * 100)
            color = "#10b981" if score_val >= 0.65 else (
                "#f59e0b" if score_val >= 0.40 else "#ef4444"
            )
            st.markdown(
                f"""<div class="module-card">
                    <div style="display:flex;justify-content:space-between;align-items:baseline">
                        <strong>{name}</strong>
                        <span style="font-size:20px;font-weight:700;color:{color}">{score_val:.2f}</span>
                    </div>
                    <div style="font-size:12px;color:#6b7280;margin-bottom:6px">
                        {verdict} &nbsp;·&nbsp; weight {weight:.2f}
                    </div>
                    <div style="background:#e5e7eb;border-radius:4px;height:8px">
                      <div style="width:{bar_w}%;background:{color};height:8px;border-radius:4px"></div>
                    </div>
                </div>""",
                unsafe_allow_html=True,
            )

        module_card("M1 — Semantic consistency", m1["m1_score"], m1["m1_verdict"], w1_norm)
        module_card("M2 — Factual grounding",   m2["m2_score"], m2["m2_verdict"], w2_norm)

        st.divider()

        # ── Formula ───────────────────────────────────────────────────────────
        st.caption(
            f"Trust = M1 × {w1_norm:.2f} + M2 × {w2_norm:.2f}  \n"
            f"= {m1['m1_score']:.2f} × {w1_norm:.2f} + {m2['m2_score']:.2f} × {w2_norm:.2f}  \n"
            f"= **{final_score:.2f}**"
        )

        # ── Raw output ────────────────────────────────────────────────────────
        with st.expander("Raw module output (debug)"):
            st.json({"M1": m1, "M2": m2, "final_score": final_score})