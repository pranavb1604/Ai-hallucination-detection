import streamlit as st

from modules.m1_consistency import score as m1_score
from modules.m2_grounding import score as m2_score

st.set_page_config(page_title="AI Trust Evaluation", layout="wide")

st.title("AI Trust Evaluation System")

question = st.text_input("Enter your question")

if st.button("Evaluate"):

    if not question.strip():
        st.warning("Enter a valid question")
        st.stop()

    # TEMP responses (replace later with LLM)
    responses = [
        "The Eiffel Tower is in Paris, France.",
        "It is located in Paris.",
        "The Eiffel Tower is in Paris."
    ]

    # DEBUG
    st.write("Running evaluation...")

    try:
        m1 = m1_score(question, responses)
        m2 = m2_score(question, responses)
    except Exception as e:
        st.error(f"Error: {e}")
        st.stop()

    final_score = m1["m1_score"] * m2["m2_score"]

    col1, col2 = st.columns([2, 1])

    # LEFT
    with col1:
        st.subheader("Answer")
        st.write(responses[0])

        st.subheader("Responses")
        for r in responses:
            st.write("- ", r)

        st.subheader("Supporting Evidence")
        if m2.get("context"):
            st.write(m2["context"])
        else:
            st.warning("No supporting evidence found.")

    # RIGHT
    with col2:
        st.subheader("Trust Score")

        st.metric("Score", f"{final_score:.2f}")
        st.progress(final_score)

        if final_score >= 0.75:
            st.success("Trusted")
        elif final_score >= 0.5:
            st.warning("Uncertain")
        else:
            st.error("Hallucination Risk")

        st.subheader("Breakdown")
        st.write(f"M1: {m1['m1_score']:.2f}")
        st.write(f"M2: {m2['m2_score']:.2f}")