from modules.m1_consistency import score as m1_score
from modules.m2_grounding import score as m2_score

def run_pipeline(question, responses):

    m1 = m1_score(question, responses)
    m2 = m2_score(question, responses)

    final_score = 0.6 * m1["m1_score"] + 0.4 * m2["m2_score"]

    return {
        "final_score": final_score,
        "m1": m1,
        "m2": m2
    }