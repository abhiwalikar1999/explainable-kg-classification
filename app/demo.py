"""Streamlit demo: why does the model put this researcher in that group?

Run from the repository root:  streamlit run app/demo.py

It shows explanations precomputed by `python -m src.pipeline --stages demo`
(`outputs/results/demo_explanations.json`), so it starts in seconds and
needs only streamlit and pandas.
"""
import json
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
EXPLANATIONS = ROOT / "outputs" / "results" / "demo_explanations.json"
FIGURES = ROOT / "outputs" / "figures"

st.set_page_config(page_title="Explainable KG classification", layout="wide")


@st.cache_data
def explanations() -> list:
    return json.loads(EXPLANATIONS.read_text(encoding="utf-8"))["persons"]


persons = explanations()

st.title("Why is this researcher in that group?")
st.caption(
    "AIFB knowledge graph · readable graph features · both models were "
    f"tuned on the training persons and never saw these {len(persons)} "
    "test persons."
)

choice = st.selectbox(
    "Researcher (test set)",
    range(len(persons)),
    format_func=lambda i: persons[i]["name"],
)
e = persons[choice]


def verdict(prediction: str) -> str:
    return "✓ correct" if prediction == e["true_group"] else "✗ wrong"


true_col, tree_col, lgbm_col = st.columns(3)
true_col.metric("True group", e["true_group"])
tree_col.metric("Decision tree", e["tree_prediction"])
tree_col.caption(verdict(e["tree_prediction"]))
lgbm_col.metric("LightGBM", e["lightgbm_prediction"])
lgbm_col.caption(
    f"{verdict(e['lightgbm_prediction'])} · "
    f"{max(e['probabilities'].values()):.0%} probability"
)

path_col, reasons_col = st.columns(2)
with path_col:
    st.subheader("Decision tree: the path it took")
    for step in e["tree_path"]:
        st.markdown(f"- {step}")
    st.caption("The whole model is these yes/no questions; nothing is hidden.")

with reasons_col:
    st.subheader("LightGBM: strongest reasons")
    reasons = pd.DataFrame(e["top_reasons"]).set_axis(
        ["Feature", "Value", "Push towards prediction"], axis=1
    )
    st.dataframe(
        reasons.style.format({"Value": "{:g}",
                              "Push towards prediction": "{:+.2f}"}),
        hide_index=True,
        width="stretch",
    )
    st.caption(
        "SHAP values in log-odds: positive pushes towards "
        f"“{e['lightgbm_prediction']}”, negative away from it."
    )

st.subheader("Which kind of information decided it")
families = pd.Series(e["family_totals"], name="SHAP total").sort_values()
st.bar_chart(families, horizontal=True)

with st.expander("Probabilities for every group"):
    st.dataframe(
        pd.Series(e["probabilities"], name="probability").to_frame().style
        .format("{:.1%}"),
        width="stretch",
    )

with st.expander("Global results (all persons)"):
    for name in ("model_comparison", "family_importance"):
        figure = FIGURES / f"{name}.png"
        if figure.exists():
            st.image(str(figure), width="stretch")
