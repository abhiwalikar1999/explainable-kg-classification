"""Explain one person's prediction in plain language (used by the demo).

For a single person: the decision tree's path as sentences, LightGBM's
class probabilities, its strongest SHAP reasons, and the SHAP total per
feature family.

`export_demo` stores this for every test person in
`outputs/results/demo_explanations.json`, so the demo only reads a file.
"""
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from src.experiments import RESULTS_DIR
from src.explain_graph import (
    GraphSetup,
    family_of,
    fit_model,
    kept_features,
    load_setup,
    tree_shap,
)

FAMILY_LABELS = {
    "topic": "topic “{}”",
    "coauthor": "co-authored with {}",
    "project": "works on project {}",
    "contact": "contact detail {}",
}
PROFILE_LABELS = {
    "n_publications": "number of publications",
    "n_projects": "number of projects",
    "n_coauthors": "number of co-authors",
    "n_edited": "number of edited volumes",
    "has_phone": "has a phone number",
    "has_fax": "has a fax number",
    "has_homepage": "has a homepage",
    "has_photo": "has a photo",
}
DEMO_FILE = RESULTS_DIR / "demo_explanations.json"


@dataclass
class Models:
    setup: GraphSetup
    tree: object      # tuned MinSupportFilter + DecisionTree pipeline
    lightgbm: object  # tuned MinSupportFilter + LightGBM pipeline


@dataclass
class PersonExplanation:
    name: str
    true_group: str
    tree_prediction: str
    tree_path: list           # sentences, root to leaf
    lightgbm_prediction: str
    probabilities: dict       # group -> probability
    top_reasons: list         # (feature, value, SHAP towards the prediction)
    family_totals: dict       # family -> summed SHAP towards the prediction


def load_models() -> Models:
    setup = load_setup()
    return Models(
        setup=setup,
        tree=fit_model(setup, "decision_tree"),
        lightgbm=fit_model(setup, "lightgbm"),
    )


def readable(feature: str) -> str:
    """'coauthor:Rudi Studer' -> 'co-authored with Rudi Studer'."""
    family, _, item = feature.partition(":")
    if family == "profile":
        return PROFILE_LABELS.get(item, item.replace("=", ": "))
    template = FAMILY_LABELS.get(family)
    return template.format(item) if template else feature


def split_sentence(feature: str, value: float, threshold: float) -> str:
    """One step of the tree path, phrased for a reader."""
    label = readable(feature)
    if threshold == 0.5 and value in (0.0, 1.0):
        return f"{label}: {'yes' if value else 'no'}"
    side = "≤" if value <= threshold else ">"
    return f"{label}: {value:g} ({side} {threshold:g})"


def tree_path(tree_pipeline, row, class_names) -> list:
    """Root-to-leaf path of one person through the tuned tree."""
    names = kept_features(tree_pipeline, row.columns)
    x = tree_pipeline[0].transform(row)[0]
    tree = tree_pipeline[-1].tree_
    node, sentences = 0, []
    while tree.children_left[node] != tree.children_right[node]:
        feature, threshold = tree.feature[node], tree.threshold[node]
        sentences.append(split_sentence(names[feature], x[feature], threshold))
        go_left = x[feature] <= threshold
        node = tree.children_left[node] if go_left else tree.children_right[
            node
        ]
    leaf_class = class_names[int(np.argmax(tree.value[node][0]))]
    sentences.append(f"→ {leaf_class}")
    return sentences


def explain_person(models: Models, person: str, top_n: int = 8):
    """Everything the demo shows for one labelled person."""
    setup = models.setup
    row = setup.X.loc[[person]]
    position = list(setup.X.index).index(person)
    classes = setup.class_names

    probabilities = models.lightgbm.predict_proba(row.to_numpy())[0]
    predicted = int(np.argmax(probabilities))
    explanation = tree_shap(models.lightgbm, row)
    shap_values = explanation.values[0, :, predicted]
    values = explanation.data[0]
    order = np.argsort(-np.abs(shap_values))[:top_n]

    totals = {}
    for name, contribution in zip(explanation.feature_names, shap_values):
        family = family_of(name)
        totals[family] = totals.get(family, 0.0) + float(contribution)

    return PersonExplanation(
        name=setup.person_names[person],
        true_group=classes[int(setup.y[position])],
        tree_prediction=classes[
            int(models.tree.predict(row.to_numpy())[0])
        ],
        tree_path=tree_path(models.tree, row, classes),
        lightgbm_prediction=classes[predicted],
        probabilities={
            c: float(p) for c, p in zip(classes, probabilities)
        },
        top_reasons=[
            (explanation.feature_names[i], float(values[i]),
             float(shap_values[i]))
            for i in order
        ],
        family_totals=totals,
    )


def demo_record(e: PersonExplanation) -> dict:
    """One explanation as stored for the demo, with readable labels."""
    record = asdict(e)
    record["top_reasons"] = [
        {"feature": readable(feature), "value": value, "shap": shap}
        for feature, value, shap in e.top_reasons
    ]
    return record


def export_demo(path: Path = DEMO_FILE) -> dict:
    """Explain every test person once, so the demo needs no models."""
    models = load_models()
    names = models.setup.person_names
    persons = sorted(models.setup.X_test.index, key=names.get)
    data = {
        "description": (
            "Per test person: the tuned decision tree's path and LightGBM's "
            "prediction with its TreeSHAP reasons (graph features). Written "
            "by `python -m src.pipeline --stages demo`; read by app/demo.py."
        ),
        "class_names": models.setup.class_names,
        "persons": [demo_record(explain_person(models, p)) for p in persons],
    }
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return data
