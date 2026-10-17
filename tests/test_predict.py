import json

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier

from src.features import FAMILIES, MinSupportFilter
from src.predict import (
    DEMO_FILE,
    demo_record,
    explain_person,
    load_models,
    readable,
    split_sentence,
)
from src.predict import tree_path as path_of


@pytest.mark.parametrize(
    "feature, expected",
    [
        ("coauthor:Rudi Studer", "co-authored with Rudi Studer"),
        ("topic:semantic web", "topic “semantic web”"),
        ("profile:has_photo", "has a photo"),
        ("profile:n_publications", "number of publications"),
        ("profile:role=PhDStudent", "role: PhDStudent"),
    ],
)
def test_readable_feature_names(feature, expected):
    assert readable(feature) == expected


def test_split_sentences():
    assert split_sentence("coauthor:A", 1.0, 0.5) == "co-authored with A: yes"
    assert split_sentence("topic:x", 3.0, 0.5) == "topic “x”: 3 (> 0.5)"


def test_tree_path_ends_in_the_predicted_class():
    X = pd.DataFrame(
        {"topic:a": [0, 0, 1, 1, 1, 0], "coauthor:B": [1, 0, 1, 0, 1, 0]},
        dtype=float,
    )
    y = np.array([0, 0, 1, 1, 1, 0])
    tree = make_pipeline(
        MinSupportFilter(2), DecisionTreeClassifier(random_state=0)
    ).fit(X, y)
    path = path_of(tree, X.iloc[[2]], ["no", "yes"])
    assert path == ["topic “a”: yes", "→ yes"]


@pytest.fixture(scope="module")
def models():
    return load_models()


def test_explain_person_is_consistent(models):
    person = models.setup.X_test.index[0]
    e = explain_person(models, person)
    assert sum(e.probabilities.values()) == pytest.approx(1.0)
    most_likely = max(e.probabilities, key=e.probabilities.get)
    assert e.lightgbm_prediction == most_likely
    assert e.tree_path[-1] == f"→ {e.tree_prediction}"
    strengths = [abs(shap) for _, _, shap in e.top_reasons]
    assert strengths == sorted(strengths, reverse=True)
    assert set(e.family_totals) <= set(FAMILIES)


def test_demo_file_matches_the_models(models):
    """The demo's stored explanations are what the code produces now."""
    stored = json.loads(DEMO_FILE.read_text(encoding="utf-8"))["persons"]
    names = models.setup.person_names
    test_persons = {names[p]: p for p in models.setup.X_test.index}
    assert sorted(r["name"] for r in stored) == sorted(test_persons)
    for record in stored:
        person = test_persons[record["name"]]
        fresh = demo_record(explain_person(models, person))
        for key in ("true_group", "tree_prediction", "tree_path",
                    "lightgbm_prediction"):
            assert record[key] == fresh[key]
        assert record["probabilities"] == pytest.approx(
            fresh["probabilities"], abs=1e-4
        )
