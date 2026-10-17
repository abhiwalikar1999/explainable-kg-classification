"""Model-quality gate, run as its own CI step (`pytest -m quality`).

1. Reproducibility: retraining the graph-feature models from the code must
   give the test accuracy published in `outputs/results/experiments.json`.
   The tolerance is one test person, to absorb tiny cross-platform float
   differences in tree splits.
2. Floors: accuracy must not fall below agreed minimums, on the test split
   and in a quick (5-fold) nested CV for the fast models.
"""
import json

import pytest
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_score,
)

from src.experiments import RESULTS_DIR, inner_cv, model_specs
from src.explain_graph import fit_model, load_setup

pytestmark = pytest.mark.quality

TEST_FLOORS = {"logreg": 0.80, "decision_tree": 0.80, "lightgbm": 0.85}
CV_FLOORS = {"logreg": 0.80, "decision_tree": 0.80}
ONE_TEST_PERSON = 1 / 36


@pytest.fixture(scope="module")
def setup():
    return load_setup()


@pytest.fixture(scope="module")
def published():
    results = json.loads((RESULTS_DIR / "experiments.json").read_text())
    return results["experiments"]["graph"]


@pytest.mark.parametrize("model_name", sorted(TEST_FLOORS))
def test_published_test_accuracy_reproduces(setup, published, model_name):
    model = fit_model(setup, model_name)
    accuracy = model.score(setup.X_test.to_numpy(), setup.y_test)
    assert accuracy == pytest.approx(
        published[model_name]["test_accuracy"], abs=ONE_TEST_PERSON + 1e-9
    )
    assert accuracy >= TEST_FLOORS[model_name]


@pytest.mark.parametrize("model_name", sorted(CV_FLOORS))
def test_nested_cv_accuracy_floor(setup, model_name):
    pipeline, grid = model_specs(min_support=2)[model_name]
    outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    scores = cross_val_score(
        GridSearchCV(pipeline, grid, cv=inner_cv()),
        setup.X.to_numpy(),
        setup.y,
        cv=outer,
        n_jobs=-1,
    )
    assert scores.mean() >= CV_FLOORS[model_name]
