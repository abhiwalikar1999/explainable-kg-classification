import numpy as np
import pandas as pd
import pytest
from sklearn.tree import DecisionTreeClassifier

from src.evaluate import majority_baseline, wilson_interval
from src.explain import fidelity_score, fit_surrogate, rank_agreement


def test_wilson_interval_matches_reference():
    low, high = wilson_interval(31, 36)
    assert low == pytest.approx(0.713, abs=1e-3)
    assert high == pytest.approx(0.939, abs=1e-3)


def test_wilson_interval_stays_in_unit_range():
    assert wilson_interval(0, 36)[0] == pytest.approx(0.0, abs=1e-12)
    assert wilson_interval(36, 36)[1] == pytest.approx(1.0, abs=1e-12)


def test_majority_baseline():
    assert majority_baseline([0, 0, 1], [0, 1, 1, 1]) == 0.25


def test_rank_agreement_identical_rankings():
    df = pd.DataFrame({"feature": list("abcde"), "score": [5, 4, 3, 2, 1]})
    result = rank_agreement(df, "score", df, "score", top_k=3)
    assert result == pytest.approx(
        {"spearman_all": 1.0, "spearman_top_union": 1.0, "jaccard_top": 1.0}
    )


def test_rank_agreement_disjoint_top_k():
    a = pd.DataFrame({"feature": list("abcd"), "s": [4, 3, 2, 1]})
    b = pd.DataFrame({"feature": list("abcd"), "s": [1, 2, 3, 4]})
    assert rank_agreement(a, "s", b, "s", top_k=2)["jaccard_top"] == 0.0


def test_surrogate_of_a_tree_is_faithful():
    rng = np.random.default_rng(0)
    X = rng.integers(0, 2, size=(200, 6))
    y = X[:, 0] & X[:, 1]
    black_box = DecisionTreeClassifier(random_state=0).fit(X, y)
    surrogate = fit_surrogate(black_box, X, max_depth=3)
    assert fidelity_score(black_box, surrogate, X) == 1.0
