import numpy as np
import pandas as pd
import pytest

from src.explain_graph import class_drivers, family_of, family_share


def test_family_is_the_prefix_before_the_first_colon():
    assert family_of("contact:fax=0721:1") == "contact"


def test_family_share_sums_per_family_to_one():
    importance = pd.Series(
        {"topic:a": 3.0, "topic:b": 1.0, "profile:n": 4.0}
    )
    share = family_share(importance)
    assert share.to_dict() == pytest.approx({"topic": 0.5, "profile": 0.5})


class FakeExplanation:
    def __init__(self, values, feature_names):
        self.values = values
        self.feature_names = feature_names


def test_class_drivers_rank_features_by_push_towards_own_class():
    # 2 persons x 2 features x 2 classes; person 0 is class 0, person 1 is
    # class 1. Feature "b" pushes class 0 hardest, feature "a" class 1.
    values = np.array(
        [
            [[0.1, 0.0], [0.9, 0.0]],
            [[0.0, 0.7], [0.0, 0.2]],
        ]
    )
    drivers = class_drivers(
        FakeExplanation(values, ["a", "b"]),
        np.array([0, 1]),
        ["first", "second"],
        top_n=1,
    )
    assert drivers == {"first": [["b", 0.9]], "second": [["a", 0.7]]}
