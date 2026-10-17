"""sklearn model definitions: Gradient Boosting (black box) + Decision Tree
(white box)."""
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier


def build_gradient_boosting(
    random_state: int = 0,
) -> GradientBoostingClassifier:
    return GradientBoostingClassifier(
        n_estimators=10000, max_depth=1, random_state=random_state
    )


def build_decision_tree(
    random_state: int = 0, max_depth: int = 5
) -> DecisionTreeClassifier:
    return DecisionTreeClassifier(
        max_depth=max_depth, random_state=random_state
    )


def get_models(random_state: int = 0) -> dict:
    return {
        "gradient_boosting": build_gradient_boosting(random_state),
        "decision_tree": build_decision_tree(random_state),
    }
