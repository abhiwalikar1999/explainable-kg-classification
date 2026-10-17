"""Evaluation: baseline, confidence intervals, repeated CV, explanations."""
import json
from pathlib import Path

import numpy as np
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline

from src.dataset import PROJECT_ROOT, load_dataset
from src.explain import (
    evaluation_summary,
    run_dt_explanations,
    run_gb_explanations,
)
from src.model import get_models

DEFAULT_RESULTS_PATH = PROJECT_ROOT / "outputs/results/metrics.json"


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple:
    """Wilson score interval for a proportion such as test accuracy."""
    p = successes / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return float(centre - half), float(centre + half)


def majority_baseline(y_train, y_test) -> float:
    """Test accuracy of always predicting the most frequent train class."""
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(np.zeros((len(y_train), 1)), y_train)
    return float(dummy.score(np.zeros((len(y_test), 1)), y_test))


def repeated_cv(
    model,
    X,
    y,
    n_splits: int = 5,
    n_repeats: int = 3,
    variance_threshold: float = 0.001,
    random_state: int = 0,
) -> np.ndarray:
    """Accuracy per fold of repeated stratified CV.

    The variance filter is refit inside every fold, so held-out persons
    never influence which features are kept.
    """
    pipeline = make_pipeline(
        VarianceThreshold(variance_threshold), clone(model)
    )
    cv = RepeatedStratifiedKFold(
        n_splits=n_splits, n_repeats=n_repeats, random_state=random_state
    )
    return cross_val_score(pipeline, X, y, cv=cv, n_jobs=-1)


def run_evaluation(
    results_path: Path = DEFAULT_RESULTS_PATH, n_repeats: int = 3
) -> dict:
    """Evaluate the saved checkpoints and write the results as JSON."""
    dataset = load_dataset()
    gb_expl = run_gb_explanations(dataset)
    dt_expl = run_dt_explanations(dataset)
    n_test = len(dataset.y_test)

    results = {
        "majority_baseline_test_accuracy": majority_baseline(
            dataset.y_train, dataset.y_test
        ),
        "models": {},
    }
    unfitted = get_models()
    for name, expl in (
        ("gradient_boosting", gb_expl),
        ("decision_tree", dt_expl),
    ):
        y_pred = expl["model"].predict(dataset.X_test)
        correct = int(np.sum(y_pred == dataset.y_test))
        cv_scores = repeated_cv(
            unfitted[name],
            dataset.X_labeled_raw.to_numpy(),
            dataset.y_labeled,
            n_repeats=n_repeats,
        )
        results["models"][name] = {
            "test_accuracy": correct / n_test,
            "test_correct": f"{correct}/{n_test}",
            "test_accuracy_wilson95": wilson_interval(correct, n_test),
            "cv_accuracy_mean": float(cv_scores.mean()),
            "cv_accuracy_std": float(cv_scores.std()),
            "cv_folds": len(cv_scores),
        }

    summary = evaluation_summary(
        gb_expl["model"], dt_expl["model"], gb_expl, dt_expl, dataset
    )
    results["explanations"] = dict(zip(summary["metric"], summary["value"]))
    results["permutation_importance_nonzero_features"] = int(
        (gb_expl["perm_df"]["perm_importance"] != 0).sum()
    )

    results_path = Path(results_path)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(results, indent=2))
    return results


if __name__ == "__main__":
    print(json.dumps(run_evaluation(), indent=2))
