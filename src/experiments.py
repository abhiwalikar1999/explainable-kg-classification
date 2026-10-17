"""Feature sets x models, tuned on train and evaluated honestly.

For every combination: hyperparameters are chosen by GridSearchCV on the
140 training persons, the chosen model is scored on the 36 test persons
(with a Wilson interval), and the whole tuning procedure is repeated inside
nested, repeated stratified CV on all 176 persons.
"""
import json
from pathlib import Path

import numpy as np
from lightgbm import LGBMClassifier
from rdflib import URIRef
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import (
    GridSearchCV,
    RepeatedStratifiedKFold,
    StratifiedKFold,
    cross_val_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text

from src.dataset import PROJECT_ROOT, load_dataset, load_graph
from src.evaluate import majority_baseline, wilson_interval
from src.features import (
    FEATURE_SETS,
    MinSupportFilter,
    build_features,
    node_name,
)

RESULTS_DIR = PROJECT_ROOT / "outputs/results"
SEED = 0


def feature_matrices(dataset, g) -> dict:
    """Feature-set name -> (table of all labelled persons, min_support).

    "onehot" is the original pipeline's encoding; "onehot_no_ids" is the
    same table without values seen for a single training person.
    """
    persons = list(dataset.X_labeled_raw.index)
    onehot = dataset.X_labeled_raw.astype(float)
    matrices = {"onehot": (onehot, 1), "onehot_no_ids": (onehot, 2)}
    for name, families in FEATURE_SETS.items():
        matrices[name] = (build_features(g, persons, families), 2)
    return matrices


def model_specs(min_support: int) -> dict:
    """Model name -> (pipeline, hyperparameter grid)."""

    def keep():
        return MinSupportFilter(min_support)

    return {
        "logreg": (
            make_pipeline(
                keep(),
                FunctionTransformer(np.log1p),
                StandardScaler(),
                LogisticRegression(max_iter=5000),
            ),
            {"logisticregression__C": [0.01, 0.1, 1.0, 10.0]},
        ),
        "decision_tree": (
            make_pipeline(keep(), DecisionTreeClassifier(random_state=SEED)),
            {
                "decisiontreeclassifier__max_depth": [2, 3, 4, 5, 6],
                "decisiontreeclassifier__min_samples_leaf": [1, 3, 5],
            },
        ),
        "lightgbm": (
            make_pipeline(
                keep(),
                LGBMClassifier(random_state=SEED, n_jobs=1, verbose=-1),
            ),
            {
                "lgbmclassifier__num_leaves": [4, 8],
                "lgbmclassifier__n_estimators": [100, 300],
                "lgbmclassifier__learning_rate": [0.05],
                "lgbmclassifier__min_child_samples": [3, 10],
            },
        ),
    }


def inner_cv() -> StratifiedKFold:
    return StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)


def tune(pipeline, grid, X_train, y_train) -> GridSearchCV:
    """Grid search with 5-fold CV on the training persons only."""
    search = GridSearchCV(pipeline, grid, cv=inner_cv(), n_jobs=-1)
    return search.fit(X_train, y_train)


def evaluate(pipeline, grid, X, y, n_train: int, n_repeats: int) -> tuple:
    """Tune on train, score on test, and run nested repeated CV."""
    search = tune(pipeline, grid, X[:n_train], y[:n_train])

    y_test = y[n_train:]
    y_pred = search.predict(X[n_train:])
    correct = int(np.sum(y_pred == y_test))

    outer = RepeatedStratifiedKFold(
        n_splits=5, n_repeats=n_repeats, random_state=SEED
    )
    nested = cross_val_score(
        GridSearchCV(pipeline, grid, cv=inner_cv()),
        X,
        y,
        cv=outer,
        n_jobs=-1,
    )

    metrics = {
        "test_accuracy": correct / len(y_test),
        "test_correct": f"{correct}/{len(y_test)}",
        "test_accuracy_wilson95": wilson_interval(correct, len(y_test)),
        "test_f1_macro": float(f1_score(y_test, y_pred, average="macro")),
        "cv_accuracy_mean": float(nested.mean()),
        "cv_accuracy_std": float(nested.std()),
        "cv_folds": len(nested),
        "n_features_used": int(search.best_estimator_[0].get_support().sum()),
        "best_params": {
            key.split("__")[1]: value
            for key, value in search.best_params_.items()
        },
    }
    return metrics, search.best_estimator_


def tree_rules(pipeline, feature_names, class_names) -> str:
    kept = np.asarray(feature_names)[pipeline[0].get_support()]
    return export_text(
        pipeline[-1], feature_names=list(kept), class_names=class_names
    )


def summary_table(results: dict) -> str:
    lines = [
        "| Features | Model | Test accuracy (95% CI) | Nested CV "
        "| #Features |",
        "|---|---|---|---|---|",
    ]
    for set_name, models in results["experiments"].items():
        for model_name, m in models.items():
            low, high = m["test_accuracy_wilson95"]
            lines.append(
                f"| {set_name} | {model_name} "
                f"| {m['test_accuracy']:.1%} ({low:.0%}–{high:.0%}) "
                f"| {m['cv_accuracy_mean']:.1%} ± {m['cv_accuracy_std']:.1%} "
                f"| {m['n_features_used']} |"
            )
    return "\n".join(lines)


def run_experiments(
    n_repeats: int = 3, results_dir: Path = RESULTS_DIR
) -> dict:
    """Run every feature set x model and write the results to disk."""
    dataset = load_dataset()
    g = load_graph()
    y = dataset.y_labeled
    n_train = len(dataset.y_train)
    class_names = [
        node_name(g, URIRef(c)) for c in dataset.label_encoder.classes_
    ]

    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    results = {
        "majority_baseline_test_accuracy": majority_baseline(
            dataset.y_train, dataset.y_test
        ),
        "experiments": {},
    }
    for set_name, (X, min_support) in feature_matrices(dataset, g).items():
        results["experiments"][set_name] = {}
        for model_name, (pipeline, grid) in model_specs(min_support).items():
            metrics, fitted = evaluate(
                pipeline, grid, X.to_numpy(), y, n_train, n_repeats
            )
            results["experiments"][set_name][model_name] = metrics
            if model_name == "decision_tree" and set_name in FEATURE_SETS:
                slug = set_name.replace("+", "_")
                (results_dir / f"tree_rules_{slug}.txt").write_text(
                    tree_rules(fitted, X.columns, class_names),
                    encoding="utf-8",
                )

    (results_dir / "experiments.json").write_text(
        json.dumps(results, indent=2)
    )
    return results


if __name__ == "__main__":
    print(summary_table(run_experiments()))
