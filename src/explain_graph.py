"""Explanations for the models trained on graph features.

- TreeSHAP for LightGBM (exact, in log-odds space), per feature and summed
  per feature family.
- Family ablation: CV accuracy without each family, and with it alone.
- A depth-5 surrogate tree trained to mimic LightGBM.
- Agreement between SHAP, LightGBM gain and permutation importance.
"""
import json
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import shap
from rdflib import URIRef
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score

from src.dataset import load_dataset, load_graph, person_nodes
from src.experiments import RESULTS_DIR, SEED, model_specs, tune
from src.explain import fidelity_score, fit_surrogate, rank_agreement
from src.features import FEATURE_SETS, build_features, node_name

FEATURE_SET = "graph"
TOP_K = 20


@dataclass
class GraphSetup:
    X: pd.DataFrame            # labelled persons: train rows, then test rows
    y: np.ndarray
    n_train: int
    X_unlabeled: pd.DataFrame  # persons in the graph without a label
    class_names: list
    person_names: dict         # labelled person URI -> name

    @property
    def X_train(self):
        return self.X.iloc[: self.n_train]

    @property
    def X_test(self):
        return self.X.iloc[self.n_train:]

    @property
    def y_train(self):
        return self.y[: self.n_train]

    @property
    def y_test(self):
        return self.y[self.n_train:]


def load_setup() -> GraphSetup:
    """Graph features for labelled and unlabelled persons (same columns)."""
    dataset = load_dataset()
    g = load_graph()
    labelled = list(dataset.X_labeled_raw.index)
    unlabelled = sorted(
        str(p) for p in person_nodes(g) if str(p) not in set(labelled)
    )
    table = build_features(g, labelled + unlabelled, FEATURE_SETS[FEATURE_SET])
    return GraphSetup(
        X=table.loc[labelled],
        y=dataset.y_labeled,
        n_train=len(dataset.y_train),
        X_unlabeled=table.loc[unlabelled],
        class_names=[
            node_name(g, URIRef(c)) for c in dataset.label_encoder.classes_
        ],
        person_names={p: node_name(g, URIRef(p)) for p in labelled},
    )


def fit_model(setup: GraphSetup, model_name: str):
    """The model with hyperparameters tuned on the training persons."""
    pipeline, grid = model_specs(min_support=2)[model_name]
    return tune(pipeline, grid, setup.X_train, setup.y_train).best_estimator_


def kept_features(pipeline, columns) -> list:
    return list(np.asarray(columns)[pipeline[0].get_support()])


def tree_shap(pipeline, X: pd.DataFrame) -> shap.Explanation:
    """Exact TreeSHAP values (log-odds) for a filter + tree-model pipeline."""
    explanation = shap.TreeExplainer(pipeline[-1])(pipeline[0].transform(X))
    explanation.feature_names = kept_features(pipeline, X.columns)
    return explanation


def mean_abs_shap(explanation: shap.Explanation) -> pd.Series:
    """Mean |SHAP| per feature, over persons and classes, descending."""
    values = np.abs(explanation.values).mean(axis=(0, 2))
    return pd.Series(values, index=explanation.feature_names).sort_values(
        ascending=False
    )


def family_of(feature: str) -> str:
    return feature.split(":", 1)[0]


def family_share(importance: pd.Series) -> pd.Series:
    """Share of total importance per feature family (sums to 1)."""
    totals = importance.groupby(importance.index.map(family_of)).sum()
    return (totals / totals.sum()).sort_values(ascending=False)


def class_drivers(explanation, y, class_names, top_n: int = 5) -> dict:
    """Per class: features that push its own members towards it most."""
    drivers = {}
    for c, name in enumerate(class_names):
        mean_push = explanation.values[y == c, :, c].mean(axis=0)
        order = np.argsort(-mean_push)[:top_n]
        drivers[name] = [
            [explanation.feature_names[i], round(float(mean_push[i]), 3)]
            for i in order
        ]
    return drivers


def family_ablation(pipeline, X: pd.DataFrame, y, n_repeats=3) -> dict:
    """CV accuracy without each family, and with only that family.

    Hyperparameters stay fixed at the values tuned on the training persons.
    """
    cv = RepeatedStratifiedKFold(
        n_splits=5, n_repeats=n_repeats, random_state=SEED
    )

    def accuracy(columns) -> float:
        scores = cross_val_score(
            clone(pipeline), X[columns].to_numpy(), y, cv=cv, n_jobs=-1
        )
        return float(scores.mean())

    all_columns = list(X.columns)
    full = accuracy(all_columns)
    families = {}
    for family in sorted({family_of(c) for c in all_columns}):
        own = [c for c in all_columns if family_of(c) == family]
        rest = [c for c in all_columns if family_of(c) != family]
        without = accuracy(rest)
        families[family] = {
            "accuracy_without": without,
            "drop_without": full - without,
            "accuracy_only": accuracy(own),
        }
    return {"accuracy_all_families": full, "families": families}


def importance_agreement(pipeline, setup: GraphSetup, shap_importance):
    """Rank agreement of SHAP, LightGBM gain and permutation importance."""
    kept = kept_features(pipeline, setup.X.columns)
    gain = pipeline[-1].booster_.feature_importance(importance_type="gain")
    perm = permutation_importance(
        pipeline,
        setup.X_test.to_numpy(),
        setup.y_test,
        n_repeats=10,
        random_state=SEED,
        scoring="accuracy",
    ).importances_mean
    perm = pd.Series(perm, index=setup.X.columns)[kept]

    rankings = {
        "SHAP": shap_importance.reindex(kept).to_numpy(),
        "Gain": gain,
        "Permutation": perm.to_numpy(),
    }
    frames = {
        name: pd.DataFrame({"feature": kept, "score": scores})
        for name, scores in rankings.items()
    }
    agreement = {}
    for a, b in combinations(frames, 2):
        agreement[f"{a} vs {b}"] = rank_agreement(
            frames[a], "score", frames[b], "score", TOP_K
        )
    agreement["permutation_nonzero_features"] = int((perm != 0).sum())
    return agreement


def run_graph_explanations(results_dir: Path = RESULTS_DIR) -> dict:
    setup = load_setup()
    lgbm = fit_model(setup, "lightgbm")

    # Global SHAP over all labelled persons: describes the model, it
    # doesn't evaluate it.
    explanation = tree_shap(lgbm, setup.X)
    importance = mean_abs_shap(explanation)

    surrogate = fit_surrogate(
        lgbm, np.vstack([setup.X_train, setup.X_unlabeled]), max_depth=5
    )
    results = {
        "feature_set": FEATURE_SET,
        "model": "lightgbm",
        "top_features_mean_abs_shap": [
            [name, round(float(value), 4)]
            for name, value in importance.head(TOP_K).items()
        ],
        "family_shap_share": family_share(importance).round(4).to_dict(),
        "class_drivers": class_drivers(
            explanation, setup.y, setup.class_names
        ),
        "family_ablation": family_ablation(lgbm, setup.X, setup.y),
        "surrogate_fidelity_test": fidelity_score(
            lgbm, surrogate, setup.X_test.to_numpy()
        ),
        "importance_agreement": importance_agreement(lgbm, setup, importance),
    }

    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "graph_explanations.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return results


if __name__ == "__main__":
    print(json.dumps(run_graph_explanations(), indent=2, ensure_ascii=False))
