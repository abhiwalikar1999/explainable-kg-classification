"""Explanation utilities: SHAP (black-box) + Gini/permutation/tree rules."""
from itertools import combinations
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
from scipy.stats import spearmanr
from sklearn.inspection import permutation_importance
from sklearn.tree import DecisionTreeClassifier, export_text

from src.dataset import PROJECT_ROOT, Dataset

DEFAULT_CHECKPOINT_DIR = PROJECT_ROOT / "outputs/checkpoints"
DEFAULT_FIGURE_DIR = PROJECT_ROOT / "outputs/figures"


def load_model(name: str, checkpoint_dir: Path = DEFAULT_CHECKPOINT_DIR):
    """Load a joblib checkpoint by model name."""
    return joblib.load(Path(checkpoint_dir) / f"{name}.joblib")


def compute_shap_values(
    model,
    X_train: np.ndarray,
    X_test: np.ndarray,
    n_background: int = 50,
    random_state: int = 0,
) -> tuple:
    """Compute SHAP values via PermutationExplainer.

    Uses model.predict_proba so it works for multi-class GBM.
    Returns (explainer, shap_explanation) where
    shap_explanation.values has shape (n_test, n_features, n_classes).
    """
    background = shap.sample(
        X_train, min(n_background, len(X_train)), random_state=random_state
    )
    explainer = shap.Explainer(
        model.predict_proba, background, seed=random_state
    )
    shap_explanation = explainer(X_test)
    return explainer, shap_explanation


def shap_mean_abs(shap_explanation, feature_names: list) -> pd.DataFrame:
    """Mean |SHAP| per feature across all classes (global importance)."""
    values = np.abs(shap_explanation.values)
    mean_abs = values.mean(axis=(0, 2))  # average over samples and classes

    return (
        pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )


def gini_importance(model, feature_names: list) -> pd.DataFrame:
    """Gini (MDI) importance from tree-based model's feature_importances_."""
    importances = model.feature_importances_
    if hasattr(model, "estimators_"):
        # GBM: estimators_ is (n_estimators, n_classes) of sub-regressors
        fi_list = [t.feature_importances_ for t in model.estimators_.flat]
        std = np.std(fi_list, axis=0)
    else:
        std = np.zeros_like(importances)
    return (
        pd.DataFrame(
            {
                "feature": feature_names,
                "gini_importance": importances,
                "gini_std": std,
            }
        )
        .sort_values("gini_importance", ascending=False)
        .reset_index(drop=True)
    )


def perm_importance(
    model,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list,
    n_repeats: int = 10,
    random_state: int = 0,
) -> pd.DataFrame:
    """Permutation importance on the provided split (use test for unbiased)."""
    result = permutation_importance(
        model,
        X,
        y,
        n_repeats=n_repeats,
        random_state=random_state,
        scoring="accuracy",
    )
    return (
        pd.DataFrame(
            {
                "feature": feature_names,
                "perm_importance": result.importances_mean,
                "perm_std": result.importances_std,
            }
        )
        .sort_values("perm_importance", ascending=False)
        .reset_index(drop=True)
    )


def importance_comparison(
    gini_df: pd.DataFrame,
    perm_df: pd.DataFrame,
    top_n: int = 20,
) -> pd.DataFrame:
    """Merge Gini and permutation rankings for top-N by Gini importance."""
    merged = gini_df.head(top_n).merge(perm_df, on="feature", how="left")
    return merged[
        [
            "feature",
            "gini_importance",
            "gini_std",
            "perm_importance",
            "perm_std",
        ]
    ]


def fidelity_score(
    black_box, surrogate, X: np.ndarray
) -> float:
    """Fraction of instances where surrogate agrees with black-box."""
    return float(np.mean(black_box.predict(X) == surrogate.predict(X)))


def fit_surrogate(
    black_box, X_fit: np.ndarray, max_depth: int = 5, random_state: int = 0
) -> DecisionTreeClassifier:
    """Global surrogate: a decision tree trained on black-box predictions.

    X_fit should reach beyond the training set. A black box that fits the
    training labels perfectly predicts exactly those labels there, so a tree
    fit on training predictions alone is just the white-box tree again.
    """
    surrogate = DecisionTreeClassifier(
        max_depth=max_depth, random_state=random_state
    )
    return surrogate.fit(X_fit, black_box.predict(X_fit))


def rank_agreement(
    a_df: pd.DataFrame,
    a_col: str,
    b_df: pd.DataFrame,
    b_col: str,
    top_k: int = 20,
) -> dict:
    """Agreement between two feature-importance rankings.

    Spearman over all features is dominated by the many features both
    methods score ~0, so it is reported next to Spearman over the union of
    the two top-k lists and the Jaccard overlap of the top-k sets.
    """
    merged = (
        a_df[["feature", a_col]]
        .rename(columns={a_col: "a"})
        .merge(
            b_df[["feature", b_col]].rename(columns={b_col: "b"}),
            on="feature",
        )
    )
    top_a = set(a_df.nlargest(top_k, a_col)["feature"])
    top_b = set(b_df.nlargest(top_k, b_col)["feature"])
    union = merged[merged["feature"].isin(top_a | top_b)]
    return {
        "spearman_all": float(spearmanr(merged["a"], merged["b"]).statistic),
        "spearman_top_union": float(
            spearmanr(union["a"], union["b"]).statistic
        ),
        "jaccard_top": len(top_a & top_b) / len(top_a | top_b),
    }


def evaluation_summary(
    gb_model,
    dt_model,
    gb_expl: dict,
    dt_expl: dict,
    dataset: Dataset,
    top_k: int = 20,
) -> pd.DataFrame:
    """Single-table summary of accuracy, fidelity, and ranking agreement."""
    surrogate = fit_surrogate(
        gb_model,
        np.vstack([dataset.X_train, dataset.X_unlabeled]),
        max_depth=dt_model.max_depth,
    )
    metrics = {
        "GB test accuracy": np.mean(
            gb_model.predict(dataset.X_test) == dataset.y_test
        ),
        "DT test accuracy": np.mean(
            dt_model.predict(dataset.X_test) == dataset.y_test
        ),
        "GB-DT agreement on test (DT fit on true labels)": fidelity_score(
            gb_model, dt_model, dataset.X_test
        ),
        "Surrogate fidelity on test (DT fit on GB predictions)": (
            fidelity_score(gb_model, surrogate, dataset.X_test)
        ),
    }

    rankings = [
        ("SHAP", gb_expl["shap_df"], "mean_abs_shap"),
        ("Gini", gb_expl["gini_df"], "gini_importance"),
        ("Permutation", gb_expl["perm_df"], "perm_importance"),
    ]
    for (a_name, a_df, a_col), (b_name, b_df, b_col) in combinations(
        rankings, 2
    ):
        agreement = rank_agreement(a_df, a_col, b_df, b_col, top_k)
        pair = f"{a_name} vs {b_name}"
        metrics[f"Spearman {pair} (all features)"] = agreement[
            "spearman_all"
        ]
        metrics[f"Spearman {pair} (union of top-{top_k})"] = agreement[
            "spearman_top_union"
        ]
        metrics[f"Top-{top_k} Jaccard {pair}"] = agreement["jaccard_top"]

    return pd.DataFrame(
        {
            "metric": list(metrics),
            "value": [round(float(v), 3) for v in metrics.values()],
        }
    )


def decision_tree_rules(model, feature_names: list) -> str:
    """Human-readable rule text from a fitted DecisionTreeClassifier."""
    return export_text(model, feature_names=feature_names)


def run_gb_explanations(dataset: Dataset) -> dict:
    """All GB explanations: SHAP + Gini + permutation."""
    gb = load_model("gradient_boosting")
    feature_names = dataset.feature_names

    explainer, shap_expl = compute_shap_values(
        gb, dataset.X_train, dataset.X_test
    )
    shap_expl.feature_names = list(feature_names)

    gini_df = gini_importance(gb, feature_names)
    perm_df = perm_importance(
        gb, dataset.X_test, dataset.y_test, feature_names
    )
    shap_df = shap_mean_abs(shap_expl, feature_names)

    return {
        "model": gb,
        "explainer": explainer,
        "shap_explanation": shap_expl,
        "shap_df": shap_df,
        "gini_df": gini_df,
        "perm_df": perm_df,
        "comparison": importance_comparison(gini_df, perm_df),
    }


def run_dt_explanations(dataset: Dataset) -> dict:
    """All DT explanations: Gini + permutation + text rules."""
    dt = load_model("decision_tree")
    feature_names = dataset.feature_names

    gini_df = gini_importance(dt, feature_names)
    perm_df = perm_importance(
        dt, dataset.X_test, dataset.y_test, feature_names
    )
    rules = decision_tree_rules(dt, feature_names)

    return {
        "model": dt,
        "gini_df": gini_df,
        "perm_df": perm_df,
        "rules": rules,
    }
