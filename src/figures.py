"""Explanation figures.

Original pipeline (one-hot features, Gradient Boosting): SHAP, Gini vs
permutation, decision tree. Graph features (LightGBM): top SHAP features,
feature-family importance, the tuned decision tree, one SHAP waterfall.
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import shap
from matplotlib.patches import Patch
from sklearn.tree import plot_tree

from src.dataset import load_dataset, simplify_uri
from src.experiments import RESULTS_DIR
from src.explain import (
    DEFAULT_FIGURE_DIR,
    run_dt_explanations,
    run_gb_explanations,
)
from src.explain_graph import family_of, fit_model, load_setup, tree_shap

TOP_N = 15

# One fixed colour per feature family (categorical slots 1-5 of the
# reference palette, in order), so a family looks the same in every chart.
FAMILY_COLORS = {
    "profile": "#2a78d6",
    "topic": "#eb6834",
    "coauthor": "#1baf7a",
    "contact": "#eda100",
    "project": "#e87ba4",
}
TEXT_MUTED = "#52514e"
GRID = "#e4e3df"


def save_figure(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def importance_bars(ax, df, column, color, error_column=None) -> None:
    """Horizontal bars for the top features of an importance table."""
    top = df.head(TOP_N)[::-1]
    errors = top[error_column] if error_column else None
    ax.barh(top["feature"], top[column], xerr=errors, color=color,
            ecolor="gray")


def shap_global_figure(shap_df):
    fig, ax = plt.subplots(figsize=(8, 5))
    importance_bars(ax, shap_df, "mean_abs_shap", "steelblue")
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title(
        f"Top-{TOP_N} features — Gradient Boosting (SHAP global importance)"
    )
    fig.tight_layout()
    return fig


def shap_waterfall_figures(gb_expl, dataset, class_labels, n_examples=3):
    """One waterfall per distinct predicted class, for up to n_examples."""
    y_pred = gb_expl["model"].predict(dataset.X_test)
    examples = [
        int(np.flatnonzero(y_pred == c)[0])
        for c in range(len(class_labels))
        if np.any(y_pred == c)
    ][:n_examples]

    figures = []
    for i in examples:
        pred_cls, true_cls = int(y_pred[i]), int(dataset.y_test[i])
        person = simplify_uri(dataset.test_labels_df["person"].iloc[i])
        # waterfall draws into the current figure, so start a fresh one.
        plt.figure()
        shap.plots.waterfall(
            gb_expl["shap_explanation"][i, :, pred_cls],
            max_display=10,
            show=False,
        )
        fig = plt.gcf()
        plt.title(
            f"{person}  |  predicted: {class_labels[pred_cls]}"
            f"  |  true: {class_labels[true_cls]}"
        )
        fig.tight_layout()
        figures.append(fig)
    return figures


def gini_vs_permutation_figure(gini_df, perm_df):
    fig, (left, right) = plt.subplots(1, 2, figsize=(14, 5))
    importance_bars(left, gini_df, "gini_importance", "steelblue",
                    "gini_std")
    left.set_xlabel("Gini importance")
    left.set_title(f"Top-{TOP_N} Gini (MDI) — GB")
    importance_bars(right, perm_df, "perm_importance", "tomato", "perm_std")
    right.set_xlabel("Permutation importance (accuracy drop)")
    right.set_title(f"Top-{TOP_N} Permutation — GB")
    fig.tight_layout()
    return fig


def decision_tree_figure(dt_model, feature_names, class_labels):
    fig, ax = plt.subplots(figsize=(20, 8))
    plot_tree(
        dt_model,
        feature_names=feature_names,
        class_names=class_labels,
        max_depth=3,
        filled=True,
        fontsize=7,
        ax=ax,
    )
    ax.set_title(
        f"Decision Tree (top 3 of {dt_model.get_depth()} levels)"
    )
    fig.tight_layout()
    return fig


def dt_gini_figure(gini_df):
    fig, ax = plt.subplots(figsize=(8, 5))
    importance_bars(ax, gini_df, "gini_importance", "mediumseagreen")
    ax.set_xlabel("Gini importance")
    ax.set_title(f"Top-{TOP_N} features — Decision Tree (Gini importance)")
    fig.tight_layout()
    return fig


def make_figures(figure_dir: Path = DEFAULT_FIGURE_DIR) -> list:
    """Regenerate every figure from the saved checkpoints."""
    figure_dir = Path(figure_dir)
    dataset = load_dataset()
    gb_expl = run_gb_explanations(dataset)
    dt_expl = run_dt_explanations(dataset)
    class_labels = [simplify_uri(c) for c in dataset.label_encoder.classes_]

    figures = {
        "shap_global_importance": shap_global_figure(gb_expl["shap_df"]),
        "gb_gini_vs_perm": gini_vs_permutation_figure(
            gb_expl["gini_df"], gb_expl["perm_df"]
        ),
        "decision_tree": decision_tree_figure(
            dt_expl["model"], dataset.feature_names, class_labels
        ),
        "dt_gini_importance": dt_gini_figure(dt_expl["gini_df"]),
    }
    waterfalls = shap_waterfall_figures(gb_expl, dataset, class_labels)
    for k, fig in enumerate(waterfalls):
        figures[f"shap_waterfall_{k}"] = fig

    return [
        save_figure(fig, figure_dir / f"{name}.png")
        for name, fig in figures.items()
    ]


def quiet_axes(ax) -> None:
    """Recessive axes: no top/right spines, light vertical grid."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(TEXT_MUTED)
    ax.tick_params(colors=TEXT_MUTED)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def graph_top_features_figure(top_features):
    """Top features by mean |SHAP|, coloured by feature family."""
    names, values = zip(*top_features[:TOP_N][::-1])
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.barh(
        names,
        values,
        height=0.7,
        color=[FAMILY_COLORS[family_of(n)] for n in names],
    )
    present = [f for f in FAMILY_COLORS if any(family_of(n) == f
                                               for n in names)]
    ax.legend(
        handles=[Patch(color=FAMILY_COLORS[f], label=f) for f in present],
        title="Feature family",
        loc="lower right",
        frameon=False,
    )
    ax.set_xlabel("Mean |SHAP value| (log-odds, all persons and classes)")
    ax.set_title(f"Top-{TOP_N} graph features — LightGBM")
    quiet_axes(ax)
    fig.tight_layout()
    return fig


def family_importance_figure(results):
    """SHAP share per family next to the accuracy lost without it."""
    share = results["family_shap_share"]
    ablation = results["family_ablation"]["families"]
    families = sorted(share, key=share.get)
    colors = [FAMILY_COLORS[f] for f in families]

    fig, (used, needed) = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    shares = [100 * share[f] for f in families]
    drops = [100 * ablation[f]["drop_without"] for f in families]
    used.barh(families, shares, height=0.6, color=colors)
    needed.barh(families, drops, height=0.6, color=colors)
    needed.axvline(0, color=TEXT_MUTED, linewidth=0.8)

    for ax, values, unit in ((used, shares, "%"), (needed, drops, " pts")):
        for y, value in enumerate(values):
            ax.annotate(
                f"{value:+.1f}{unit}" if unit == " pts" else f"{value:.0f}%",
                (max(value, 0), y),
                xytext=(4, 0),
                textcoords="offset points",
                va="center",
                color=TEXT_MUTED,
                fontsize=9,
            )
        quiet_axes(ax)
    used.set_title("What the model uses")
    used.set_xlabel("Share of SHAP attribution (%)")
    needed.set_title("What the model needs")
    needed.set_xlabel("CV accuracy lost when the family is removed")
    used.set_xlim(0, max(shares) * 1.2)
    needed.set_xlim(min(min(drops), 0) - 1, max(drops) * 1.3)
    fig.tight_layout()
    return fig


def graph_tree_figure(tree, setup):
    kept = list(np.asarray(setup.X.columns)[tree[0].get_support()])
    fig, ax = plt.subplots(figsize=(20, 8))
    plot_tree(
        tree[-1],
        feature_names=kept,
        class_names=setup.class_names,
        filled=True,
        impurity=False,
        fontsize=8,
        ax=ax,
    )
    ax.set_title(
        f"Decision tree on graph features (depth {tree[-1].get_depth()})"
    )
    fig.tight_layout()
    return fig


def graph_waterfall_figure(lgbm, setup, class_name="Knowledge Management"):
    """SHAP waterfall for the first correctly classified test person."""
    c = setup.class_names.index(class_name)
    y_pred = lgbm.predict(setup.X_test.to_numpy())
    i = int(np.flatnonzero((setup.y_test == c) & (y_pred == c))[0])
    explanation = tree_shap(lgbm, setup.X_test.iloc[[i]])

    plt.figure()
    shap.plots.waterfall(explanation[0, :, c], max_display=10, show=False)
    fig = plt.gcf()
    person = simplify_uri(setup.X_test.index[i])
    plt.title(f"{person}  |  predicted and true: {class_name}")
    fig.tight_layout()
    return fig


KIND_COLORS = {"interpretable": "#2a78d6", "black box": "#eb6834"}


def load_result(name: str) -> dict:
    return json.loads((RESULTS_DIR / name).read_text(encoding="utf-8"))


def model_comparison_figure():
    """CV accuracy of every model on AIFB and MUTAG, as a dot plot."""
    experiments = load_result("experiments.json")
    benchmarks = load_result("benchmarks.json")
    aifb = experiments["experiments"]["graph"]
    mutag = benchmarks["mutag"]["models"]
    panels = {
        "AIFB: 4 research groups": (
            [aifb, load_result("gnn_aifb.json")["variants"]["strict"]],
            experiments["majority_baseline_test_accuracy"],
        ),
        "MUTAG: mutagenic or not": (
            [mutag, load_result("gnn_mutag.json")["variants"]["standard"]],
            benchmarks["mutag"]["majority_baseline_test_accuracy"],
        ),
    }
    rows = [
        ("Decision tree", "decision_tree", "interpretable"),
        ("Logistic regression", "logreg", "interpretable"),
        ("LightGBM", "lightgbm", "black box"),
        ("R-GCN (graph neural net)", None, "black box"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4), sharey=True)
    for ax, (title, ((models, gnn_result), baseline)) in zip(
        axes, panels.items()
    ):
        for y, (label, key, kind) in enumerate(rows[::-1]):
            result = gnn_result if key is None else models[key]
            mean = result["cv_accuracy_mean"] * 100
            ax.errorbar(
                mean, y, xerr=result["cv_accuracy_std"] * 100, fmt="o",
                markersize=9, color=KIND_COLORS[kind], ecolor=GRID,
                elinewidth=3, capsize=0,
            )
            ax.annotate(f"{mean:.0f}%", (mean, y), xytext=(0, 9),
                        textcoords="offset points", ha="center",
                        color=TEXT_MUTED, fontsize=9)
        ax.axvline(baseline * 100, color=TEXT_MUTED, linestyle="--",
                   linewidth=1)
        ax.annotate("majority class", (baseline * 100, -0.6),
                    xytext=(4, 0), textcoords="offset points",
                    color=TEXT_MUTED, fontsize=8)
        ax.set_yticks(range(len(rows)), [r[0] for r in rows[::-1]])
        ax.set_ylim(-0.8, len(rows) - 0.4)
        ax.set_xlim(35, 100)
        ax.set_title(title)
        ax.set_xlabel("Cross-validation accuracy (%), mean ± std")
        quiet_axes(ax)
    fig.legend(
        handles=[Patch(color=c, label=k) for k, c in KIND_COLORS.items()],
        loc="lower center", ncol=2, frameon=False,
        bbox_to_anchor=(0.5, -0.06),
    )
    fig.tight_layout()
    return fig


def make_graph_figures(figure_dir: Path = DEFAULT_FIGURE_DIR) -> list:
    """Figures for the graph-feature models (run src.explain_graph first)."""
    results = json.loads(
        (RESULTS_DIR / "graph_explanations.json").read_text(encoding="utf-8")
    )
    setup = load_setup()
    figures = {
        "graph_shap_top_features": graph_top_features_figure(
            results["top_features_mean_abs_shap"]
        ),
        "family_importance": family_importance_figure(results),
        "graph_decision_tree": graph_tree_figure(
            fit_model(setup, "decision_tree"), setup
        ),
        "graph_shap_waterfall": graph_waterfall_figure(
            fit_model(setup, "lightgbm"), setup
        ),
        "model_comparison": model_comparison_figure(),
    }
    return [
        save_figure(fig, Path(figure_dir) / f"{name}.png")
        for name, fig in figures.items()
    ]


if __name__ == "__main__":
    for path in make_figures() + make_graph_figures():
        print(f"Saved {path.name}")
