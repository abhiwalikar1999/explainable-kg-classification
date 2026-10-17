"""The generic pipeline on several RDF node-classification benchmarks.

Each benchmark is one config entry: how to load its graph (with the label
relations removed), where its labels are, and which nodes must never become
feature values. Features come from `src.generic_features`, so nothing here
is specific to one ontology.
"""
import gzip
import json
import tarfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pandas as pd
from rdflib import Graph, URIRef
from sklearn.preprocessing import LabelEncoder

from src.dataset import (
    DEFAULT_TEST_PATH,
    DEFAULT_TRAIN_PATH,
    PROJECT_ROOT,
    label_nodes,
    load_graph,
)
from src.evaluate import majority_baseline
from src.experiments import RESULTS_DIR, evaluate, model_specs, tree_rules
from src.generic_features import build_generic_features

DOWNLOAD_DIR = PROJECT_ROOT / "data/downloads"
MUTAG_DIR = DOWNLOAD_DIR / "mutag"
MUTAG_URL = "https://data.dgl.ai/dataset/mutag.tgz"
MUTAG_FILES = (
    "mutag_stripped.nt.gz",
    "trainingSet.tsv",
    "testSet.tsv",
    "completeDataset.tsv",
)
MUTAG_LABEL = URIRef("http://dl-learner.org/carcinogenesis#isMutagenic")


@dataclass(frozen=True)
class Benchmark:
    name: str
    load_graph: Callable[[], Graph]
    train_path: Path
    test_path: Path
    entity_column: str
    label_column: str
    blocked_nodes: Callable[[Graph], set]
    class_names: dict = None  # raw label -> readable name


def download_mutag(target: Path = MUTAG_DIR) -> Path:
    """Fetch the R-GCN benchmark copy of MUTAG (data files only)."""
    if all((target / name).exists() for name in MUTAG_FILES):
        return target
    target.mkdir(parents=True, exist_ok=True)
    archive = target / "mutag.tgz"
    urllib.request.urlretrieve(MUTAG_URL, archive)
    with tarfile.open(archive) as tar:
        members = [m for m in tar.getmembers() if m.name in MUTAG_FILES]
        tar.extractall(target, members=members, filter="data")
    return target


def load_mutag_graph() -> Graph:
    directory = download_mutag()
    g = Graph()
    with gzip.open(directory / "mutag_stripped.nt.gz", "rb") as source:
        g.parse(source, format="nt")
    g.remove((None, MUTAG_LABEL, None))  # already stripped; kept explicit
    return g


BENCHMARKS = {
    "aifb": Benchmark(
        name="aifb",
        load_graph=load_graph,
        train_path=DEFAULT_TRAIN_PATH,
        test_path=DEFAULT_TEST_PATH,
        entity_column="person",
        label_column="label_affiliation",
        blocked_nodes=label_nodes,
    ),
    "mutag": Benchmark(
        name="mutag",
        load_graph=load_mutag_graph,
        train_path=MUTAG_DIR / "trainingSet.tsv",
        test_path=MUTAG_DIR / "testSet.tsv",
        entity_column="bond",
        label_column="label_mutagenic",
        blocked_nodes=lambda g: set(),
        class_names={0.0: "not mutagenic", 1.0: "mutagenic"},
    ),
}


def load_benchmark(benchmark: Benchmark) -> tuple:
    """(graph, entities train-then-test, y, n_train, class names)."""
    g = benchmark.load_graph()
    train = pd.read_csv(benchmark.train_path, sep="\t")
    test = pd.read_csv(benchmark.test_path, sep="\t")
    encoder = LabelEncoder().fit(train[benchmark.label_column])
    y_train = encoder.transform(train[benchmark.label_column])
    y_test = encoder.transform(test[benchmark.label_column])
    entities = list(train[benchmark.entity_column]) + list(
        test[benchmark.entity_column]
    )
    y = pd.Series(list(y_train) + list(y_test)).to_numpy()
    names = benchmark.class_names or {}
    classes = [
        names.get(c, str(c).rsplit("/", 1)[-1]) for c in encoder.classes_
    ]
    return g, entities, y, len(train), classes


def run_benchmark(benchmark: Benchmark, results_dir: Path, n_repeats=3):
    g, entities, y, n_train, classes = load_benchmark(benchmark)
    X = build_generic_features(g, entities, benchmark.blocked_nodes(g))
    result = {
        "n_entities": len(entities),
        "n_classes": len(classes),
        "n_generic_features": X.shape[1],
        "majority_baseline_test_accuracy": majority_baseline(
            y[:n_train], y[n_train:]
        ),
        "models": {},
    }
    for model_name, (pipeline, grid) in model_specs(min_support=2).items():
        metrics, fitted = evaluate(
            pipeline, grid, X.to_numpy(), y, n_train, n_repeats
        )
        result["models"][model_name] = metrics
        if model_name == "decision_tree":
            (results_dir / f"tree_rules_generic_{benchmark.name}.txt"
             ).write_text(
                tree_rules(fitted, X.columns, classes), encoding="utf-8"
            )
    return result


def run_benchmarks(results_dir: Path = RESULTS_DIR) -> dict:
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    results = {
        name: run_benchmark(benchmark, results_dir)
        for name, benchmark in BENCHMARKS.items()
    }
    (results_dir / "benchmarks.json").write_text(json.dumps(results, indent=2))
    return results


if __name__ == "__main__":
    for name, result in run_benchmarks().items():
        print(f"{name}: {result['n_entities']} entities, "
              f"{result['n_generic_features']} generic features, "
              f"majority {result['majority_baseline_test_accuracy']:.1%}")
        for model, m in result["models"].items():
            print(f"  {model:14s} test {m['test_accuracy']:.1%}  "
                  f"nested CV {m['cv_accuracy_mean']:.1%} "
                  f"± {m['cv_accuracy_std']:.1%}")
