"""R-GCN baseline (PyTorch Geometric) on AIFB and MUTAG.

Featureless 2-layer R-GCN as in Schlichtkrull et al. (2018): every RDF node,
literals included, gets a learned embedding, and every triple becomes a
typed edge plus its inverse. Trained with the paper's per-dataset settings,
without tuning.

On AIFB the graph is built in three variants to measure what the GNN reads:
    standard        only affiliation / employs removed (as in the literature)
    strict          every direct person<->group edge removed (this project)
    no_group_nodes  every edge touching a research group removed, so no path
                    reaches the label nodes (comparable to `graph` features)

Needs `pip install -r requirements-gnn.txt`.
Usage: python -m src.gnn aifb|mutag
"""
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from rdflib import Graph, URIRef
from sklearn.model_selection import RepeatedStratifiedKFold
from torch_geometric.nn import RGCNConv

from src.benchmarks import BENCHMARKS, load_benchmark
from src.dataset import (
    LEAKAGE_PREDICATES,
    load_dataset,
    load_graph,
    remove_leakage,
)
from src.dataset import label_nodes as research_groups
from src.evaluate import wilson_interval
from src.experiments import RESULTS_DIR

VARIANTS = ("standard", "strict", "no_group_nodes")
HIDDEN = 16
EPOCHS = 50
LEARNING_RATE = 0.01
N_SEEDS = 10


@dataclass(frozen=True)
class Settings:
    """Per-dataset R-GCN settings from the paper (Table 2)."""
    num_bases: int = None  # None = no basis decomposition
    weight_decay: float = 0.0


SETTINGS = {
    "aifb": Settings(num_bases=None, weight_decay=0.0),
    "mutag": Settings(num_bases=30, weight_decay=5e-4),
}


def variant_graph(variant: str) -> Graph:
    """The AIFB graph with the given amount of label leakage removed."""
    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant: {variant}")
    g = load_graph(strip_leakage=False)
    groups = research_groups(g)
    if variant == "standard":
        for predicate in LEAKAGE_PREDICATES:
            g.remove((None, URIRef(predicate), None))
        return g
    remove_leakage(g)
    if variant == "no_group_nodes":
        for triple in [t for t in g if t[0] in groups or t[2] in groups]:
            g.remove(triple)
    return g


def to_tensors(g: Graph, entities) -> tuple:
    """(edge_index, edge_type, num_nodes, num_relations, entity_index).

    Relation r's inverse is r + R; triples are sorted so ids are stable.
    """
    triples = sorted(g, key=lambda t: (str(t[0]), str(t[1]), str(t[2])))
    nodes, relations = {}, {}
    for s, p, o in triples:
        nodes.setdefault(s, len(nodes))
        nodes.setdefault(o, len(nodes))
        relations.setdefault(p, len(relations))

    src = [nodes[s] for s, _, _ in triples]
    dst = [nodes[o] for _, _, o in triples]
    rel = [relations[p] for _, p, _ in triples]
    n_rel = len(relations)
    edge_index = torch.tensor([src + dst, dst + src], dtype=torch.long)
    edge_type = torch.tensor(rel + [r + n_rel for r in rel], dtype=torch.long)
    entity_index = torch.tensor(
        [nodes[URIRef(e)] for e in entities], dtype=torch.long
    )
    return edge_index, edge_type, len(nodes), 2 * n_rel, entity_index


class RGCN(torch.nn.Module):
    def __init__(self, num_nodes, num_relations, num_classes, num_bases):
        super().__init__()
        self.conv1 = RGCNConv(num_nodes, HIDDEN, num_relations,
                              num_bases=num_bases)
        self.conv2 = RGCNConv(HIDDEN, num_classes, num_relations,
                              num_bases=num_bases)

    def forward(self, edge_index, edge_type):
        x = F.relu(self.conv1(None, edge_index, edge_type))
        return self.conv2(x, edge_index, edge_type)


def train_and_predict(
    graph_tensors, y, train_idx, test_idx, seed, settings: Settings
) -> np.ndarray:
    """Train on train_idx entities, return predictions for test_idx."""
    edge_index, edge_type, n_nodes, n_rel, entity_index = graph_tensors
    torch.manual_seed(seed)
    model = RGCN(n_nodes, n_rel, int(y.max()) + 1, settings.num_bases)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=settings.weight_decay,
    )
    targets = torch.as_tensor(y, dtype=torch.long)
    train_nodes = entity_index[train_idx]

    model.train()
    for _ in range(EPOCHS):
        optimizer.zero_grad()
        out = model(edge_index, edge_type)
        loss = F.cross_entropy(out[train_nodes], targets[train_idx])
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        out = model(edge_index, edge_type)
    return out[entity_index[test_idx]].argmax(dim=1).numpy()


def evaluate_graph(
    g: Graph, entities, y, n_train: int, settings: Settings, n_repeats=1
) -> dict:
    """Standard split over N_SEEDS seeds, plus stratified 5-fold CV.

    One CV repeat by default: a CPU training run takes ~35 s on AIFB.
    """
    graph_tensors = to_tensors(g, entities)
    train_idx = np.arange(n_train)
    test_idx = np.arange(n_train, len(y))

    def accuracy(fit, held_out, seed) -> float:
        predictions = train_and_predict(
            graph_tensors, y, fit, held_out, seed, settings
        )
        return float(np.mean(predictions == y[held_out]))

    test_scores = [
        accuracy(train_idx, test_idx, seed) for seed in range(N_SEEDS)
    ]
    cv = RepeatedStratifiedKFold(
        n_splits=5, n_repeats=n_repeats, random_state=0
    )
    cv_scores = [
        accuracy(fit, held_out, seed=fold)
        for fold, (fit, held_out) in enumerate(cv.split(entities, y))
    ]
    median_correct = int(round(np.median(test_scores) * len(test_idx)))
    return {
        "num_edges": int(graph_tensors[0].shape[1] // 2),
        "num_nodes": graph_tensors[2],
        "test_accuracy_mean": float(np.mean(test_scores)),
        "test_accuracy_std": float(np.std(test_scores)),
        "test_accuracy_per_seed": test_scores,
        "test_accuracy_median_wilson95": wilson_interval(
            median_correct, len(test_idx)
        ),
        "cv_accuracy_mean": float(np.mean(cv_scores)),
        "cv_accuracy_std": float(np.std(cv_scores)),
        "cv_folds": len(cv_scores),
    }


def run_gnn(dataset_name: str, results_dir: Path = RESULTS_DIR) -> dict:
    settings = SETTINGS[dataset_name]
    if dataset_name == "aifb":
        dataset = load_dataset()
        persons = list(dataset.X_labeled_raw.index)
        n_train = len(dataset.y_train)
        variants = {
            v: evaluate_graph(
                variant_graph(v), persons, dataset.y_labeled, n_train,
                settings,
            )
            for v in VARIANTS
        }
    else:
        g, entities, y, n_train, _ = load_benchmark(BENCHMARKS[dataset_name])
        variants = {
            "standard": evaluate_graph(g, entities, y, n_train, settings)
        }

    results = {
        "model": f"R-GCN, 2 layers, {HIDDEN} hidden, Adam {LEARNING_RATE}, "
                 f"{EPOCHS} epochs, featureless, {settings}",
        "variants": variants,
    }
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / f"gnn_{dataset_name}.json").write_text(
        json.dumps(results, indent=2)
    )
    return results


if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "aifb"
    for variant, r in run_gnn(name)["variants"].items():
        print(
            f"{name} {variant:15s} test {r['test_accuracy_mean']:.1%} "
            f"± {r['test_accuracy_std']:.1%} | CV {r['cv_accuracy_mean']:.1%}"
            f" ± {r['cv_accuracy_std']:.1%} | edges {r['num_edges']}"
        )
