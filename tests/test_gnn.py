"""R-GCN graph construction. Skipped when torch isn't installed (CI)."""
import pytest
from rdflib import Graph, Namespace

pytest.importorskip("torch_geometric")

from src import gnn  # noqa: E402
from src.dataset import label_nodes, person_nodes  # noqa: E402

EX = Namespace("http://example.org/")


def test_every_triple_becomes_an_edge_and_its_inverse():
    g = Graph()
    g.add((EX.a, EX.knows, EX.b))
    g.add((EX.b, EX.likes, EX.c))
    edge_index, edge_type, n_nodes, n_rel, entity_index = gnn.to_tensors(
        g, [str(EX.a)]
    )
    assert n_nodes == 3 and n_rel == 4
    assert edge_index.shape == (2, 4)
    assert sorted(edge_type.tolist()) == [0, 1, 2, 3]
    forward = set(zip(edge_index[0][:2].tolist(), edge_index[1][:2].tolist()))
    backward = set(zip(edge_index[1][2:].tolist(), edge_index[0][2:].tolist()))
    assert forward == backward


def direct_person_group_edges(g: Graph) -> int:
    persons, groups = person_nodes(g), label_nodes(g)
    return sum(
        1
        for s, _, o in g
        if (s in persons and o in groups) or (s in groups and o in persons)
    )


def test_standard_variant_keeps_group_member_edges():
    assert direct_person_group_edges(gnn.variant_graph("standard")) == 79


def test_strict_variant_has_no_direct_label_edges():
    assert direct_person_group_edges(gnn.variant_graph("strict")) == 0


def test_no_group_nodes_variant_never_touches_a_group():
    g = gnn.variant_graph("no_group_nodes")
    groups = label_nodes(gnn.variant_graph("strict"))
    assert not any(s in groups or o in groups for s, _, o in g)
