import pytest
from rdflib import Graph, Literal, Namespace

from src.generic_features import build_generic_features, entity_features

EX = Namespace("http://example.org/")


@pytest.fixture
def molecule_graph():
    """d1 has two atoms (C, N) with charges, one assay, and a label node."""
    g = Graph()
    g.add((EX.d1, EX.ames, Literal("true")))
    for atom, element, charge in ((EX.a1, EX.C, "-0.1"), (EX.a2, EX.N, "0.3")):
        g.add((EX.d1, EX.hasAtom, atom))
        g.add((atom, EX.type, element))
        g.add((atom, EX.charge, Literal(charge)))
    g.add((EX.d1, EX.group, EX.LabelNode))
    g.add((EX.d1, EX.other, EX.thing))
    g.add((EX.thing, EX.backTo, EX.d1))
    return g


def test_one_and_two_hop_features(molecule_graph):
    features = entity_features(molecule_graph, EX.d1, blocked={EX.LabelNode})
    assert features["ames=true"] == 1
    assert features["hasAtom:count"] == 2
    assert features["hasAtom/type=C"] == 1
    assert features["hasAtom/type=N"] == 1
    assert features["hasAtom/charge:mean"] == pytest.approx(0.1)


def test_blocked_nodes_never_become_features(molecule_graph):
    features = entity_features(molecule_graph, EX.d1, blocked={EX.LabelNode})
    assert not any("LabelNode" in name for name in features)
    assert "group:count" not in features


def test_walk_does_not_return_to_the_entity(molecule_graph):
    features = entity_features(molecule_graph, EX.d1, blocked=set())
    assert not any(name.startswith("other/backTo") for name in features)


def test_table_rows_follow_entity_order(molecule_graph):
    table = build_generic_features(
        molecule_graph, [str(EX.thing), str(EX.d1)]
    )
    assert list(table.index) == [str(EX.thing), str(EX.d1)]
    assert table.loc[str(EX.thing), "hasAtom:count"] == 0
