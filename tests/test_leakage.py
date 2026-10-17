"""The label must not be reachable from a person through a single edge."""
from collections import Counter

from rdflib import URIRef

from src import dataset as D

PERSON_NS = "/Personen/viewPersonOWL/"
GROUP_NS = "/Forschungsgruppen/viewForschungsgruppeOWL/"


def _links_person_and_group(s, o) -> bool:
    # Matches on URI namespaces rather than rdf:type, so this check does not
    # share any logic with dataset.find_label_edges.
    s, o = str(s), str(o)
    return (PERSON_NS in s and GROUP_NS in o) or (
        GROUP_NS in s and PERSON_NS in o
    )


def test_no_direct_person_group_edges(graph):
    leaks = [t for t in graph if _links_person_and_group(t[0], t[2])]
    assert leaks == []


def test_label_predicates_removed(graph):
    for predicate in D.LEAKAGE_PREDICATES:
        triples = graph.triples((None, URIRef(predicate), None))
        assert next(triples, None) is None


def test_project_membership_kept(graph):
    # swrc:member is shared by projects and groups; only groups leak.
    member = URIRef(D.SWRC + "member")
    subjects = {s for s, _, _ in graph.triples((None, member, None))}
    assert subjects
    assert not subjects & D.label_nodes(graph)


def test_detector_finds_all_leaks_in_raw_graph(raw_graph):
    found = Counter(
        D.simplify_uri(str(p)) for _, p, _ in D.find_label_edges(raw_graph)
    )
    expected = Counter(
        D.simplify_uri(str(p))
        for s, p, o in raw_graph
        if _links_person_and_group(s, o)
    )
    assert found == expected
    assert set(found) == {"affiliation", "employs", "member", "head"}
