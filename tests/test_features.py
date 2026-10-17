import numpy as np
import pytest
from rdflib import URIRef

from src import dataset as D
from src import features as F

EMPTY_PHONE_PERSON = (
    "http://www.aifb.uni-karlsruhe.de/Personen/viewPersonOWL/id1909instance"
)


def test_feature_names_carry_their_family(graph_features):
    families = {name.split(":")[0] for name in graph_features.columns}
    assert families == set(F.FAMILIES)


def test_only_group_paths_reach_research_groups(graph, labelled_persons):
    # Every family except group_path is built from these neighbourhoods,
    # so none of them may contain a research-group node.
    groups = D.label_nodes(graph)
    for person in map(URIRef, labelled_persons):
        publications = F.publications_of(graph, person)
        neighbours = (
            publications
            | F.projects_of(graph, person)
            | F.topics_worked_on(graph, person)
            | F.coauthors_of(graph, person, publications)
        )
        assert not neighbours & groups, person


def test_nobody_is_their_own_coauthor(graph, graph_features):
    for person in graph_features.index:
        column = f"coauthor:{F.node_name(graph, URIRef(person))}"
        if column in graph_features:
            assert graph_features.loc[person, column] == 0


def test_empty_phone_does_not_count_as_phone(graph_features):
    assert graph_features.loc[EMPTY_PHONE_PERSON, "profile:has_phone"] == 0


@pytest.mark.parametrize(
    "raw",
    ["+49 (721) 608 4548", "+49 (0) 721 608 4548", "0721 608-4548"],
)
def test_phone_numbers_share_one_format(raw):
    assert F.phone_digits(raw) == "07216084548"


def test_unknown_family_is_rejected(graph):
    with pytest.raises(ValueError):
        F.build_features(graph, [], ["profile", "salary"])


def test_min_support_filter_drops_single_person_values():
    X = np.array([[1, 1, 0], [0, 1, 0], [0, 1, 3]])
    kept = F.MinSupportFilter(min_support=2).fit(X)
    assert kept.get_support().tolist() == [False, True, False]
    assert kept.transform(X).shape == (3, 1)
