import pytest

from src import dataset as D
from src.features import FEATURE_SETS, build_features


@pytest.fixture(scope="session")
def graph():
    return D.load_graph()


@pytest.fixture(scope="session")
def raw_graph():
    return D.load_graph(strip_leakage=False)


@pytest.fixture(scope="session")
def data():
    return D.load_dataset()


@pytest.fixture(scope="session")
def labelled_persons():
    train, test = D.load_labels()
    return list(train["person"]) + list(test["person"])


@pytest.fixture(scope="session")
def graph_features(graph, labelled_persons):
    return build_features(
        graph, labelled_persons, FEATURE_SETS["graph+group_paths"]
    )
