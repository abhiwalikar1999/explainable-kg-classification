"""RDF loading, leakage-safe predicate stripping, and tabular conversion."""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from rdflib import RDF, RDFS, Graph, URIRef
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import LabelEncoder

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SWRC = "http://swrc.ontoware.org/ontology#"

# The label predicate and its inverse; removed everywhere (standard AIFB
# benchmark preprocessing).
LEAKAGE_PREDICATES = (
    SWRC + "affiliation",
    SWRC + "employs",
)

PERSON_CLASS = URIRef(SWRC + "Person")
LABEL_CLASS = URIRef(SWRC + "ResearchGroup")

# Base for the file's relative URIs. Without it rdflib resolves them against
# the local file path, which differs between machines.
BASE_URI = "http://www.aifb.uni-karlsruhe.de/"

DEFAULT_RDF_PATH = PROJECT_ROOT / "data/aifb-hetero/aifbfixed_complete.n3"
DEFAULT_TRAIN_PATH = PROJECT_ROOT / "data/aifb-hetero/trainingSet.tsv"
DEFAULT_TEST_PATH = PROJECT_ROOT / "data/aifb-hetero/testSet.tsv"

LABEL_COLUMN = "label_affiliation"
PERSON_COLUMN = "person"


def person_nodes(g: Graph) -> set:
    """Nodes typed swrc:Person or any of its subclasses."""
    classes = set(g.transitive_subjects(RDFS.subClassOf, PERSON_CLASS))
    return {s for c in classes for s in g.subjects(RDF.type, c)}


def label_nodes(g: Graph) -> set:
    """Nodes typed swrc:ResearchGroup, i.e. the label values."""
    return set(g.subjects(RDF.type, LABEL_CLASS))


def find_label_edges(g: Graph) -> list:
    """Triples that link a person directly to a research group."""
    persons = person_nodes(g)
    groups = label_nodes(g)
    return [
        (s, p, o)
        for s, p, o in g
        if (s in persons and o in groups) or (s in groups and o in persons)
    ]


def remove_leakage(g: Graph) -> Graph:
    """Remove every triple that reveals a person's research group.

    Besides the label predicate and its inverse, the groups also point at
    their people via swrc:member and swrc:head. Those predicates are not
    dropped wholesale because projects use swrc:member too.
    """
    for predicate in LEAKAGE_PREDICATES:
        g.remove((None, URIRef(predicate), None))
    for triple in find_label_edges(g):
        g.remove(triple)
    return g


def load_graph(
    rdf_path: Path = DEFAULT_RDF_PATH, strip_leakage: bool = True
) -> Graph:
    """Parse the AIFB RDF graph."""
    g = Graph()
    g.parse(rdf_path, format="n3", publicID=BASE_URI)
    if strip_leakage:
        remove_leakage(g)
    return g


def simplify_uri(uri: str) -> str:
    """Extract the human-readable fragment from a URI."""
    if "#" in uri:
        return uri.split("#")[-1]
    return uri.rstrip("/").split("/")[-1]


def graph_to_dataframe(g: Graph) -> pd.DataFrame:
    """Pivot RDF triples into a subject-by-predicate table."""
    ordered_triples = sorted(
        g, key=lambda t: (str(t[0]), str(t[1]), str(t[2]))
    )

    table = {}
    for subject, predicate, obj in ordered_triples:
        subject_key = str(subject)
        if subject_key not in table:
            table[subject_key] = {}
        table[subject_key][predicate] = str(obj)

    df = pd.DataFrame.from_dict(table, orient="index")
    df = df.fillna("")
    df.index = df.index.astype(str)
    return df


def _first_non_empty(values) -> str:
    """First non-empty string in values, or ""."""
    for value in values:
        if value != "":
            return value
    return ""


def _coalesce_duplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Merge same-named columns, keeping the first non-empty value per row."""
    if not df.columns.duplicated().any():
        return df

    merged_columns = {}
    for col_name in df.columns.unique():
        matching = df.loc[:, df.columns == col_name]
        if matching.shape[1] == 1:
            merged_columns[col_name] = matching.iloc[:, 0]
            continue

        merged_values = []
        for _, row in matching.iterrows():
            merged_values.append(_first_non_empty(row.tolist()))
        merged_columns[col_name] = pd.Series(
            merged_values, index=matching.index
        )

    return pd.DataFrame(merged_columns, index=df.index)


def simplify_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Simplify column names and cell values to URI fragments."""
    df = df.copy()
    df.columns = df.columns.astype(str).str.split("#").str[-1]
    df.columns = df.columns.str.split("/").str[-1]

    for col in df.columns:
        df[col] = df[col].map(simplify_uri)

    return _coalesce_duplicate_columns(df)


def build_tabular_features(g: Graph) -> pd.DataFrame:
    """RDF graph -> simplified subject x predicate table."""
    df = graph_to_dataframe(g)
    return simplify_dataframe(df)


def load_labels(
    train_path: Path = DEFAULT_TRAIN_PATH,
    test_path: Path = DEFAULT_TEST_PATH,
) -> tuple:
    """Load the train/test person -> research-group label TSVs."""
    train_labels_df = pd.read_csv(train_path, sep="\t")
    test_labels_df = pd.read_csv(test_path, sep="\t")
    return train_labels_df, test_labels_df


@dataclass
class Dataset:
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    feature_names: list
    label_encoder: LabelEncoder
    train_labels_df: pd.DataFrame
    test_labels_df: pd.DataFrame
    # All 176 labelled persons (train rows, then test rows) before variance
    # filtering, so cross-validation can refit the filter inside each fold.
    X_labeled_raw: pd.DataFrame
    y_labeled: np.ndarray
    # Persons in the graph without a label, filtered like X_train. Used to
    # fit global surrogates on more than the training set.
    X_unlabeled: np.ndarray


def load_dataset(
    rdf_path: Path = DEFAULT_RDF_PATH,
    train_path: Path = DEFAULT_TRAIN_PATH,
    test_path: Path = DEFAULT_TEST_PATH,
    variance_threshold: float = 0.001,
) -> Dataset:
    """Full pipeline: RDF -> tabular -> one-hot -> filtered arrays."""
    g = load_graph(rdf_path)
    df = build_tabular_features(g)

    train_labels_df, test_labels_df = load_labels(train_path, test_path)

    df_encoded = pd.get_dummies(df)
    df_encoded.index = df_encoded.index.astype(str)

    train_persons = train_labels_df[PERSON_COLUMN]
    test_persons = test_labels_df[PERSON_COLUMN]
    labeled = set(train_persons) | set(test_persons)
    unlabeled_persons = sorted(
        str(p) for p in person_nodes(g) if str(p) not in labeled
    )
    x_train_raw = df_encoded.loc[train_persons]
    x_test_raw = df_encoded.loc[test_persons]

    selector = VarianceThreshold(threshold=variance_threshold)
    x_train = selector.fit_transform(x_train_raw)
    x_test = selector.transform(x_test_raw)
    x_unlabeled = selector.transform(df_encoded.loc[unlabeled_persons])

    kept_columns = selector.get_support(indices=True)
    feature_names = [df_encoded.columns[i] for i in kept_columns]

    encoder = LabelEncoder()
    y_train = encoder.fit_transform(train_labels_df[LABEL_COLUMN])
    y_test = encoder.transform(test_labels_df[LABEL_COLUMN])

    return Dataset(
        X_train=x_train,
        X_test=x_test,
        y_train=y_train,
        y_test=y_test,
        feature_names=feature_names,
        label_encoder=encoder,
        train_labels_df=train_labels_df,
        test_labels_df=test_labels_df,
        X_labeled_raw=pd.concat([x_train_raw, x_test_raw]),
        y_labeled=np.concatenate([y_train, y_test]),
        X_unlabeled=x_unlabeled,
    )


if __name__ == "__main__":
    dataset = load_dataset()
    print(f"Train features: {dataset.X_train.shape}")
    print(f"Test features:  {dataset.X_test.shape}")
    print(f"Classes: {list(dataset.label_encoder.classes_)}")
