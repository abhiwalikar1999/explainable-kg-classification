"""Readable, identifier-free graph features for persons.

Every feature is named "<family>:<item>", e.g. "topic:knowledge systems" or
"project:OntoLoRe", so explanations read in the graph's own terms.

Families:
    profile     role, which contact fields are filled in, activity counts
    contact     shared fax numbers and phone-number blocks
    project     projects the person works on
    topic       research topics of their publications and projects (counts)
    coauthor    people they published with
    group_path  papers / projects linked to each research group. These are
                two-hop paths to the label nodes, i.e. label proxies, so
                they form a separate, optional family.
"""
import re
from collections import Counter

import numpy as np
import pandas as pd
from rdflib import RDF, Graph, URIRef
from sklearn.base import BaseEstimator, TransformerMixin

from src.dataset import SWRC, label_nodes, simplify_uri

NAME = URIRef(SWRC + "name")
AUTHOR = URIRef(SWRC + "author")
PUBLICATION = URIRef(SWRC + "publication")
EDITOR = URIRef(SWRC + "editor")
WORKS_AT_PROJECT = URIRef(SWRC + "worksAtProject")
IS_WORKED_ON_BY = URIRef(SWRC + "isWorkedOnBy")
MEMBER = URIRef(SWRC + "member")
IS_ABOUT = URIRef(SWRC + "isAbout")
PUBLISHES = URIRef(SWRC + "publishes")
CARRIES_OUT = URIRef(SWRC + "carriesOut")
PROJECT_CLASS = URIRef(SWRC + "Project")
TOPIC_CLASS = URIRef(SWRC + "ResearchTopic")

CONTACT_FIELDS = ("phone", "fax", "homepage", "photo")
MISSING_VALUES = {"", "-", "None"}

FAMILIES = ("profile", "contact", "project", "topic", "coauthor", "group_path")
FEATURE_SETS = {
    "graph": ("profile", "contact", "project", "topic", "coauthor"),
    "graph+group_paths": FAMILIES,
}


def clean_text(value) -> str:
    """Literal as text, repairing UTF-8 that was decoded as Latin-1."""
    text = str(value).strip()
    try:
        return text.encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return text


def node_name(g: Graph, node) -> str:
    name = g.value(node, NAME)
    return clean_text(name) if name is not None else simplify_uri(str(node))


def phone_digits(value) -> str:
    """Phone number in national format: '+49 (721) 608-4548' -> '07216084548'.

    The graph mixes '+49 (721) …', '+49 (0) 721 …' and '0721 …'.
    """
    digits = re.sub(r"\D", "", str(value).replace("(0)", ""))
    return "0" + digits[2:] if digits.startswith("49") else digits


def is_a(g: Graph, node, cls) -> bool:
    return (node, RDF.type, cls) in g


def publications_of(g: Graph, person) -> set:
    return set(g.objects(person, PUBLICATION)) | set(
        g.subjects(AUTHOR, person)
    )


def projects_of(g: Graph, person) -> set:
    candidates = (
        set(g.objects(person, WORKS_AT_PROJECT))
        | set(g.subjects(IS_WORKED_ON_BY, person))
        | set(g.subjects(MEMBER, person))
    )
    return {node for node in candidates if is_a(g, node, PROJECT_CLASS)}


def topics_worked_on(g: Graph, person) -> set:
    """Research topics that name the person via swrc:isWorkedOnBy."""
    return {
        node
        for node in g.subjects(IS_WORKED_ON_BY, person)
        if is_a(g, node, TOPIC_CLASS)
    }


def coauthors_of(g: Graph, person, publications: set) -> set:
    coauthors = set()
    for pub in publications:
        coauthors |= set(g.objects(pub, AUTHOR))
        coauthors |= set(g.subjects(PUBLICATION, pub))
    return coauthors - {person}


def profile_features(g, person, publications, projects, coauthors) -> dict:
    features = {}
    for role in g.objects(person, RDF.type):
        role_name = simplify_uri(str(role))
        if role_name != "Person":
            features[f"profile:role={role_name}"] = 1
    for field in CONTACT_FIELDS:
        values = {
            str(v).strip() for v in g.objects(person, URIRef(SWRC + field))
        }
        features[f"profile:has_{field}"] = int(bool(values - MISSING_VALUES))
    features["profile:n_publications"] = len(publications)
    features["profile:n_projects"] = len(projects)
    features["profile:n_edited"] = len(set(g.subjects(EDITOR, person)))
    features["profile:n_coauthors"] = len(coauthors)
    return features


def contact_features(g, person) -> dict:
    features = {}
    for value in g.objects(person, URIRef(SWRC + "fax")):
        digits = phone_digits(value)
        if digits:
            features[f"contact:fax={digits}"] = 1
    for value in g.objects(person, URIRef(SWRC + "phone")):
        digits = phone_digits(value)
        if len(digits) > 2:
            features[f"contact:phone_block={digits[:-2]}xx"] = 1
    return features


def topic_features(g, items, direct_topics) -> dict:
    """Topic counts: topics of the items plus topics worked on directly."""
    topics = [t for item in items for t in g.objects(item, IS_ABOUT)]
    topics += list(direct_topics)
    return dict(Counter(f"topic:{node_name(g, t)}" for t in topics))


def group_path_features(publications, projects, group_outputs) -> dict:
    features = {}
    for name, (papers, group_projects) in group_outputs.items():
        features[f"group_path:papers_by={name}"] = len(publications & papers)
        features[f"group_path:projects_by={name}"] = len(
            projects & group_projects
        )
    return features


def person_features(g, person, families, group_outputs) -> dict:
    """All requested feature families for one person."""
    publications = publications_of(g, person)
    projects = projects_of(g, person)
    coauthors = coauthors_of(g, person, publications)

    builders = {
        "profile": lambda: profile_features(
            g, person, publications, projects, coauthors
        ),
        "contact": lambda: contact_features(g, person),
        "project": lambda: {
            f"project:{node_name(g, p).split(' - ')[0]}": 1 for p in projects
        },
        "topic": lambda: topic_features(
            g, publications | projects, topics_worked_on(g, person)
        ),
        "coauthor": lambda: {
            f"coauthor:{node_name(g, c)}": 1 for c in coauthors
        },
        "group_path": lambda: group_path_features(
            publications, projects, group_outputs
        ),
    }
    features = {}
    for family in families:
        features.update(builders[family]())
    return features


def build_features(g: Graph, persons, families) -> pd.DataFrame:
    """Persons x features table (rows in the order given)."""
    unknown = set(families) - set(FAMILIES)
    if unknown:
        raise ValueError(f"Unknown feature families: {sorted(unknown)}")

    group_outputs = {
        node_name(g, group): (
            set(g.objects(group, PUBLISHES)),
            set(g.objects(group, CARRIES_OUT)),
        )
        for group in label_nodes(g)
    }
    rows = [
        person_features(g, URIRef(p), families, group_outputs)
        for p in persons
    ]
    table = pd.DataFrame(rows, index=list(persons)).fillna(0).astype(float)
    return table[sorted(table.columns)]


class MinSupportFilter(BaseEstimator, TransformerMixin):
    """Keep features that are non-zero for at least `min_support` rows.

    Fitted on training rows only: a value seen for a single person is an
    identifier, not a pattern that can transfer to new persons.
    """

    def __init__(self, min_support: int = 2):
        self.min_support = min_support

    def fit(self, X, y=None):
        support = (np.asarray(X) != 0).sum(axis=0)
        self.support_ = support >= self.min_support
        return self

    def transform(self, X):
        return np.asarray(X, dtype=float)[:, self.support_]

    def get_support(self) -> np.ndarray:
        return self.support_
