"""Schema-agnostic features for entities of any RDF graph.

Walks one and two hops out from each entity and counts what it finds, so it
needs no knowledge of the ontology (a simplified version of Ristoski &
Paulheim's propositionalization strategies). Feature names:

    p=value       a categorical value reached via predicate p (count)
    p:count       number of p-neighbours
    p/q=value     a categorical value two hops away, via p then q (count)
    p:mean        mean of a numeric literal via p (or p/q)
    in:p:count    number of incoming p-edges

Values that identify a single entity (neighbour ids, names) are dropped
later by `MinSupportFilter`, inside every CV fold.
"""
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from rdflib import Graph, Literal, URIRef

MAX_CATEGORICAL_LENGTH = 40


def local_name(node) -> str:
    """Last fragment of a URI ('…/carcinogenesis#Nitrogen-36' -> …-36)."""
    text = str(node)
    return text.rsplit("#", 1)[-1].rstrip("/").rsplit("/", 1)[-1]


def numeric(value: Literal):
    try:
        return float(str(value))
    except ValueError:
        return None


def add_value(counts: Counter, numbers: dict, key: str, value) -> None:
    """Record one value: numeric literals for averaging, the rest counted."""
    if isinstance(value, Literal):
        number = numeric(value)
        if number is not None:
            numbers[key].append(number)
            return
        text = str(value).strip()
        if len(text) <= MAX_CATEGORICAL_LENGTH:
            counts[f"{key}={text}"] += 1
        return
    counts[f"{key}={local_name(value)}"] += 1


def entity_features(g: Graph, entity, blocked: set) -> dict:
    """1- and 2-hop features of one entity, never passing through `blocked`.

    `blocked` holds nodes that must not appear as feature values, e.g. the
    label nodes, so no feature is a path to the label.
    """
    counts, numbers = Counter(), defaultdict(list)
    for p, o in g.predicate_objects(entity):
        if o in blocked:
            continue
        key = local_name(p)
        if isinstance(o, Literal):
            add_value(counts, numbers, key, o)
            continue
        counts[f"{key}:count"] += 1
        add_value(counts, numbers, key, o)
        for q, o2 in g.predicate_objects(o):
            if o2 in blocked or o2 == entity:
                continue
            add_value(counts, numbers, f"{key}/{local_name(q)}", o2)

    for s, p in g.subject_predicates(entity):
        if s not in blocked:
            counts[f"in:{local_name(p)}:count"] += 1

    features = dict(counts)
    for key, values in numbers.items():
        features[f"{key}:mean"] = float(np.mean(values))
    return features


def build_generic_features(g: Graph, entities, blocked=frozenset()):
    """Entities x features table (rows in the order given)."""
    rows = [entity_features(g, URIRef(e), set(blocked)) for e in entities]
    table = pd.DataFrame(rows, index=list(entities)).fillna(0).astype(float)
    return table[sorted(table.columns)]
