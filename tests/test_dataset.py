import numpy as np


def test_split_shapes(data):
    n_features = len(data.feature_names)
    assert data.X_train.shape == (140, n_features)
    assert data.X_test.shape == (36, n_features)
    assert data.X_unlabeled.shape[1] == n_features
    assert len(data.X_labeled_raw) == len(data.y_labeled) == 176


def test_train_and_test_persons_are_disjoint(data):
    train = set(data.train_labels_df["person"])
    test = set(data.test_labels_df["person"])
    assert not train & test


def test_unlabeled_rows_exclude_labelled_persons(data):
    n_labeled = len(data.train_labels_df) + len(data.test_labels_df)
    assert len(data.X_unlabeled) > n_labeled


def test_four_classes(data):
    assert len(data.label_encoder.classes_) == 4
    assert set(np.unique(data.y_test)) <= set(np.unique(data.y_train))
