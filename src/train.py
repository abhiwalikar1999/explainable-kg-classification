"""Training loop, evaluation metrics, and checkpoint saving."""
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

from src.dataset import PROJECT_ROOT, load_dataset
from src.model import get_models

DEFAULT_CHECKPOINT_DIR = PROJECT_ROOT / "outputs/checkpoints"


def compute_metrics(y_true, y_pred) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "recall_macro": recall_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }


def evaluate_model(model, x, y) -> dict:
    y_pred = model.predict(x)
    return compute_metrics(y, y_pred)


def train_and_evaluate(model, x_train, y_train, x_test, y_test):
    model.fit(x_train, y_train)
    metrics = {
        "train": evaluate_model(model, x_train, y_train),
        "test": evaluate_model(model, x_test, y_test),
    }
    return model, metrics


def save_model(
    model, name: str, checkpoint_dir: Path = DEFAULT_CHECKPOINT_DIR
) -> Path:
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path = checkpoint_dir / f"{name}.joblib"
    joblib.dump(model, path)
    return path


def build_metrics_table(results: dict) -> pd.DataFrame:
    rows = []
    for model_name, splits in results.items():
        for split_name, metrics in splits.items():
            row = {"model": model_name, "split": split_name, **metrics}
            rows.append(row)
    return pd.DataFrame(rows)


def run_training_pipeline(
    checkpoint_dir: Path = DEFAULT_CHECKPOINT_DIR,
) -> pd.DataFrame:
    dataset = load_dataset()
    models = get_models()

    results = {}
    for name, model in models.items():
        _, metrics = train_and_evaluate(
            model,
            dataset.X_train,
            dataset.y_train,
            dataset.X_test,
            dataset.y_test,
        )
        results[name] = metrics
        save_model(model, name, checkpoint_dir)

    return build_metrics_table(results)


if __name__ == "__main__":
    print(run_training_pipeline())
