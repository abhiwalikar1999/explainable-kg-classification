"""Run the whole project in order with one command.

    python -m src.pipeline                      # all stages except the GNN
    python -m src.pipeline --with-gnn           # also R-GCN (~45 min CPU)
    python -m src.pipeline --stages figures     # just some stages

Stage order matters: `evaluate` needs the checkpoints from `train`, and
`figures` reads the result files of every earlier stage.
"""
import argparse
import time


def run_experiments():
    from src.experiments import run_experiments as run
    run()


def run_explain_graph():
    from src.explain_graph import run_graph_explanations
    run_graph_explanations()


def run_demo():
    from src.predict import export_demo
    export_demo()


def run_benchmarks():
    from src.benchmarks import run_benchmarks as run
    run()


def run_train():
    from src.train import run_training_pipeline
    run_training_pipeline()


def run_evaluate():
    from src.evaluate import run_evaluation
    run_evaluation()


def run_gnn():
    from src.gnn import run_gnn as run  # needs requirements-gnn.txt
    run("aifb")
    run("mutag")


def run_figures():
    from src.figures import make_figures, make_graph_figures
    make_figures()
    make_graph_figures()


STAGES = {
    "experiments": run_experiments,
    "explain_graph": run_explain_graph,
    "demo": run_demo,
    "benchmarks": run_benchmarks,
    "train": run_train,
    "evaluate": run_evaluate,
    "gnn": run_gnn,
    "figures": run_figures,
}
DEFAULT_STAGES = [name for name in STAGES if name != "gnn"]


def selected_stages(stages=None, with_gnn: bool = False) -> list:
    """Requested stages in pipeline order."""
    wanted = set(stages or DEFAULT_STAGES) | ({"gnn"} if with_gnn else set())
    unknown = wanted - set(STAGES)
    if unknown:
        raise ValueError(f"Unknown stages: {sorted(unknown)}")
    return [name for name in STAGES if name in wanted]


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--stages", nargs="+", choices=list(STAGES))
    parser.add_argument("--with-gnn", action="store_true")
    args = parser.parse_args(argv)

    for name in selected_stages(args.stages, args.with_gnn):
        start = time.perf_counter()
        print(f"[pipeline] {name} ...", flush=True)
        STAGES[name]()
        print(f"[pipeline] {name} done in {time.perf_counter() - start:.0f}s")


if __name__ == "__main__":
    main()
