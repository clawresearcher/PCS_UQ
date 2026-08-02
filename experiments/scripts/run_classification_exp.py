# Standard imports
import numpy as np
from pathlib import Path
import os
import pickle
import argparse
import tempfile

# Sklearn imports
from sklearn.model_selection import train_test_split

# Metrics imports
from src.metrics.classification_metrics import get_all_metrics, get_all_class_metrics


# Experiment configs
from experiments.configs.classification_configs import (
    get_classification_datasets,
    get_conformal_methods,
    get_pcs_methods,
)
from experiments.configs.classification_consts import (
    VALID_UQ_METHODS,
    VALID_ESTIMATORS,
    SINGLE_CONFORMAL_METHODS,
)


def atomic_pickle_dump(value, path):
    path = Path(path)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            pickle.dump(value, handle)
            handle.flush()
            os.fsync(handle.fileno())
        Path(temporary).replace(path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def run_classification_experiments(
    dataset_name,
    seed,
    uq_method,
    uq_method_name,
    method_name,
    results_dir="experiments/results/class_max",
    max_samples=5000,
    train_size=0.8,
):

    X_df, y, bin_df, importance = get_classification_datasets(dataset_name)
    X = X_df.to_numpy()
    # X,y, bin_df, X_df = X[:max_samples], y[:max_samples], bin_df[:max_samples], X_df[:max_samples]

    # Create results directory structure
    results_path = Path(results_dir)
    dataset_path = results_path / dataset_name
    # seed_path = dataset_path / str(seed)
    print(f"data created\n", flush=True)
    # Create directories if they don't exist
    dataset_path.mkdir(parents=True, exist_ok=True)
    stem = f"{method_name}_seed_{seed}_train_size_{train_size}"
    artifact_files = {
        "metrics": dataset_path / f"{stem}_metrics.pkl",
        "full_metrics": dataset_path / f"{stem}_full_metrics.pkl",
        "class_metrics": dataset_path / f"{stem}_class_metrics.pkl",
        "full_class_metrics": dataset_path / f"{stem}_full_class_metrics.pkl",
    }

    if all(path.exists() for path in artifact_files.values()):
        print(
            f"Complete result set already exists for {method_name}; skipping.\n",
            flush=True,
        )
        return
    for path in artifact_files.values():
        path.unlink(missing_ok=True)

    (
        X_train,
        X_test,
        y_train,
        y_test,
        bin_df_train,
        bin_df_test,
        X_df_train,
        X_df_test,
    ) = train_test_split(
        X, y, bin_df, X_df, train_size=train_size, random_state=seed, stratify=y
    )

    print(f"Fitting {method_name} on {dataset_name} with seed {seed}\n", flush=True)
    uq_method.fit(X_train, y_train)
    y_pred = uq_method.predict(X_test)

    full_metrics = get_all_metrics(y_test, y_pred, empty_set="to_full")
    full_class_metrics = get_all_class_metrics(y_test, y_pred, empty_set="to_full")
    metrics = get_all_metrics(y_test, y_pred, empty_set=None)
    class_metrics = get_all_class_metrics(y_test, y_pred, empty_set=None)
    print(f"{method_name} metrics: {metrics}\n", flush=True)
    print(f"{method_name} full metrics: {full_metrics}\n", flush=True)
    print(f"{method_name} class metrics: {class_metrics}\n", flush=True)
    print(f"{method_name} full class metrics: {full_class_metrics}\n", flush=True)

    for kind, value in {
        "full_metrics": full_metrics,
        "full_class_metrics": full_class_metrics,
        "metrics": metrics,
        "class_metrics": class_metrics,
    }.items():
        atomic_pickle_dump(value, artifact_files[kind])


if __name__ == "__main__":
    # Example methods dictionary
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="data_chess",
        help="Name of dataset to run experiments on",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--UQ_method", type=str, default="pcs_uq", help="UQ method to use"
    )
    parser.add_argument(
        "--estimator", type=str, default="ExtraTrees", help="Estimator to use"
    )
    parser.add_argument("--train_size", type=float, default=0.8, help="Train size")
    args = parser.parse_args()

    # Validate UQ method argument
    print(
        f"Running {args.UQ_method} on {args.dataset} with seed {args.seed} and train size {args.train_size}\n",
        flush=True,
    )

    if args.UQ_method not in VALID_UQ_METHODS:
        raise ValueError(
            f"Invalid UQ method '{args.UQ_method}'. Must be one of: {VALID_UQ_METHODS}"
        )

    if args.estimator not in VALID_ESTIMATORS:
        raise ValueError(
            f"Invalid estimator '{args.estimator}'. Must be one of: {VALID_ESTIMATORS}"
        )

    if args.UQ_method in SINGLE_CONFORMAL_METHODS:
        uq_method, method_name = get_conformal_methods(
            args.UQ_method, args.estimator, args.seed
        )

    elif args.UQ_method == "majority_vote":
        uq_method, method_name = get_conformal_methods(
            "majority_vote", args.estimator, args.seed
        )
        method_name = f"majority_vote"

    elif args.UQ_method == "pcs_uq":
        uq_method = get_pcs_methods("pcs_uq", args.seed)
        method_name = "pcs_uq"

    elif args.UQ_method == "pcs_oob":
        uq_method = get_pcs_methods("pcs_oob", args.seed)
        method_name = "pcs_oob"

    else:
        raise ValueError(
            f"Invalid UQ method '{args.UQ_method}'. Must be one of: {VALID_UQ_METHODS}"
        )

    # Set random seed
    np.random.seed(args.seed)
    print(f"starting experiment\n", flush=True)
    run_classification_experiments(
        dataset_name=args.dataset,
        seed=args.seed,
        uq_method=uq_method,
        uq_method_name=args.UQ_method,
        method_name=method_name,
        train_size=args.train_size,
    )
