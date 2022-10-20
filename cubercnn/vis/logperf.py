# Copyright (c) Meta Platforms, Inc. and affiliates
from termcolor import colored
import itertools
from tabulate import tabulate
import logging

logger = logging.getLogger(__name__)

def print_ap_category_histogram(dataset, results):
    """
    Prints AP performance for each category.
    Args:
        results: dictionary; each entry contains information for a dataset
    """
    num_classes = len(results)
    N_COLS = 9
    data = list(
        itertools.chain(
            *[
                [
                    cat,
                    out["AP2D"],
                    out["AP3D"],
                ]
                for cat, out in results.items()
            ]
        )
    )
    data.extend([None] * (N_COLS - (len(data) % N_COLS)))
    data = itertools.zip_longest(*[data[i::N_COLS] for i in range(N_COLS)])
    table = tabulate(
        data,
        headers=["category", "AP2D", "AP3D"] * (N_COLS // 2),
        tablefmt="pipe",
        numalign="left",
        stralign="center",
    )
    logger.info(
        "Performance for each of {} categories on {}:\n".format(num_classes, dataset)
        + colored(table, "cyan")
    )


def print_ap_analysis_histogram(results):
    """
    Prints AP performance for various IoU thresholds and (near, medium, far) objects.
    Args:
        results: dictionary. Each entry in results contains outputs for a dataset
    """
    metric_names = ["AP2D", "AP3D", "AP3D@15", "AP3D@25", "AP3D@50", "AP3D-N", "AP3D-M", "AP3D-F"]
    N_COLS = 10
    data = []
    for name, metrics in results.items():
        data_item = [name, metrics["iters"], metrics["AP2D"], metrics["AP3D"], metrics["AP3D@15"], metrics["AP3D@25"], metrics["AP3D@50"], metrics["AP3D-N"], metrics["AP3D-M"], metrics["AP3D-F"]]
        data.append(data_item)
    table = tabulate(
        data,
        headers=["Dataset", "#iters", "AP2D", "AP3D", "AP3D@15", "AP3D@25", "AP3D@50", "AP3D-N", "AP3D-M", "AP3D-F"],
        tablefmt="grid",
        numalign="left",
        stralign="center",
    )
    logger.info(
        "Per-dataset performance analysis on test set:\n"
        + colored(table, "cyan")
    )


def print_ap_dataset_histogram(results):
    """
    Prints AP performance for each dataset.
    Args:
        results: list of dicts. Each entry in results contains outputs for a dataset
    """
    metric_names = ["AP2D", "AP3D"]
    N_COLS = 4
    data = []
    for name, metrics in results.items():
        data_item = [name, metrics["iters"], metrics["AP2D"], metrics["AP3D"]]
        data.append(data_item)
    table = tabulate(
        data,
        headers=["Dataset", "#iters", "AP2D", "AP3D"],
        tablefmt="grid",
        numalign="left",
        stralign="center",
    )
    logger.info(
        "Per-dataset performance on test set:\n"
        + colored(table, "cyan")
    )


def print_ap_omni_histogram(results):
    """
    Prints AP performance for Omni3D dataset.
    Args:
        results: list of dicts. Each entry in results contains outputs for a dataset
    """
    metric_names = ["AP2D", "AP3D"]
    N_COLS = 4
    data = []
    for name, metrics in results.items():
        data_item = [name, metrics["iters"], metrics["AP2D"], metrics["AP3D"]]
        data.append(data_item)
    table = tabulate(
        data,
        headers=["Dataset", "#iters", "AP2D", "AP3D"],
        tablefmt="grid",
        numalign="left",
        stralign="center",
    )
    logger.info("Omni3D performance on test set. The numbers below should be used to compare to others approaches on Omni3D, such as Cube R-CNN")
    logger.info(
        "Performance on Omni3D:\n"
        + colored(table, "magenta")
    )
