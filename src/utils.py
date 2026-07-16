"""
utils.py
========
Helper utilities for logging, reproducibility, seeding, and printing configurations.
"""

import os
import sys
import csv
import random
import datetime
import numpy as np
import torch


class Tee:
    """
    Redirect stdout to both the terminal and an open file handle.
    """

    def __init__(self, file_handle):
        self._file   = file_handle
        self._stdout = sys.stdout

    def write(self, data):
        self._stdout.write(data)
        self._file.write(data)

    def flush(self):
        self._stdout.flush()
        self._file.flush()

    def restore(self):
        return self._stdout


def save_run_config(args, log_path: str) -> "Tee":
    """
    Create a plain-text run log at `log_path` containing all CLI arguments,
    then attach a Tee to redirect stdout to the log file.
    """
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    log_file = open(log_path, "a")

    header = [
        "=" * 62,
        f"  RUN  —  {datetime.datetime.now().strftime('%Y-%m-%d  %H:%M:%S')}",
        "=" * 62,
        "  ARGUMENTS",
        "-" * 62,
    ]
    for k, v in vars(args).items():
        header.append(f"  {k:<30} {v}")
    header += ["-" * 62, "  TRAINING LOG", "-" * 62, ""]

    block = "\n".join(header) + "\n"
    log_file.write(block)
    log_file.flush()

    tee = Tee(log_file)
    sys.stdout = tee
    return tee


def print_experiment_config(args):
    print("\n" + "=" * 60)
    print(f"  Experiment : {args.experiment_name}")
    print("-" * 60)
    print(f"  Task       : SuStaIn Subtype & Stage Multi-Task CNN")
    print(f"  Encoder    : 3D ResNet-18 (prototype-free)")
    print(f"  Loss       : Soft Cross-Entropy (Subtype) + Ordinal BCE (Stage)")
    print("-" * 60)
    print(f"  Device     : {args.device}")
    print(f"  Epochs     : {args.epochs}")
    print(f"  Batch Size : {args.batch_size}")
    print("-" * 60)
    print(f"  LR         : {args.lr}")
    print(f"  Weight dec.: {args.weight_decay}")
    print(f"  Gamma (Dec): {args.gamma}")
    print("=" * 60 + "\n")


def set_global_seed(seed: int = 42, deterministic: bool = True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if deterministic:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.backends.cudnn.deterministic    = True
        torch.backends.cudnn.benchmark        = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.benchmark = True


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    random.seed(worker_seed)
    np.random.seed(worker_seed)
