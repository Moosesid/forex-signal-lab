"""Walk-forward splitter: train on the past, test on what comes next."""

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Fold:
    number: int
    train: np.ndarray
    test: np.ndarray


def walk_forward_splits(n_rows: int, train_size: int, test_size: int) -> Iterator[Fold]:
    number = 0
    train_start = 0
    while True:
        train_end = train_start + train_size
        test_start = train_end
        test_end = test_start + test_size
        if test_end > n_rows:
            break
        yield Fold(
            number=number,
            train=np.arange(train_start, train_end),
            test=np.arange(test_start, test_end),
        )
        number += 1
        train_start += test_size
