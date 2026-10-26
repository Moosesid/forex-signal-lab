import pytest

from fxlab.eval.walkforward import walk_forward_splits


def _folds():
    return list(walk_forward_splits(n_rows=1000, train_size=300, test_size=50))


def test_train_always_before_test():
    for fold in _folds():
        assert fold.train.max() < fold.test.min()


def test_fold_count():
    # default purge 1: 1000 - 300 - 1 = 699 rows left, 699 / 50 = 13 full folds
    assert len(_folds()) == 13


def test_test_windows_do_not_overlap():
    tested_rows = []
    for fold in _folds():
        tested_rows.extend(fold.test)
    assert len(tested_rows) == len(set(tested_rows))


def test_never_runs_past_the_data():
    for fold in _folds():
        assert fold.test.max() < 1000


def test_too_little_data_gives_no_folds():
    folds = list(walk_forward_splits(n_rows=100, train_size=300, test_size=50))
    assert folds == []


def test_purge_gap_covers_horizon():
    horizon = 5
    for fold in walk_forward_splits(1000, 300, 50, horizon=horizon, purge=horizon):
        last_label_reaches = fold.train.max() + horizon
        assert last_label_reaches < fold.test.min()


def test_purge_smaller_than_horizon_is_rejected():
    with pytest.raises(ValueError, match="purge"):
        list(walk_forward_splits(1000, 300, 50, horizon=5, purge=2))
