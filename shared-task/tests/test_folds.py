"""Focused tests for deterministic paragraph-grouped cross-validation."""

from collections import Counter
import os
import subprocess
import sys
import textwrap

import pytest

from daleel.constants import GENRES
from daleel.data import TRAIN_TASK1, TRAIN_TASK2, load_records
from daleel.folds import (
    DEFAULT_FOLD_SEED,
    Fold,
    fold_manifest,
    fold_manifest_hash,
    make_stratified_folds,
    validate_folds,
)
from daleel.splits import clean_task1_gold


@pytest.fixture(scope="module")
def official_inputs():
    if not TRAIN_TASK1.exists() or not TRAIN_TASK2.exists():
        pytest.skip("official Daleel training clone is not available")
    task1 = load_records(TRAIN_TASK1)
    task2 = load_records(TRAIN_TASK2)
    cleaned_labels = clean_task1_gold(task1, task2)
    genres = {record["paragraph_id"]: record["type"] for record in task1}
    return tuple(cleaned_labels), cleaned_labels, genres


@pytest.fixture(scope="module")
def official_folds(official_inputs):
    ids, labels, genres = official_inputs
    return make_stratified_folds(
        ids,
        labels,
        genres,
        n_splits=5,
        seed=DEFAULT_FOLD_SEED,
    )


def test_official_folds_are_an_exact_balanced_partition(
    official_inputs, official_folds
):
    ids, labels, genres = official_inputs
    expected = set(ids)

    validation_occurrences = Counter(
        paragraph_id
        for fold in official_folds
        for paragraph_id in fold.val_ids
    )
    assert set(validation_occurrences) == expected
    assert set(validation_occurrences.values()) == {1}

    for fold in official_folds:
        assert not set(fold.train_ids) & set(fold.val_ids)
        assert set(fold.train_ids) | set(fold.val_ids) == expected

    fold_sizes = [len(fold.val_ids) for fold in official_folds]
    assert max(fold_sizes) - min(fold_sizes) <= 1

    # Both genre marginals should track the exact fold-size balance closely.
    for genre in GENRES:
        genre_counts = [
            sum(genres[paragraph_id] == genre for paragraph_id in fold.val_ids)
            for fold in official_folds
        ]
        assert max(genre_counts) - min(genre_counts) <= 1

    # CO and ST must reach every fold whenever their marginal (or a
    # genre-specific marginal) has at least one example available per fold.
    for label in ("CO", "ST"):
        label_total = sum(label in labels[paragraph_id] for paragraph_id in ids)
        label_counts = [
            sum(label in labels[paragraph_id] for paragraph_id in fold.val_ids)
            for fold in official_folds
        ]
        if label_total >= len(official_folds):
            assert min(label_counts) >= 1

        for genre in GENRES:
            joint_total = sum(
                label in labels[paragraph_id] and genres[paragraph_id] == genre
                for paragraph_id in ids
            )
            joint_counts = [
                sum(
                    label in labels[paragraph_id]
                    and genres[paragraph_id] == genre
                    for paragraph_id in fold.val_ids
                )
                for fold in official_folds
            ]
            if joint_total >= len(official_folds):
                assert min(joint_counts) >= 1


def test_assignment_is_deterministic_and_input_order_independent(official_inputs):
    ids, labels, genres = official_inputs
    expected = make_stratified_folds(ids, labels, genres, n_splits=5, seed=91)

    reversed_ids = tuple(reversed(ids))
    reversed_labels = {paragraph_id: labels[paragraph_id] for paragraph_id in reversed_ids}
    reversed_genres = {paragraph_id: genres[paragraph_id] for paragraph_id in reversed_ids}
    actual = make_stratified_folds(
        reversed_ids,
        reversed_labels,
        reversed_genres,
        n_splits=5,
        seed=91,
    )
    assert actual == expected


def test_manifest_hash_is_pythonhashseed_independent(official_inputs):
    # The 5 outer folds run as separately launched processes, so any
    # PYTHONHASHSEED sensitivity (e.g. float sums over frozenset iteration)
    # would make their manifests disagree and reject aggregation. Hash-seed
    # state is fixed at interpreter start, hence the subprocesses.
    del official_inputs  # only used to skip when the corpus is absent
    code = textwrap.dedent(
        """
        from daleel.data import TRAIN_TASK1, TRAIN_TASK2, load_records
        from daleel.folds import (
            DEFAULT_FOLD_SEED,
            fold_manifest,
            fold_manifest_hash,
            make_stratified_folds,
        )
        from daleel.splits import clean_task1_gold

        task1 = load_records(TRAIN_TASK1)
        task2 = load_records(TRAIN_TASK2)
        labels = clean_task1_gold(task1, task2)
        genres = {record["paragraph_id"]: record["type"] for record in task1}
        folds = make_stratified_folds(
            tuple(labels), labels, genres, n_splits=5, seed=DEFAULT_FOLD_SEED
        )
        print(fold_manifest_hash(fold_manifest(folds, seed=DEFAULT_FOLD_SEED)))
        """
    )
    digests = set()
    for hash_seed in ("0", "31337"):
        result = subprocess.run(
            [sys.executable, "-c", code],
            env={**os.environ, "PYTHONHASHSEED": hash_seed},
            capture_output=True,
            text=True,
            check=True,
        )
        digests.add(result.stdout.strip())
    assert len(digests) == 1


def test_manifest_and_hash_are_stable(official_folds):
    manifest = fold_manifest(official_folds, seed=DEFAULT_FOLD_SEED)
    reordered_manifest = fold_manifest(
        tuple(reversed(official_folds)), seed=DEFAULT_FOLD_SEED
    )
    assert reordered_manifest == manifest

    digest = fold_manifest_hash(manifest)
    assert digest == fold_manifest_hash(reordered_manifest)
    assert len(digest) == 64
    assert set(digest) <= set("0123456789abcdef")

    changed = dict(manifest)
    changed["seed"] = DEFAULT_FOLD_SEED + 1
    assert fold_manifest_hash(changed) != digest


def test_input_and_partition_validation_fail_loudly():
    ids = (1, 2, 3, 4)
    labels = {1: {"ST"}, 2: {"CO"}, 3: {"AS"}, 4: set()}
    genres = {1: "debate", 2: "editorial", 3: "debate", 4: "editorial"}

    with pytest.raises(ValueError, match="at least 2"):
        make_stratified_folds(ids, labels, genres, n_splits=1)
    with pytest.raises(ValueError, match="cannot exceed"):
        make_stratified_folds(ids, labels, genres, n_splits=5)
    with pytest.raises(ValueError, match="duplicate IDs"):
        make_stratified_folds((1, 1, 2), {1: set(), 2: set()}, {1: "debate", 2: "debate"})

    missing_labels = dict(labels)
    del missing_labels[4]
    with pytest.raises(ValueError, match="cover record_ids exactly"):
        make_stratified_folds(ids, missing_labels, genres, n_splits=2)

    valid = make_stratified_folds(ids, labels, genres, n_splits=2)
    broken = (
        Fold(valid[0].index, valid[0].train_ids, valid[0].val_ids + (valid[0].val_ids[0],)),
        valid[1],
    )
    with pytest.raises(ValueError, match="duplicate validation IDs"):
        validate_folds(broken, ids)
