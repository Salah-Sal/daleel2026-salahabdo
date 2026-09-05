"""Deterministic paragraph-grouped folds for Daleel model selection.

The task is multilabel, so a conventional single-target stratifier cannot
preserve the six Task 1 label marginals.  ``make_stratified_folds`` uses a
small iterative multilabel allocator instead.  It balances three kinds of
binary feature at once:

* the paragraph genre;
* each cleaned Task 1 label; and
* each observed genre/label interaction.

Rare features are allocated first.  This is important for CO and ST and,
in particular, for debate/ST: there are only five cleaned debate/ST
paragraphs in the current training release.  A paragraph is the indivisible
unit throughout, so no text can leak between train and validation sides.

No Python hash values are used for tie-breaking.  All ties use SHA-256 over
the explicit seed and stable input values, making assignments independent of
input order and ``PYTHONHASHSEED``.  The manifest helpers use canonical JSON
for the same reason.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any

from .constants import GENRES, LABELS


DEFAULT_FOLD_SEED = 20260710
FOLD_STRATEGY = "iterative-multilabel-genre-v1"
FOLD_MANIFEST_VERSION = 1


@dataclass(frozen=True, slots=True)
class Fold:
    """One cross-validation split.

    ``val_ids`` across a complete fold collection partition the paragraph
    IDs exactly once.  ``train_ids`` is always its exact complement.
    Tuples keep the object immutable and directly serializable by the
    manifest helper.
    """

    index: int
    train_ids: tuple[int, ...]
    val_ids: tuple[int, ...]

    @property
    def validation_ids(self) -> tuple[int, ...]:
        """Verbose alias for callers that prefer ``validation_ids``."""

        return self.val_ids


def _validate_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}")
    return value


def _mapping_coverage(
    mapping: Mapping[int, object], expected: set[int], name: str
) -> None:
    actual = set(mapping)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing or extra:
        details = []
        if missing:
            details.append(f"missing IDs {missing}")
        if extra:
            details.append(f"unexpected IDs {extra}")
        raise ValueError(f"{name} must cover record_ids exactly: " + "; ".join(details))


def _normalize_inputs(
    record_ids: Iterable[int],
    labels_by_id: Mapping[int, Collection[str]],
    genres_by_id: Mapping[int, str],
    n_splits: int,
    seed: int,
) -> tuple[tuple[int, ...], dict[int, frozenset[str]], dict[int, str], int, int]:
    n_splits = _validate_integer(n_splits, "n_splits")
    seed = _validate_integer(seed, "seed")

    supplied_ids = tuple(record_ids)
    for paragraph_id in supplied_ids:
        _validate_integer(paragraph_id, "record ID")
    if len(set(supplied_ids)) != len(supplied_ids):
        duplicates = sorted(
            paragraph_id
            for paragraph_id, count in Counter(supplied_ids).items()
            if count > 1
        )
        raise ValueError(f"record_ids contains duplicate IDs: {duplicates}")

    ids = tuple(sorted(supplied_ids))
    if n_splits < 2:
        raise ValueError(f"n_splits must be at least 2, got {n_splits}")
    if n_splits > len(ids):
        raise ValueError(
            f"n_splits ({n_splits}) cannot exceed the number of records ({len(ids)})"
        )

    expected = set(ids)
    _mapping_coverage(labels_by_id, expected, "labels_by_id")
    _mapping_coverage(genres_by_id, expected, "genres_by_id")

    normalized_labels: dict[int, frozenset[str]] = {}
    normalized_genres: dict[int, str] = {}
    allowed_labels = set(LABELS)
    allowed_genres = set(GENRES)
    for paragraph_id in ids:
        raw_labels = labels_by_id[paragraph_id]
        if isinstance(raw_labels, (str, bytes)) or not isinstance(raw_labels, Collection):
            raise TypeError(
                f"labels_by_id[{paragraph_id}] must be a collection of label codes"
            )
        labels = frozenset(raw_labels)
        unknown_labels = sorted(label for label in labels if label not in allowed_labels)
        if unknown_labels:
            raise ValueError(
                f"labels_by_id[{paragraph_id}] contains unknown labels: {unknown_labels}"
            )
        normalized_labels[paragraph_id] = labels

        genre = genres_by_id[paragraph_id]
        if genre not in allowed_genres:
            raise ValueError(
                f"genres_by_id[{paragraph_id}] is {genre!r}; expected one of "
                f"{sorted(allowed_genres)}"
            )
        normalized_genres[paragraph_id] = genre

    return ids, normalized_labels, normalized_genres, n_splits, seed


def _stable_tie(seed: int, namespace: str, value: object) -> bytes:
    payload = json.dumps(
        [seed, namespace, value],
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).digest()


def _paragraph_features(
    paragraph_id: int,
    labels_by_id: Mapping[int, frozenset[str]],
    genres_by_id: Mapping[int, str],
) -> frozenset[str]:
    genre = genres_by_id[paragraph_id]
    labels = labels_by_id[paragraph_id]
    return frozenset(
        [f"genre:{genre}"]
        + [f"label:{label}" for label in labels]
        + [f"genre-label:{genre}:{label}" for label in labels]
    )


def make_stratified_folds(
    record_ids: Iterable[int],
    labels_by_id: Mapping[int, Collection[str]],
    genres_by_id: Mapping[int, str],
    *,
    n_splits: int = 5,
    seed: int = DEFAULT_FOLD_SEED,
) -> tuple[Fold, ...]:
    """Create deterministic, paragraph-grouped multilabel folds.

    Args:
        record_ids: Every paragraph ID to include, exactly once.
        labels_by_id: Cleaned Task 1 label set for every supplied ID.  Callers
            should normally pass ``clean_task1_gold(...)`` from
            :mod:`daleel.splits`, not the uncleaned official Task 1 labels.
        genres_by_id: Official ``type`` value for every supplied ID.
        n_splits: Number of folds; must be in ``[2, len(record_ids)]``.
        seed: Explicit deterministic tie-breaking seed.

    Returns:
        An immutable tuple of :class:`Fold` objects ordered by fold index.
        Fold sizes differ by at most one and validation IDs partition the
        supplied IDs exactly once.

    The allocator follows the core idea of iterative multilabel
    stratification: repeatedly choose the rarest feature still unassigned,
    choose its hardest remaining paragraph, and put that paragraph in the
    fold with the greatest deficit for that feature.  Aggregate feature
    deficit and remaining capacity break substantive ties; SHA-256 breaks
    otherwise exact ties reproducibly.
    """

    ids, labels, genres, n_splits, seed = _normalize_inputs(
        record_ids, labels_by_id, genres_by_id, n_splits, seed
    )

    features_by_id = {
        paragraph_id: _paragraph_features(paragraph_id, labels, genres)
        for paragraph_id in ids
    }
    feature_totals = Counter(
        feature
        for paragraph_id in ids
        for feature in features_by_id[paragraph_id]
    )

    # Exact capacities guarantee the strongest possible fold-size balance.
    base_size, remainder = divmod(len(ids), n_splits)
    extra_order = sorted(
        range(n_splits), key=lambda fold: _stable_tie(seed, "extra-fold", fold)
    )
    extra_folds = set(extra_order[:remainder])
    capacities = [base_size + (fold in extra_folds) for fold in range(n_splits)]

    assigned_ids: list[list[int]] = [[] for _ in range(n_splits)]
    assigned_feature_counts = {
        feature: [0] * n_splits for feature in feature_totals
    }
    # Slightly larger folds receive proportionally larger feature targets.
    feature_targets = {
        feature: [
            total * capacities[fold] / len(ids) for fold in range(n_splits)
        ]
        for feature, total in feature_totals.items()
    }

    unassigned = set(ids)
    remaining_feature_counts = Counter(feature_totals)
    while unassigned:
        positive_features = [
            feature
            for feature, count in remaining_feature_counts.items()
            if count > 0
        ]
        if not positive_features:  # Defensive: genre features make this unreachable.
            paragraph_id = min(
                unassigned,
                key=lambda pid: _stable_tie(seed, "unfeatured-paragraph", pid),
            )
            selected_feature = None
        else:
            rarest_count = min(remaining_feature_counts[f] for f in positive_features)
            rarest_features = [
                feature
                for feature in positive_features
                if remaining_feature_counts[feature] == rarest_count
            ]
            selected_feature = min(
                rarest_features,
                key=lambda feature: _stable_tie(seed, "feature", feature),
            )
            candidates = [
                paragraph_id
                for paragraph_id in unassigned
                if selected_feature in features_by_id[paragraph_id]
            ]

            def paragraph_priority(paragraph_id: int) -> tuple[float, int, bytes]:
                paragraph_features = features_by_id[paragraph_id]
                # fsum over a sorted view: frozenset iteration order varies
                # with PYTHONHASHSEED, and plain float addition is not
                # associative, so this sum must not depend on either.
                rarity = math.fsum(
                    1.0 / remaining_feature_counts[feature]
                    for feature in sorted(paragraph_features)
                    if remaining_feature_counts[feature] > 0
                )
                return (
                    -rarity,
                    -len(paragraph_features),
                    _stable_tie(seed, "paragraph", paragraph_id),
                )

            paragraph_id = min(candidates, key=paragraph_priority)

        paragraph_features = features_by_id[paragraph_id]
        available_folds = [
            fold
            for fold in range(n_splits)
            if len(assigned_ids[fold]) < capacities[fold]
        ]

        def fold_priority(fold: int) -> tuple[float, float, int, bytes]:
            if selected_feature is None:
                selected_deficit = 0.0
            else:
                selected_deficit = (
                    feature_targets[selected_feature][fold]
                    - assigned_feature_counts[selected_feature][fold]
                )
            aggregate_deficit = math.fsum(
                (
                    feature_targets[feature][fold]
                    - assigned_feature_counts[feature][fold]
                )
                / feature_totals[feature]
                for feature in sorted(paragraph_features)
            )
            capacity_left = capacities[fold] - len(assigned_ids[fold])
            return (
                -selected_deficit,
                -aggregate_deficit,
                -capacity_left,
                _stable_tie(seed, f"fold-for:{paragraph_id}", fold),
            )

        destination = min(available_folds, key=fold_priority)
        assigned_ids[destination].append(paragraph_id)
        unassigned.remove(paragraph_id)
        for feature in paragraph_features:
            assigned_feature_counts[feature][destination] += 1
            remaining_feature_counts[feature] -= 1

    all_ids = set(ids)
    folds = tuple(
        Fold(
            index=fold,
            train_ids=tuple(sorted(all_ids - set(assigned_ids[fold]))),
            val_ids=tuple(sorted(assigned_ids[fold])),
        )
        for fold in range(n_splits)
    )
    validate_folds(folds, ids)
    return folds


def validate_folds(folds: Sequence[Fold], record_ids: Iterable[int]) -> None:
    """Raise ``ValueError`` unless folds form an exact CV partition.

    Validation checks indices, duplicate IDs, train/validation disjointness,
    per-fold coverage, and exactly-once validation coverage across folds.
    """

    ids = tuple(record_ids)
    for paragraph_id in ids:
        _validate_integer(paragraph_id, "record ID")
    if len(set(ids)) != len(ids):
        raise ValueError("record_ids must be unique when validating folds")
    expected = set(ids)
    if len(folds) < 2:
        raise ValueError("at least two folds are required")

    indices = [fold.index for fold in folds]
    if sorted(indices) != list(range(len(folds))):
        raise ValueError(
            f"fold indices must be exactly 0..{len(folds) - 1}, got {indices}"
        )

    validation_counts: Counter[int] = Counter()
    for fold in folds:
        train = tuple(fold.train_ids)
        val = tuple(fold.val_ids)
        if len(train) != len(set(train)):
            raise ValueError(f"fold {fold.index} has duplicate train IDs")
        if len(val) != len(set(val)):
            raise ValueError(f"fold {fold.index} has duplicate validation IDs")
        train_set, val_set = set(train), set(val)
        overlap = sorted(train_set & val_set)
        if overlap:
            raise ValueError(
                f"fold {fold.index} has train/validation overlap: {overlap}"
            )
        if train_set | val_set != expected:
            missing = sorted(expected - (train_set | val_set))
            extra = sorted((train_set | val_set) - expected)
            raise ValueError(
                f"fold {fold.index} does not cover record_ids exactly; "
                f"missing={missing}, extra={extra}"
            )
        validation_counts.update(val)

    wrong_counts = {
        paragraph_id: validation_counts[paragraph_id]
        for paragraph_id in sorted(expected)
        if validation_counts[paragraph_id] != 1
    }
    extra_validation_ids = sorted(set(validation_counts) - expected)
    if wrong_counts or extra_validation_ids:
        raise ValueError(
            "validation IDs must partition record_ids exactly once; "
            f"wrong_counts={wrong_counts}, extra={extra_validation_ids}"
        )


def fold_manifest(
    folds: Sequence[Fold],
    *,
    seed: int,
    strategy: str = FOLD_STRATEGY,
) -> dict[str, Any]:
    """Return a canonicalizable fold manifest suitable for run artifacts."""

    seed = _validate_integer(seed, "seed")
    ordered_folds = tuple(sorted(folds, key=lambda fold: fold.index))
    if not ordered_folds:
        raise ValueError("cannot build a manifest for an empty fold collection")
    universe = tuple(
        sorted(set(ordered_folds[0].train_ids) | set(ordered_folds[0].val_ids))
    )
    validate_folds(ordered_folds, universe)
    return {
        "schema_version": FOLD_MANIFEST_VERSION,
        "strategy": strategy,
        "seed": seed,
        "n_splits": len(ordered_folds),
        "record_count": len(universe),
        "record_ids": list(universe),
        "folds": [
            {
                "index": fold.index,
                "train_ids": list(sorted(fold.train_ids)),
                "val_ids": list(sorted(fold.val_ids)),
            }
            for fold in ordered_folds
        ],
    }


def fold_manifest_hash(manifest: Mapping[str, Any]) -> str:
    """SHA-256 hex digest of a manifest's canonical JSON representation."""

    canonical = json.dumps(
        manifest,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


__all__ = [
    "DEFAULT_FOLD_SEED",
    "FOLD_MANIFEST_VERSION",
    "FOLD_STRATEGY",
    "Fold",
    "fold_manifest",
    "fold_manifest_hash",
    "make_stratified_folds",
    "validate_folds",
]
