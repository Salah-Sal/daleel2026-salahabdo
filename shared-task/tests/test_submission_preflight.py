import pytest

from daleel.submission import validate_records_against_source, validate_task2_records


SOURCE = [
    {"paragraph_id": 11, "text": "النص الأول", "type": "editorial"},
    {"paragraph_id": 12, "text": "النص الثاني", "type": "debate"},
]


def _task1_predictions():
    return [
        {**SOURCE[0], "labels": ["AS"]},
        {**SOURCE[1], "labels": []},
    ]


def _task2_predictions():
    return [
        {
            **SOURCE[0],
            "labels": [{"label": "AS", "start_offset": 0, "end_offset": 4}],
        },
        {**SOURCE[1], "labels": []},
    ]


@pytest.mark.parametrize(
    ("task", "predictions"),
    [("task_1", _task1_predictions), ("task_2", _task2_predictions)],
)
def test_source_bound_preflight_accepts_exact_task_records(task, predictions):
    assert validate_records_against_source(predictions(), SOURCE, task) == []


def test_source_bound_preflight_reports_missing_extra_duplicate_and_count():
    records = _task1_predictions()
    records.append({**SOURCE[0], "labels": ["CO"]})
    records[1] = {
        "paragraph_id": 99,
        "text": "ليس من ملف الإدخال",
        "type": "debate",
        "labels": [],
    }

    errors = validate_records_against_source(records, SOURCE, "task_1")

    assert any("record count mismatch" in error for error in errors)
    assert any("duplicate paragraph_id 11" in error for error in errors)
    assert any("missing paragraph_id values" in error and "12" in error for error in errors)
    assert any("unexpected paragraph_id values" in error and "99" in error for error in errors)


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("text", "النص الأول ", "text does not exactly match source"),
        ("type", "debate", "does not exactly match source"),
    ],
)
def test_source_bound_preflight_requires_exact_echoes(field, replacement, message):
    records = _task1_predictions()
    records[0][field] = replacement

    errors = validate_records_against_source(records, SOURCE, "task_1")

    assert any(message in error for error in errors)


def test_source_bound_preflight_includes_existing_schema_checks():
    records = _task1_predictions()
    records[0]["labels"] = ["NOT_A_LABEL"]

    errors = validate_records_against_source(records, SOURCE, "task_1")

    assert any("unknown labels" in error for error in errors)


def test_source_bound_preflight_rejects_duplicate_source_ids():
    source = [SOURCE[0], {**SOURCE[0], "text": "نسخة أخرى"}]

    errors = validate_records_against_source([_task1_predictions()[0]], source, "task_1")

    assert any("duplicate paragraph_id 11 in source" in error for error in errors)


def test_task2_rejects_only_exact_same_label_span_duplicates():
    record = {
        "paragraph_id": 1,
        "text": "abcdefghij",
        "type": "editorial",
        "labels": [
            {"label": "AS", "start_offset": 0, "end_offset": 8},
            {"label": "AS", "start_offset": 2, "end_offset": 6},  # same-label nesting
            {"label": "OT", "start_offset": 0, "end_offset": 8},  # cross-label identical
            {"label": "TE", "start_offset": 1, "end_offset": 7},  # cross-label nesting
            {"label": "AS", "start_offset": 0, "end_offset": 8},  # exact duplicate
        ],
    }

    errors = validate_task2_records([record])

    duplicate_errors = [error for error in errors if "duplicate same-label span" in error]
    assert duplicate_errors == ["record 1, span 5: duplicate same-label span 'AS' [0, 8)"]


@pytest.mark.parametrize(
    ("start", "end"),
    [(False, 4), (0, True)],
)
def test_task2_rejects_boolean_offsets(start, end):
    records = [
        {
            "paragraph_id": 1,
            "text": "abcdefghij",
            "type": "editorial",
            "labels": [{"label": "AS", "start_offset": start, "end_offset": end}],
        }
    ]
    assert any("bad offsets" in error for error in validate_task2_records(records))


def test_source_bound_preflight_rejects_unknown_task():
    with pytest.raises(ValueError, match="task must be one of"):
        validate_records_against_source([], SOURCE, "task1")
