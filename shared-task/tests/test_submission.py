import zipfile

import pytest

from daleel.io import write_jsonl
from daleel.submission import (
    package_submission,
    validate_task1_records,
    validate_task2_records,
)

VALID = [
    {"paragraph_id": 1, "text": "نص", "type": "debate", "labels": ["AS", "OT"]},
    {"paragraph_id": 2, "text": "نص آخر", "type": "editorial", "labels": ["CO"]},
]

VALID_T2 = [
    {
        "paragraph_id": 1,
        "text": "نص طويل بما يكفي",
        "type": "debate",
        "labels": [
            {"label": "AS", "start_offset": 0, "end_offset": 6},
            {"label": "OT", "start_offset": 3, "end_offset": 9},  # overlap is legal
        ],
    },
]


def test_valid_records_pass():
    assert validate_task1_records(VALID) == []


def test_empty_labels_list_is_valid():
    # 60/612 official training paragraphs have no ADU label.
    records = [{"paragraph_id": 1, "text": "نص", "type": "debate", "labels": []}]
    assert validate_task1_records(records) == []


def test_missing_text_and_type_caught():
    errors = validate_task1_records([{"paragraph_id": 1, "labels": ["AS"]}])
    assert any("'text'" in e for e in errors)
    assert any("'type'" in e for e in errors)


def test_valid_task2_records_pass():
    assert validate_task2_records(VALID_T2) == []


def test_task2_bad_span_caught():
    records = [
        {
            "paragraph_id": 1,
            "text": "قصير",
            "type": "editorial",
            "labels": [
                {"label": "XX", "start_offset": 0, "end_offset": 2},
                {"label": "AS", "start_offset": 3, "end_offset": 3},
                {"label": "AS", "start_offset": 0, "end_offset": 99},
                {"start_offset": 0, "end_offset": 2},
            ],
        }
    ]
    errors = validate_task2_records(records)
    assert any("unknown label" in e for e in errors)
    assert any("bad offsets" in e for e in errors)
    assert any("beyond text length" in e for e in errors)
    assert any("needs keys" in e for e in errors)


def test_bad_label_and_duplicate_id_caught():
    records = [
        {"paragraph_id": 1, "labels": ["XX"]},
        {"paragraph_id": 1, "labels": "AS"},
    ]
    errors = validate_task1_records(records)
    assert any("unknown labels" in e for e in errors)
    assert any("duplicate paragraph_id" in e for e in errors)
    assert any("not a list" in e for e in errors)


def test_package_submission_roundtrip(tmp_path):
    jsonl = tmp_path / "preds.jsonl"
    source = tmp_path / "source.jsonl"
    write_jsonl(jsonl, VALID)
    write_jsonl(source, [{key: row[key] for key in ("paragraph_id", "text", "type")} for row in VALID])
    zip_path = package_submission(
        jsonl, "TeamName", "editorial", tmp_path / "out", source_path=source
    )
    assert zip_path.name == "TeamName_editorial.zip"
    assert zip_path.parent.name == "task_1"
    assert zip_path.with_suffix(".receipt.json").is_file()
    with zipfile.ZipFile(zip_path) as zf:
        assert zf.namelist() == ["task_1.jsonl"]


def test_package_rejects_bad_setting_and_team(tmp_path):
    jsonl = tmp_path / "preds.jsonl"
    source = tmp_path / "source.jsonl"
    write_jsonl(jsonl, VALID)
    write_jsonl(source, [{key: row[key] for key in ("paragraph_id", "text", "type")} for row in VALID])
    with pytest.raises(ValueError, match="training_setting"):
        package_submission(jsonl, "TeamName", "editorials", tmp_path, source_path=source)
    with pytest.raises(ValueError, match="ambiguous"):
        package_submission(jsonl, "Team_Name", "editorial", tmp_path, source_path=source)


def test_package_rejects_invalid_records(tmp_path):
    jsonl = tmp_path / "preds.jsonl"
    source = tmp_path / "source.jsonl"
    write_jsonl(jsonl, [{"paragraph_id": 1, "labels": ["XX"]}])
    write_jsonl(source, [{"paragraph_id": 1, "text": "نص", "type": "debate"}])
    with pytest.raises(ValueError, match="unknown labels"):
        package_submission(jsonl, "TeamName", "both", tmp_path, source_path=source)


def test_package_task2_roundtrip(tmp_path):
    jsonl = tmp_path / "preds.jsonl"
    source = tmp_path / "source.jsonl"
    write_jsonl(jsonl, VALID_T2)
    write_jsonl(source, [{key: VALID_T2[0][key] for key in ("paragraph_id", "text", "type")}])
    zip_path = package_submission(
        jsonl,
        "TeamName",
        "debate",
        tmp_path / "out",
        task="task_2",
        source_path=source,
    )
    assert zip_path.name == "TeamName_debate.zip"
    assert zip_path.parent.name == "task_2"
    with zipfile.ZipFile(zip_path) as zf:
        assert zf.namelist() == ["task_2.jsonl"]


def test_package_refuses_predictions_for_a_different_source(tmp_path):
    predictions = tmp_path / "preds.jsonl"
    source = tmp_path / "source.jsonl"
    write_jsonl(predictions, VALID)
    wrong_source = [
        {"paragraph_id": 1, "text": "نص مختلف", "type": "debate"},
        {"paragraph_id": 2, "text": "نص آخر", "type": "editorial"},
    ]
    write_jsonl(source, wrong_source)
    with pytest.raises(ValueError, match="text does not exactly match source"):
        package_submission(
            predictions,
            "TeamName",
            "both",
            tmp_path / "out",
            source_path=source,
        )
