import json
import os
from pathlib import Path

from core.reply_response_structure import (
    MAX_DEPTH,
    MAX_FILE_BYTES,
    MAX_KEYS,
    ReplyResponseStructureRecorder,
    build_reply_response_structure,
)


def test_structure_builder_records_shape_without_business_or_secret_values(capsys):
    secrets = {
        "cookie": "COOKIE_SECRET_4b5737",
        "token": "TOKEN_SECRET_b23719",
        "authorization": "AUTH_SECRET_d778a1",
        "comment": "COMMENT_SECRET_353e3b",
        "nickname": "NICKNAME_SECRET_d7936c",
        "avatar": "https://avatar.invalid/AVATAR_SECRET_4cd630",
        "media": "https://media.invalid/MEDIA_SECRET_49b7f8",
        "html": "<html>HTML_SECRET_453d1f</html>",
        "exception": "EXCEPTION_SECRET_9d02e4",
        "dynamic_key": "DYNAMIC_SECRET_COOKIE_24d157",
    }
    payload = {
        "status_code": 0,
        "has_more": False,
        "cursor": 12,
        "cookie": secrets["cookie"],
        "token": secrets["token"],
        "authorization": secrets["authorization"],
        "status_msg": secrets["exception"],
        "data": {
            "comments": [
                {
                    "cid": "reply-1",
                    "text": secrets["comment"],
                    "nickname": secrets["nickname"],
                    "avatar_url": secrets["avatar"],
                    "media_url": secrets["media"],
                    "html": secrets["html"],
                }
            ],
            secrets["dynamic_key"]: "dynamic-value",
        },
    }

    structure = build_reply_response_structure(payload)
    serialized = json.dumps(structure, ensure_ascii=False, sort_keys=True)
    captured = capsys.readouterr()

    for secret in secrets.values():
        assert secret not in serialized
        assert secret not in captured.out
        assert secret not in captured.err
    assert structure["schema_version"] == 1
    assert structure["root_type"] == "object"
    assert {field["path"] for field in structure["fields"]} >= {
        "$",
        "$.data",
        "$.data.comments",
        "$.data.comments[0]",
    }
    comments = next(
        field for field in structure["fields"] if field["path"] == "$.data.comments"
    )
    assert comments == {
        "path": "$.data.comments",
        "type": "list",
        "length": 1,
    }
    assert structure["allowed_state"] == {
        "$.cursor": 12,
        "$.has_more": False,
        "$.status_code": 0,
    }


def test_structure_builder_applies_one_global_key_budget():
    payload = {
        "first": {f"first_key_{index}": index for index in range(40)},
        "second": {f"second_key_{index}": index for index in range(40)},
    }

    structure = build_reply_response_structure(payload)

    recorded_keys = sum(len(field.get("keys", ())) for field in structure["fields"])
    assert recorded_keys <= MAX_KEYS


def test_structure_builder_limits_depth_and_samples_only_first_list_item():
    payload = {
        "items": [
            {"level_1": {"level_2": {"level_3": {"level_4": "hidden"}}}},
            {"second_item_secret": "must-not-shape-output"},
        ]
    }

    structure = build_reply_response_structure(payload)
    paths = {field["path"] for field in structure["fields"]}

    assert "$.items[0]" in paths
    assert not any("[1]" in path for path in paths)
    assert not any(path.count(".") + path.count("[") > MAX_DEPTH for path in paths)


def test_structure_fingerprint_ignores_string_and_allowed_state_values():
    first = build_reply_response_structure(
        {"status_code": 0, "cursor": 10, "comments": [{"cid": "reply-one"}]}
    )
    second = build_reply_response_structure(
        {"status_code": 9, "cursor": 20, "comments": [{"cid": "reply-two"}]}
    )

    assert first["fingerprint"] == second["fingerprint"]
    assert first["allowed_state"] != second["allowed_state"]


async def test_recorder_deduplicates_shape_and_counts_observations(tmp_path):
    destination = tmp_path / "reply-response-structure.json"
    recorder = ReplyResponseStructureRecorder(destination)

    await recorder.capture(
        {"status_code": 0, "comments": [{"cid": "reply-one", "text": "first"}]}
    )
    await recorder.capture(
        {"status_code": 0, "comments": [{"cid": "reply-two", "text": "second"}]}
    )

    artifact = json.loads(destination.read_text(encoding="utf-8"))
    assert artifact["schema_version"] == 1
    assert artifact["capture_count"] == 2
    assert artifact["dropped_structure_count"] == 0
    assert len(artifact["structures"]) == 1
    assert artifact["structures"][0]["occurrence_count"] == 2


async def test_recorder_replaces_destination_only_after_complete_temp_write(
    tmp_path, monkeypatch
):
    destination = tmp_path / "reply-response-structure.json"
    destination.write_text('{"old":true}\n', encoding="utf-8")
    recorder = ReplyResponseStructureRecorder(destination)
    observed = {}
    real_replace = os.replace

    def observe_replace(source, target):
        observed["destination_before_replace"] = destination.read_text(encoding="utf-8")
        observed["temporary_artifact"] = json.loads(Path(source).read_text(encoding="utf-8"))
        real_replace(source, target)

    monkeypatch.setattr("core.reply_response_structure.os.replace", observe_replace)

    assert await recorder.capture({"status_code": 0, "comments": []}) is True

    assert observed["destination_before_replace"] == '{"old":true}\n'
    assert observed["temporary_artifact"]["capture_count"] == 1
    assert json.loads(destination.read_text(encoding="utf-8"))["capture_count"] == 1


async def test_recorder_bounds_unique_structures_and_file_size(tmp_path):
    destination = tmp_path / "reply-response-structure.json"
    recorder = ReplyResponseStructureRecorder(destination)

    for variant in range(12):
        payload = {
            f"variant_{variant}_field_{index}_{'x' * 32}": index
            for index in range(64)
        }
        await recorder.capture(payload)

    artifact = json.loads(destination.read_text(encoding="utf-8"))
    assert artifact["capture_count"] == 12
    assert artifact["dropped_structure_count"] > 0
    assert len(artifact["structures"]) <= 8
    assert destination.stat().st_size <= MAX_FILE_BYTES


async def test_recorder_failure_never_emits_exception_or_response_values(
    tmp_path, monkeypatch, capsys
):
    secret = "WRITE_EXCEPTION_SECRET_21dcf2"
    response_secret = "RESPONSE_SECRET_93a4eb"
    recorder = ReplyResponseStructureRecorder(tmp_path / "reply-response-structure.json")

    async def fail_write():
        raise RuntimeError(secret)

    monkeypatch.setattr(recorder, "_write", fail_write)

    assert await recorder.capture({"comments": [{"text": response_secret}]}) is False

    captured = capsys.readouterr()
    combined = captured.out + captured.err
    assert secret not in combined
    assert response_secret not in combined
