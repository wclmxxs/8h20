import pytest
from app import main
from fastapi import HTTPException


def test_job_path_rejects_path_traversal(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "JOB_ROOT", tmp_path)
    with pytest.raises(HTTPException) as raised:
        main.job_file("../escape")
    assert raised.value.status_code == 400


def test_metadata_round_trip(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "JOB_ROOT", tmp_path)
    main.save_metadata("video_123", {"id": "video_123", "business": {"nfe": 6}})
    assert main.load_metadata("video_123") == {
        "id": "video_123",
        "business": {"nfe": 6},
    }


def test_job_status_is_only_written_on_transition(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "JOB_ROOT", tmp_path)
    metadata = {"id": "video_123", "created_at": 100}
    main.save_metadata("video_123", metadata)

    observed = main.record_job_status(
        "video_123", "queued", metadata=metadata, now=100
    )
    assert observed["_watchdog"] == {
        "status": "queued",
        "status_changed_at": 100,
        "terminal": False,
    }
    unchanged = main.record_job_status(
        "video_123", "queued", metadata=observed, now=200
    )
    assert unchanged["_watchdog"]["status_changed_at"] == 100
    completed = main.record_job_status(
        "video_123", "completed", metadata=unchanged, now=300
    )
    assert completed["_watchdog"]["terminal"] is True
    assert completed["_watchdog"]["status_changed_at"] == 300


def test_oldest_queued_job_is_exposed_as_running(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "JOB_ROOT", tmp_path)
    first = {"id": "first", "created_at": 100}
    second = {"id": "second", "created_at": 200}
    main.save_metadata("first", first)
    main.record_job_status("first", "queued", metadata=first, now=100)
    main.save_metadata("second", second)
    second = main.record_job_status("second", "queued", metadata=second, now=200)

    second_status, _ = main.effective_job_status("second", "queued", second)
    assert second_status == "queued"

    first = main.load_metadata("first")
    assert first is not None
    first_status, first = main.effective_job_status("first", "queued", first)
    assert first_status == "running"
    assert first["_watchdog"]["status"] == "running"


def test_next_queued_job_runs_after_head_is_terminal(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "JOB_ROOT", tmp_path)
    for task_id, created_at in (("first", 100), ("second", 200)):
        metadata = {"id": task_id, "created_at": created_at}
        main.save_metadata(task_id, metadata)
        main.record_job_status(
            task_id, "queued", metadata=metadata, now=created_at
        )

    first = main.load_metadata("first")
    assert first is not None
    status, _ = main.effective_job_status("first", "succeeded", first)
    assert status == "succeeded"

    second = main.load_metadata("second")
    assert second is not None
    status, _ = main.effective_job_status("second", "queued", second)
    assert status == "running"
