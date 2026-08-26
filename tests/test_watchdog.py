import importlib.util
from pathlib import Path

import httpx

MODULE_PATH = Path(__file__).resolve().parents[1] / "watchdog/main.py"
SPEC = importlib.util.spec_from_file_location("h3_h20_watchdog", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_new_queued_jobs_do_not_count_as_processing_progress():
    previous = {"old": "queued"}
    current = {"old": "queued", "new": "queued"}

    assert MODULE.made_processing_progress(previous, current) is False


def test_status_advancement_counts_as_processing_progress():
    assert MODULE.made_processing_progress(
        {"job": "queued"}, {"job": "running"}
    )
    assert MODULE.made_processing_progress(
        {"job": "queued"}, {"job": "succeeded"}
    )


def test_stalled_active_queue_requests_restart(monkeypatch):
    monkeypatch.setattr(MODULE, "STALL_SECONDS", 300)
    monkeypatch.setattr(MODULE, "INITIAL_GRACE_SECONDS", 120)
    tracker = MODULE.SlotTracker(now=1000)
    tracker.observe_health(True, now=1000)
    tracker.observe_statuses({"job": "queued"}, now=1000)

    assert tracker.stall_reason(now=1299, query_successes=1) is None
    assert "no processing progress" in tracker.stall_reason(
        now=1300, query_successes=1
    )


def test_no_active_jobs_never_requests_restart(monkeypatch):
    monkeypatch.setattr(MODULE, "STALL_SECONDS", 60)
    monkeypatch.setattr(MODULE, "INITIAL_GRACE_SECONDS", 0)
    tracker = MODULE.SlotTracker(now=1000)
    tracker.observe_health(True, now=1000)
    tracker.observe_statuses({"job": "succeeded"}, now=1000)

    assert tracker.stall_reason(now=2000, query_successes=1) is None


def test_fatal_oom_detection_ignores_nonfatal_allocator_warning():
    assert MODULE.fatal_oom_line("allocator cache is enabled") is None
    assert MODULE.fatal_oom_line(
        "[rank1]:[W824 CUDACachingAllocator.cpp:508] expandable_segments: "
        "memory mapping failed with OOM on device 1 while trying to map "
        "20971520 bytes (free: 15728640, total: 150111977472)."
    ) is None


def test_fatal_oom_detection_keeps_real_cuda_oom():
    assert "CUDA out of memory" in MODULE.fatal_oom_line(
        "2026-08-21 worker failed: CUDA out of memory"
    )
    assert "torch.OutOfMemoryError" in MODULE.fatal_oom_line(
        "torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 20 MiB"
    )


def test_query_jobs_uses_internal_api_and_marks_404_not_found():
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(str(request.url))
        if request.url.path.endswith("/missing"):
            return httpx.Response(404, json={"detail": "not found"})
        return httpx.Response(200, json={"id": "queued", "status": "queued"})

    records = [
        {
            "id": "queued",
            "status": "queued",
            "terminal": False,
            "created_at": 1,
        },
        {
            "id": "done",
            "status": "succeeded",
            "terminal": True,
            "created_at": 2,
        },
        {
            "id": "missing",
            "status": "queued",
            "terminal": False,
            "created_at": 3,
        },
    ]
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        statuses, successes, errors = MODULE.query_jobs(
            client,
            {"internal_url": "http://h3-api-0:30010"},
            records,
        )

    assert requests == [
        "http://h3-api-0:30010/v1/videos/queued",
        "http://h3-api-0:30010/v1/videos/missing",
    ]
    assert statuses == {
        "queued": "queued",
        "done": "succeeded",
        "missing": "not_found",
    }
    assert successes == 2
    assert errors == []
