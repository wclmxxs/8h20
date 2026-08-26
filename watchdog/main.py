from __future__ import annotations

import json
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx

CONFIG_PATH = Path(os.getenv("WATCHDOG_CONFIG", "/config/instances.json"))
STATE_PATH = Path(os.getenv("WATCHDOG_STATE", "/state/status.json"))
SLOTS_ROOT = Path(os.getenv("WATCHDOG_SLOTS_ROOT", "/slots"))
API_KEY = os.getenv("API_KEY", "")
ENABLED = os.getenv("WATCHDOG_ENABLED", "1").lower() in {"1", "true", "yes", "on"}
INTERVAL_SECONDS = max(5, int(os.getenv("WATCHDOG_INTERVAL_SECONDS", "15")))
STALL_SECONDS = max(60, int(os.getenv("WATCHDOG_STALL_SECONDS", "300")))
INITIAL_GRACE_SECONDS = max(
    0, int(os.getenv("WATCHDOG_INITIAL_GRACE_SECONDS", "120"))
)
RESTART_COOLDOWN_SECONDS = max(
    60, int(os.getenv("WATCHDOG_RESTART_COOLDOWN_SECONDS", "300"))
)
JOB_WINDOW_SECONDS = max(
    STALL_SECONDS * 2, int(os.getenv("WATCHDOG_JOB_WINDOW_SECONDS", "21600"))
)
MAX_JOB_RECORDS = max(100, int(os.getenv("WATCHDOG_MAX_JOB_RECORDS", "2000")))
MAX_ACTIVE_POLLS = max(1, int(os.getenv("WATCHDOG_MAX_ACTIVE_POLLS", "128")))

ACTIVE_STATUSES = frozenset({"queued", "in_progress", "running", "processing"})
TERMINAL_STATUSES = frozenset(
    {"completed", "succeeded", "failed", "deleted", "cancelled", "not_found"}
)
STATUS_RANK = {
    "unknown": 0,
    "queued": 1,
    "in_progress": 2,
    "running": 2,
    "processing": 2,
    "completed": 3,
    "succeeded": 3,
    "failed": 3,
    "deleted": 3,
    "cancelled": 3,
    "not_found": 3,
}
# expandable_segments mapping failures are allocator warnings; a request may
# fall back and complete successfully, so only exception-level OOMs are fatal.
FATAL_OOM_PATTERN = re.compile(
    r"(?:torch\.(?:cuda\.)?OutOfMemoryError|CUDA out of memory)",
    re.IGNORECASE,
)


def write_state(payload: dict[str, Any]) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = STATE_PATH.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(STATE_PATH)


def load_config() -> dict[str, Any]:
    return json.loads(CONFIG_PATH.read_text())


def normalize_status(value: object) -> str:
    return str(value or "unknown").strip().lower()


def load_job_records(slot: int, now: float) -> list[dict[str, Any]]:
    root = SLOTS_ROOT / str(slot) / "api-data" / "jobs"
    if not root.is_dir():
        return []
    cutoff = now - JOB_WINDOW_SECONDS
    paths: list[tuple[float, Path]] = []
    for path in root.glob("*.json"):
        try:
            modified = path.stat().st_mtime
        except FileNotFoundError:
            continue
        if modified >= cutoff:
            paths.append((modified, path))
    records: list[dict[str, Any]] = []
    for _, path in sorted(paths, reverse=True)[:MAX_JOB_RECORDS]:
        try:
            metadata = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        task_id = str(metadata.get("id") or path.stem)
        observation = metadata.get("_watchdog") or {}
        records.append(
            {
                "id": task_id,
                "status": normalize_status(observation.get("status")),
                "terminal": bool(observation.get("terminal")),
                "created_at": int(metadata.get("created_at") or 0),
            }
        )
    return records


def made_processing_progress(
    previous: dict[str, str], current: dict[str, str]
) -> bool:
    """New submissions do not mask a queue that is no longer advancing."""
    for task_id, old_status in previous.items():
        new_status = current.get(task_id)
        if new_status is None:
            continue
        if STATUS_RANK.get(new_status, 0) > STATUS_RANK.get(old_status, 0):
            return True
        if old_status in ACTIVE_STATUSES and new_status in TERMINAL_STATUSES:
            return True
    return False


def fatal_oom_line(logs: str) -> str | None:
    matches = [line for line in logs.splitlines() if FATAL_OOM_PATTERN.search(line)]
    return matches[-1][-1000:] if matches else None


def active_statuses(statuses: dict[str, str]) -> dict[str, str]:
    return {
        task_id: status
        for task_id, status in statuses.items()
        if status in ACTIVE_STATUSES
    }


class SlotTracker:
    def __init__(self, now: float) -> None:
        self.started_at = now
        self.last_progress_at = now
        self.last_restart_at = 0.0
        self.last_restart_reason: str | None = None
        self.restart_count = 0
        self.statuses: dict[str, str] = {}
        self.was_healthy = False
        # Include a short history so a watchdog starting after an OOM can still
        # recover the poisoned worker, without replaying arbitrarily old logs.
        self.log_since = int(now) - 300
        self.last_oom_line: str | None = None

    def observe_health(self, healthy: bool, now: float) -> None:
        if healthy and not self.was_healthy:
            self.started_at = now
            self.last_progress_at = now
            self.statuses = {}
        self.was_healthy = healthy

    def observe_statuses(self, statuses: dict[str, str], now: float) -> None:
        if made_processing_progress(self.statuses, statuses):
            self.last_progress_at = now
        if not active_statuses(statuses):
            self.last_progress_at = now
        self.statuses = statuses

    def stall_reason(self, now: float, query_successes: int) -> str | None:
        active = active_statuses(self.statuses)
        if not active or query_successes == 0:
            return None
        if now - self.started_at < INITIAL_GRACE_SECONDS:
            return None
        stalled_for = now - self.last_progress_at
        if stalled_for < STALL_SECONDS:
            return None
        counts: dict[str, int] = {}
        for status in active.values():
            counts[status] = counts.get(status, 0) + 1
        return f"queue made no processing progress for {int(stalled_for)}s; active={counts}"

    def in_cooldown(self, now: float) -> bool:
        return bool(self.last_restart_at) and (
            now - self.last_restart_at < RESTART_COOLDOWN_SECONDS
        )

    def restarted(self, reason: str, now: float) -> None:
        self.last_restart_at = now
        self.last_restart_reason = reason
        self.restart_count += 1
        self.started_at = now
        self.last_progress_at = now
        self.statuses = {}
        self.was_healthy = False
        self.log_since = int(now) + 1


def container_health(container: Any) -> tuple[str, str]:
    container.reload()
    state = container.attrs.get("State") or {}
    health = (state.get("Health") or {}).get("Status", "none")
    return str(state.get("Status") or "unknown"), str(health)


def container_started_at(container: Any) -> int:
    value = str((container.attrs.get("State") or {}).get("StartedAt") or "")
    try:
        return int(datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp())
    except ValueError:
        return 0


def query_jobs(
    client: httpx.Client,
    instance: dict[str, Any],
    records: list[dict[str, Any]],
    known_statuses: dict[str, str] | None = None,
) -> tuple[dict[str, str], int, list[str]]:
    statuses: dict[str, str] = {}
    errors: list[str] = []
    successes = 0
    headers = {"Authorization": f"Bearer {API_KEY}"} if API_KEY else {}
    base_url = str(instance["internal_url"]).rstrip("/")
    pollable: list[dict[str, Any]] = []
    for record in records:
        task_id = str(record["id"])
        local_status = normalize_status(record.get("status"))
        known_status = normalize_status((known_statuses or {}).get(task_id))
        if local_status == "unknown" and known_status in TERMINAL_STATUSES:
            local_status = known_status
        statuses[task_id] = local_status
        if record.get("terminal") or local_status in TERMINAL_STATUSES:
            continue
        pollable.append(record)

    pollable.sort(key=lambda item: (int(item.get("created_at") or 0), str(item["id"])))
    for record in pollable[:MAX_ACTIVE_POLLS]:
        task_id = str(record["id"])
        local_status = statuses[task_id]
        try:
            response = client.get(f"{base_url}/v1/videos/{task_id}", headers=headers)
            if response.status_code == 404:
                statuses[task_id] = "not_found"
                successes += 1
                continue
            response.raise_for_status()
            statuses[task_id] = normalize_status(response.json().get("status"))
            successes += 1
        except Exception as exc:  # noqa: BLE001 - one bad task must not stop recovery
            statuses[task_id] = local_status
            errors.append(f"{task_id}: {type(exc).__name__}: {exc}")
    return statuses, successes, errors


def restart_worker(
    container: Any, tracker: SlotTracker, reason: str, now: float
) -> bool:
    if tracker.in_cooldown(now):
        return False
    print(f"watchdog restarting {container.name}: {reason}", flush=True)
    container.restart(timeout=10)
    tracker.restarted(reason, now)
    return True


def main() -> None:
    import docker

    config = load_config()
    instances = config.get("instances") or []
    docker_client = docker.from_env()
    trackers = {
        int(instance["group_index"]): SlotTracker(time.time()) for instance in instances
    }
    with httpx.Client(timeout=8) as client:
        while True:
            cycle_started = time.time()
            state: dict[str, Any] = {
                "ok": True,
                "enabled": ENABLED,
                "timestamp": int(cycle_started),
                "stall_seconds": STALL_SECONDS,
                "restart_cooldown_seconds": RESTART_COOLDOWN_SECONDS,
                "instances": [],
            }
            for instance in instances:
                slot = int(instance["group_index"])
                tracker = trackers[slot]
                detail: dict[str, Any] = {
                    "group_index": slot,
                    "container": f"minimax-h3-h20-sglang-{slot}",
                    "restarts_by_watchdog": tracker.restart_count,
                    "last_restart_at": int(tracker.last_restart_at) or None,
                    "last_restart_reason": tracker.last_restart_reason,
                }
                try:
                    container = docker_client.containers.get(detail["container"])
                    container_state, health = container_health(container)
                    healthy = container_state == "running" and health == "healthy"
                    detail.update({"container_state": container_state, "health": health})
                    tracker.observe_health(healthy, cycle_started)
                    if not ENABLED:
                        detail["active_jobs"] = 0
                        state["instances"].append(detail)
                        continue

                    if container_state == "running":
                        tracker.log_since = max(
                            tracker.log_since, container_started_at(container)
                        )
                        logs = container.logs(
                            since=tracker.log_since,
                            timestamps=True,
                            tail=10000,
                        ).decode(errors="replace")
                        tracker.log_since = int(cycle_started)
                        oom_line = fatal_oom_line(logs)
                        if oom_line and oom_line != tracker.last_oom_line:
                            reason = f"fatal CUDA OOM: {oom_line[-500:]}"
                            restarted = restart_worker(
                                container, tracker, reason, cycle_started
                            )
                            detail["restart_triggered"] = restarted
                            detail["restart_reason"] = reason
                            if restarted:
                                tracker.last_oom_line = oom_line
                            state["instances"].append(detail)
                            continue

                    if not healthy:
                        detail["active_jobs"] = 0
                        state["instances"].append(detail)
                        continue

                    records = load_job_records(slot, cycle_started)
                    statuses, successes, errors = query_jobs(
                        client,
                        instance,
                        records,
                        known_statuses=tracker.statuses,
                    )
                    tracker.observe_statuses(statuses, cycle_started)
                    active = active_statuses(statuses)
                    detail.update(
                        {
                            "tracked_jobs": len(statuses),
                            "active_jobs": len(active),
                            "active_statuses": {
                                status: list(active.values()).count(status)
                                for status in sorted(set(active.values()))
                            },
                            "query_successes": successes,
                            "query_errors": errors[:20],
                            "seconds_since_progress": int(
                                cycle_started - tracker.last_progress_at
                            ),
                        }
                    )
                    reason = tracker.stall_reason(cycle_started, successes)
                    if reason:
                        detail["restart_triggered"] = restart_worker(
                            container, tracker, reason, cycle_started
                        )
                        detail["restart_reason"] = reason
                except Exception as exc:  # noqa: BLE001 - monitor remaining slots
                    detail["error"] = f"{type(exc).__name__}: {exc}"
                    state["ok"] = False
                    print(f"watchdog slot {slot} failed: {detail['error']}", flush=True)
                state["instances"].append(detail)
            write_state(state)
            elapsed = time.time() - cycle_started
            time.sleep(max(1.0, INTERVAL_SECONDS - elapsed))


if __name__ == "__main__":
    main()
