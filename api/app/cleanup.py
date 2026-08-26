from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

CLEANUP_ROOT = Path(os.getenv("CLEANUP_ROOT", "/slots")).resolve()
STATE_PATH = Path(os.getenv("CLEANUP_STATE", "/state/status.json")).resolve()
RETENTION_SECONDS = max(
    3600, int(float(os.getenv("VIDEO_RETENTION_HOURS", "12")) * 3600)
)
INTERVAL_SECONDS = max(60, int(os.getenv("CLEANUP_INTERVAL_SECONDS", "600")))


def write_state(payload: dict[str, Any]) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = STATE_PATH.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(STATE_PATH)


def expired(path: Path, cutoff: float) -> bool:
    try:
        return path.is_file() and path.stat().st_mtime < cutoff
    except FileNotFoundError:
        return False


def remove_empty_directories(root: Path) -> None:
    if not root.is_dir():
        return
    directories = sorted(
        (path for path in root.rglob("*") if path.is_dir()),
        key=lambda path: len(path.parts),
        reverse=True,
    )
    for directory in directories:
        try:
            directory.rmdir()
        except (FileNotFoundError, OSError):
            pass


def cleanup_once(
    root: Path = CLEANUP_ROOT,
    retention_seconds: int = RETENTION_SECONDS,
    now: float | None = None,
) -> dict[str, Any]:
    started = time.time() if now is None else now
    cutoff = started - retention_seconds
    deleted_files = 0
    deleted_bytes = 0
    errors: list[str] = []

    if root.is_dir():
        for slot in sorted(path for path in root.iterdir() if path.is_dir()):
            targets = [slot / "output", slot / "api-data" / "jobs"]
            for target in targets:
                if not target.is_dir():
                    continue
                for path in list(target.rglob("*")):
                    if not expired(path, cutoff):
                        continue
                    try:
                        size = path.stat().st_size
                        path.unlink()
                        deleted_files += 1
                        deleted_bytes += size
                    except FileNotFoundError:
                        continue
                    except OSError as exc:
                        errors.append(f"{path}: {exc}")
                remove_empty_directories(target)

    return {
        "ok": not errors,
        "timestamp": int(started),
        "retention_hours": retention_seconds / 3600,
        "cutoff_timestamp": int(cutoff),
        "deleted_files": deleted_files,
        "deleted_bytes": deleted_bytes,
        "errors": errors[:100],
    }


def main() -> None:
    while True:
        try:
            result = cleanup_once()
        except Exception as exc:  # noqa: BLE001 - cleaner must remain alive
            result = {
                "ok": False,
                "timestamp": int(time.time()),
                "retention_hours": RETENTION_SECONDS / 3600,
                "deleted_files": 0,
                "deleted_bytes": 0,
                "errors": [f"{type(exc).__name__}: {exc}"],
            }
        write_state(result)
        print(
            f"cleanup: ok={result['ok']} deleted={result['deleted_files']} "
            f"bytes={result['deleted_bytes']} errors={len(result['errors'])}",
            flush=True,
        )
        time.sleep(INTERVAL_SECONDS)


if __name__ == "__main__":
    main()
