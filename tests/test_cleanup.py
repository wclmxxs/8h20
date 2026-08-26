import importlib.util
import os
from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parents[1] / "api/app/cleanup.py"
SPEC = importlib.util.spec_from_file_location("h3_cleanup", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def write_at(path: Path, content: bytes, timestamp: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    os.utime(path, (timestamp, timestamp))


def test_cleanup_removes_only_expired_output_and_job_metadata(tmp_path):
    now = 1_800_000_000.0
    old = now - 12 * 3600 - 1
    fresh = now - 12 * 3600 + 1
    slot = tmp_path / "0"

    write_at(slot / "output/old.mp4", b"old-video", old)
    write_at(slot / "output/nested/fresh.mp4", b"fresh-video", fresh)
    write_at(slot / "api-data/jobs/old.json", b"{}", old)
    write_at(slot / "api-data/jobs/fresh.json", b"{}", fresh)
    write_at(slot / "api-data/keep.txt", b"not-a-job", old)

    result = MODULE.cleanup_once(tmp_path, retention_seconds=12 * 3600, now=now)

    assert result["ok"] is True
    assert result["deleted_files"] == 2
    assert result["deleted_bytes"] == len(b"old-video") + len(b"{}")
    assert not (slot / "output/old.mp4").exists()
    assert (slot / "output/nested/fresh.mp4").is_file()
    assert not (slot / "api-data/jobs/old.json").exists()
    assert (slot / "api-data/jobs/fresh.json").is_file()
    assert (slot / "api-data/keep.txt").is_file()
