import asyncio
from pathlib import Path
from unittest.mock import AsyncMock

from app import main, storage


def configure(monkeypatch) -> storage.TosConfig:
    values = {
        "TOS_BUCKET": "capcut-end-cloud-integration-sg",
        "TOS_ACCESS_KEY": "test-ak",
        "TOS_KEY_PREFIX": "minimax_h3_data_cache/outputs",
        "TOS_PUBLIC_BASE_URL": (
            "https://tosv.byted.org/obj/capcut-end-cloud-integration-sg"
        ),
        "TOS_IDC": "sg1",
    }
    for name, value in values.items():
        monkeypatch.setenv(name, value)
    return storage.TosConfig.from_env()


def test_tos_url_matches_existing_bucket_contract(monkeypatch):
    config = configure(monkeypatch)

    assert config.enabled is True
    key = config.object_key("074f1ad1-aed2-49e0-9c56-c85b05250db7")
    assert key == (
        "minimax_h3_data_cache/outputs/074f1ad1-aed2-49e0-9c56-c85b05250db7.mp4"
    )
    assert config.public_url(key) == (
        "https://tosv.byted.org/obj/capcut-end-cloud-integration-sg/"
        "minimax_h3_data_cache/outputs/"
        "074f1ad1-aed2-49e0-9c56-c85b05250db7.mp4"
    )


def test_publish_file_uses_legacy_tos_client(tmp_path, monkeypatch):
    configure(monkeypatch)
    video = tmp_path / "video.mp4"
    video.write_bytes(b"video-data")
    calls = []

    class FakeResponse:
        status_code = 200

    class FakeClient:
        def put_object(self, object_key, content):
            calls.append((object_key, content.read()))
            return FakeResponse()

    monkeypatch.setattr(storage, "_client", lambda _: FakeClient())
    published = storage.publish_file(video, "video_123")

    assert calls == [
        (
            "minimax_h3_data_cache/outputs/video_123.mp4",
            b"video-data",
        )
    ]
    assert published["url"].endswith("/minimax_h3_data_cache/outputs/video_123.mp4")
    assert published["size"] == len(b"video-data")


def test_business_result_prefers_published_tos_url():
    from app import business

    task = business.task_payload(
        {
            "id": "video_123",
            "status": "completed",
            "created_at": 1,
            "completed_at": 2,
            "_deployment": {
                "_storage": {
                    "url": "https://tosv.byted.org/obj/bucket/output/video_123.mp4"
                }
            },
        }
    )

    assert task["content"]["url"] == (
        "https://tosv.byted.org/obj/bucket/output/video_123.mp4"
    )


def test_tos_publish_prefers_shared_sglang_output(tmp_path, monkeypatch):
    task_id = "local_output_123"
    output_root = tmp_path / "videos"
    output_root.mkdir()
    output = output_root / f"{task_id}.mp4"
    output.write_bytes(b"generated-video")
    uploaded_paths: list[Path] = []

    def publish_file(path: Path, published_task_id: str):
        assert published_task_id == task_id
        uploaded_paths.append(path)
        return {"url": "https://tosv.byted.org/obj/bucket/local.mp4"}

    download = AsyncMock()
    monkeypatch.setattr(main, "SGLANG_OUTPUT_ROOT", output_root)
    monkeypatch.setattr(main, "JOB_ROOT", tmp_path / "jobs")
    monkeypatch.setattr(main, "download_upstream_content", download)
    monkeypatch.setattr(main.storage, "publish_file", publish_file)
    main.UPLOAD_LOCKS.pop(task_id, None)

    result = asyncio.run(main.ensure_tos_output(task_id, {"id": task_id}))

    assert uploaded_paths == [output]
    download.assert_not_awaited()
    assert result["_storage"]["source"] == "local_sglang_output"
    assert result["_storage"]["upload_time_s"] >= 0


def test_tos_publish_falls_back_to_upstream_http(tmp_path, monkeypatch):
    task_id = "remote_output_123"
    uploaded_content = []

    async def download(_task_id: str, destination: Path):
        assert _task_id == task_id
        destination.write_bytes(b"downloaded-video")

    def publish_file(path: Path, published_task_id: str):
        assert published_task_id == task_id
        uploaded_content.append(path.read_bytes())
        return {"url": "https://tosv.byted.org/obj/bucket/remote.mp4"}

    monkeypatch.setattr(main, "SGLANG_OUTPUT_ROOT", tmp_path / "missing")
    monkeypatch.setattr(main, "JOB_ROOT", tmp_path / "jobs")
    monkeypatch.setattr(main, "UPLOAD_TMP_ROOT", tmp_path / "upload-tmp")
    monkeypatch.setattr(main, "download_upstream_content", download)
    monkeypatch.setattr(main.storage, "publish_file", publish_file)
    main.UPLOAD_LOCKS.pop(task_id, None)

    result = asyncio.run(main.ensure_tos_output(task_id, {"id": task_id}))

    assert uploaded_content == [b"downloaded-video"]
    assert result["_storage"]["source"] == "upstream_http_fallback"
    assert result["_storage"]["upload_time_s"] >= 0
    assert list((tmp_path / "upload-tmp").iterdir()) == []
