from app import storage


def configure(monkeypatch) -> storage.TosConfig:
    values = {
        "TOS_BUCKET": "capcut-end-cloud-integration-sg",
        "TOS_ACCESS_KEY": "test-ak",
        "TOS_SECRET_KEY": "test-sk",
        "TOS_KEY_PREFIX": "minimax_h3_data_cache/outputs",
        "TOS_PUBLIC_BASE_URL": (
            "https://tosv.byted.org/obj/capcut-end-cloud-integration-sg"
        ),
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


def test_publish_file_uploads_video_with_content_type(tmp_path, monkeypatch):
    configure(monkeypatch)
    video = tmp_path / "video.mp4"
    video.write_bytes(b"video-data")
    calls = []

    class FakeFileSystem:
        def put_file(self, source, destination, **kwargs):
            calls.append((source, destination, kwargs))

    monkeypatch.setattr(storage, "_filesystem", lambda _: FakeFileSystem())
    published = storage.publish_file(video, "video_123")

    assert calls == [
        (
            str(video),
            (
                "tos://capcut-end-cloud-integration-sg/"
                "minimax_h3_data_cache/outputs/video_123.mp4"
            ),
            {"ContentType": "video/mp4"},
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
