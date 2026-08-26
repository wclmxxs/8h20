from pathlib import Path

import pytest
from app import business
from pydantic import ValidationError


def request(**updates):
    payload = {
        "model": "MiniMax-H3",
        "content": [{"type": "text", "text": "A cinematic sunrise."}],
        "resolution": "768P",
        "duration": 5,
        "ratio": "16:9",
        "num_inference_steps": 6,
        "seed": 42,
    }
    payload.update(updates)
    return business.GenerationRequest.model_validate(payload)


def test_business_nfe_is_translated_to_sglang_sigma_grid():
    payload = business.to_upstream_request(request(num_inference_steps=6))
    assert payload["task"] == "t2va"
    assert payload["num_inference_steps"] == 7
    assert payload["target"] == {
        "short_edge": 768,
        "aspect_ratio": "16:9",
        "duration_seconds": 5.0,
    }


def test_request_optimization_is_forwarded_without_changing_defaults():
    default_payload = business.to_upstream_request(request())
    assert "optimization" not in default_payload

    payload = business.to_upstream_request(
        request(
            optimization={
                "sol_attn": {
                    "enabled": True,
                    "dense_steps": 2,
                    "tau": 1.25,
                    "sink_conditioning": "exact_kv_and_rows",
                    "dense_prefix_seconds": 3.0,
                    "future_option": "ignored",
                },
                "cache_dit": {
                    "enabled": True,
                    "warmup": 2,
                    "rdt": 0.08,
                    "max_continuous_cached_steps": 2,
                },
                "future_backend": {"enabled": True},
            }
        )
    )

    assert payload["optimization"] == {
        "sol_attn": {
            "enabled": True,
            "dense_steps": 2,
            "tau": 1.25,
            "sink_conditioning": "exact_kv_and_rows",
            "dense_prefix_seconds": 3.0,
        },
        "cache_dit": {
            "enabled": True,
            "warmup": 2,
            "rdt": 0.08,
            "max_continuous_cached_steps": 2,
        },
    }


@pytest.mark.parametrize(
    "optimization",
    [
        {"sol_attn": {"dense_steps": -1}},
        {"sol_attn": {"tau": 0}},
        {"sol_attn": {"sink_conditioning": "approximate"}},
        {"sol_attn": {"dense_prefix_seconds": -0.1}},
        {"sol_attn": {"dense_prefix_seconds": 15.1}},
        {"cache_dit": {"warmup": -1}},
        {"cache_dit": {"rdt": 1.1}},
        {"cache_dit": {"max_continuous_cached_steps": 0}},
    ],
)
def test_request_optimization_rejects_unsafe_ranges(optimization):
    with pytest.raises(ValidationError):
        request(optimization=optimization)


def test_dense_prefix_longer_than_request_is_forwarded_for_dense_runtime_fallback():
    payload = business.to_upstream_request(
        request(
            duration=5,
            optimization={
                "sol_attn": {
                    "enabled": True,
                    "dense_prefix_seconds": 8.0,
                }
            },
        )
    )

    assert payload["optimization"]["sol_attn"] == {
        "enabled": True,
        "dense_prefix_seconds": 8.0,
    }


def test_query_returns_resolved_seed_for_reproduction():
    task = business.task_payload(
        {
            "id": "video_123",
            "status": "completed",
            "created_at": 1,
            "completed_at": 2,
            "_deployment": {
                "request": {"seed": 987654321},
                "business": {
                    "resolution": "768P",
                    "duration": 5,
                    "ratio": "16:9",
                },
            },
        }
    )

    assert task["seed"] == 987654321


def test_query_returns_server_inference_time():
    task = business.task_payload(
        {
            "id": "video_123",
            "status": "completed",
            "created_at": 100,
            "completed_at": 130,
            "inference_time_s": 27.4567,
            "_deployment": {},
        }
    )

    assert task["inference_time_s"] == 27.457


def test_running_status_uses_local_transition_time():
    task = business.task_payload(
        {
            "id": "video_123",
            "status": "running",
            "created_at": 100,
            "updated_at": 100,
            "_deployment": {
                "_watchdog": {
                    "status": "running",
                    "status_changed_at": 120,
                    "terminal": False,
                }
            },
        }
    )

    assert task["status"] == "running"
    assert task["updated_at"] == 120
    assert task["inference_time_s"] is None


def test_fl2va_preserves_first_and_last_frame_order():
    value = request(
        content=[
            {"type": "text", "text": "Camera pushes in."},
            {
                "type": "image_url",
                "role": "last_frame",
                "image_url": {"url": "https://example.com/last.jpg"},
            },
            {
                "type": "image_url",
                "role": "first_frame",
                "image_url": {"url": "https://example.com/first.jpg"},
            },
        ],
        ratio="adaptive",
    )
    conditions = business.to_upstream_request(value)["conditions"]
    assert [item["frame_index"] for item in conditions] == [0, -1]


def test_reference_media_is_rejected_until_ref2va_is_deployed():
    with pytest.raises(ValidationError, match="ref2va is not deployed"):
        request(
            content=[
                {"type": "text", "text": "Animate this subject."},
                {
                    "type": "image_url",
                    "role": "reference_image",
                    "image_url": {"url": "https://example.com/ref.jpg"},
                },
            ]
        )


def test_turbo_profile_rejects_unvalidated_nfe():
    with pytest.raises(ValidationError, match="one of"):
        request(num_inference_steps=20)


def test_remote_media_host_policy_blocks_private_addresses():
    assert business.is_safe_remote_hostname("127.0.0.1") is False
    assert business.is_safe_remote_hostname("169.254.169.254") is False
    assert business.is_safe_remote_hostname("internal.byted.org") is True
    assert business.is_safe_remote_hostname("8.8.8.8") is True


def test_generation_accepts_text_role_model_alias_and_unknown_fields():
    value = request(
        model="gateway-routing-alias",
        content=[
            {
                "type": "text",
                "role": "user_prompt",
                "text": "A cinematic lake at sunrise.",
                "client_content_metadata": {"request_id": "req-1"},
            }
        ],
        client_request_metadata={"experiment": "test"},
    )

    assert value.model == "gateway-routing-alias"
    assert value.content[0].role == "user_prompt"
    assert "client_request_metadata" not in value.model_dump()
    assert "client_content_metadata" not in value.content[0].model_dump()
    upstream = business.to_upstream_request(value)
    assert upstream["model"] == business.BUSINESS_MODEL
    assert upstream["prompt"] == "A cinematic lake at sunrise."


def test_generation_ignores_unknown_media_url_fields():
    value = request(
        model="another-model-name",
        content=[
            {"type": "text", "role": "prompt", "text": "Camera pushes in."},
            {
                "type": "image_url",
                "role": "first_frame",
                "image_url": {
                    "url": "https://example.com/first.jpg",
                    "mime_type": "image/jpeg",
                },
                "unused": True,
            },
        ],
        resolution="704P",
        duration=4,
        ratio="adaptive",
        vendor_options={"unused": True},
    )

    assert value.content[1].image_url is not None
    assert value.content[1].image_url.model_dump() == {
        "url": "https://example.com/first.jpg"
    }
    assert business.to_upstream_request(value)["task"] == "fl2va"


def test_query_accepts_model_alias_and_unknown_fields():
    value = business.QueryRequest.model_validate(
        {
            "model": "client-model-alias",
            "task_id": "video_123",
            "trace_context": {"trace_id": "trace-1"},
        }
    )

    assert value.model == "client-model-alias"
    assert value.task_id == "video_123"
    assert value.model_dump() == {
        "model": "client-model-alias",
        "task_id": "video_123",
    }


def test_api_image_keeps_gateway_connections_alive():
    dockerfile = Path(__file__).parents[2] / "docker" / "Dockerfile.api"
    assert '"--timeout-keep-alive", "120"' in dockerfile.read_text()
