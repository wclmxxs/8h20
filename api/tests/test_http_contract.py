from app import main
from fastapi.testclient import TestClient


def test_missing_or_null_seed_is_randomized(monkeypatch):
    seeds = iter((123, 456))
    monkeypatch.setattr(main.secrets, "randbelow", lambda _: next(seeds))

    assert main.with_resolved_seed({"prompt": "one"})["seed"] == 123
    assert main.with_resolved_seed({"prompt": "two", "seed": None})["seed"] == 456


def test_explicit_seed_is_preserved():
    payload = {"prompt": "reproducible", "seed": 42}

    assert main.with_resolved_seed(payload)["seed"] == 42
    assert payload == {"prompt": "reproducible", "seed": 42}


def test_business_validation_error_uses_gateway_shape():
    response = TestClient(main.app).post(
        "/ic/capcut/edit_gateway/v2/video_generation",
        json={
            "model": "MiniMax-H3",
            "content": [
                {"type": "text", "text": "Animate the subject."},
                {
                    "type": "image_url",
                    "role": "reference_image",
                    "image_url": {"url": "https://example.com/ref.jpg"},
                },
            ],
            "resolution": "768P",
            "duration": 5,
            "ratio": "adaptive",
            "num_inference_steps": 6,
        },
    )
    assert response.status_code == 400
    assert response.json()["error"]["type"] == "invalid_request_error"
    assert "ref2va is not deployed" in response.json()["error"]["message"]


def test_health_endpoint_requires_configured_api_key(monkeypatch):
    monkeypatch.setattr(main, "API_KEY", "secret")
    response = TestClient(main.app).get("/healthz")
    assert response.status_code == 401
    assert response.json() == {"detail": "invalid API key"}
