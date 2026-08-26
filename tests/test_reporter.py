import importlib.util
import json
from pathlib import Path

import httpx

MODULE_PATH = Path(__file__).resolve().parents[1] / "reporter/main.py"
SPEC = importlib.util.spec_from_file_location("h3_h20_reporter", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def instance():
    return {
        "id": "i-test-8h20-0",
        "host": "16.78.214.130",
        "port": 30010,
        "internal_url": "http://h3-api-0:30010",
        "group_index": 0,
        "gpu_indexes": list(range(8)),
        "gpu_uuids": [f"GPU-{index}" for index in range(8)],
        "cpu": 128,
        "memory_mb": 1048576,
    }


def test_probe_registers_one_eight_gpu_instance():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url == "http://h3-api-0:30010/healthz"
        return httpx.Response(200, json={"ok": True, "healthy_workers": 1})

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        bernard, detail = MODULE.probe_instance(client, instance())

    assert bernard["host"] == "16.78.214.130"
    assert bernard["ports"] == [30010]
    assert bernard["healthCheckResults"] == [{"alive": True}]
    assert bernard["containerInfos"]["h3-8h20-0"]["request"]["nvidia.com/gpu"] == 8
    assert detail["gpu_indexes"] == list(range(8))


def test_catalog_uses_h20_service_id(monkeypatch):
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={"success": True})

    monkeypatch.setattr(MODULE, "CATALOG_URL", "https://gateway.test/report_catalog")
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        MODULE.report_catalog(client, [{"id": "slot-0"}])

    assert captured["body"]["service_id"] == "Minimax-H3-Lora-H20"
    assert json.loads(captured["body"]["instances_json"]) == [{"id": "slot-0"}]
