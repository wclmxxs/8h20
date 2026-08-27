import importlib.util
import json
import sys
from pathlib import Path

import pytest
import yaml

MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts/generate_compose.py"
SPEC = importlib.util.spec_from_file_location("generate_compose", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def h20s(count: int):
    return [
        {
            "index": index,
            "uuid": f"GPU-{index}",
            "name": "NVIDIA H20",
            "memory_mb": 97880,
        }
        for index in range(count)
    ]


def test_eight_h20s_form_one_group_using_every_gpu():
    groups = MODULE.partition_gpus(h20s(8))
    assert [[gpu["index"] for gpu in group] for group in groups] == [
        [0, 1, 2, 3, 4, 5, 6, 7],
    ]


@pytest.mark.parametrize("count", [4, 7, 9, 16])
def test_any_non_eight_gpu_count_fails_closed(count):
    with pytest.raises(SystemExit, match="exactly 8 GPUs"):
        MODULE.partition_gpus(h20s(count))


def test_non_h20_fails_closed():
    gpus = h20s(8)
    gpus[2]["name"] = "NVIDIA H200"
    with pytest.raises(SystemExit, match="must be H20"):
        MODULE.partition_gpus(gpus)


def test_main_renders_one_registered_service_over_all_eight_gpus(monkeypatch, tmp_path):
    monkeypatch.setattr(MODULE, "detect_gpus", lambda: h20s(8))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate_compose.py",
            "--output-dir",
            str(tmp_path),
            "--data-root",
            "/srv/h3-data",
            "--model-cache-root",
            "/mnt/model-ebs/hf-cache",
            "--advertise-host",
            "16.78.214.130",
            "--instance-id",
            "i-test",
            "--release-id",
            "release-test",
            "--sglang-image",
            "sglang:test",
            "--api-image",
            "api:test",
        ],
    )
    MODULE.main()
    compose = yaml.safe_load((tmp_path / "compose.yaml").read_text())
    config = json.loads((tmp_path / "instances.json").read_text())

    assert len(config["instances"]) == 1
    assert config["instances"][0]["port"] == 30010
    assert config["instances"][0]["host"] == "16.78.214.130"
    assert config["instances"][0]["gpu_indexes"] == list(range(8))
    assert config["instances"][0]["id"] == "i-test-8h20-0"
    assert len(compose["services"]) == 5
    reservations = compose["services"]["h3-sglang-0"]["deploy"]["resources"][
        "reservations"
    ]["devices"]
    assert reservations[0]["device_ids"] == [str(index) for index in range(8)]
    assert compose["services"]["h3-api-0"]["ports"] == ["30010:30010"]
    assert compose["services"]["h3-sglang-0"]["image"] == "${SGLANG_IMAGE}"
    assert config["instances"][0]["attention_profile"] == "sage_attn"
    assert (
        "/mnt/model-ebs/hf-cache:/cache/huggingface"
        in compose["services"]["h3-sglang-0"]["volumes"]
    )
    assert compose["services"]["h3-cleaner"]["environment"] == {
        "CLEANUP_ROOT": "/slots",
        "CLEANUP_STATE": "/state/status.json",
    }
    assert (
        compose["services"]["h3-sglang-0"]["environment"][
            "PYTORCH_CUDA_ALLOC_CONF"
        ]
        == "${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:False}"
    )
    assert compose["services"]["h3-sglang-0"]["environment"]["NCCL_P2P_DISABLE"] == "${NCCL_P2P_DISABLE:-0}"
    assert compose["services"]["h3-sglang-0"]["environment"]["NCCL_GRAPH_REGISTER"] == "${NCCL_GRAPH_REGISTER:-0}"
    watchdog = compose["services"]["h3-watchdog"]
    assert watchdog["image"] == "${WATCHDOG_IMAGE}"
    assert "/var/run/docker.sock:/var/run/docker.sock" in watchdog["volumes"]
    assert "/srv/h3-data/slots:/slots:ro" in watchdog["volumes"]


def test_sglang_command_uses_shared_eight_gpu_launcher():
    command = MODULE.sglang_command()
    assert command == "exec /opt/minimax-h3/bin/launch_sglang.sh\n"
    launcher = (MODULE_PATH.parents[1] / "scripts/launch_sglang.sh").read_text()
    assert '--num-gpus "${NUM_GPUS}"' in launcher
    assert '--tp-size "${TP}"' in launcher
    assert '--ulysses-degree "${ULYSSES}"' in launcher
    assert "--model-type diffusion" in launcher
    assert "NUM_GPUS=${NUM_GPUS:-8}" in launcher
    assert "TP=${TP:-1}" in launcher
    assert "ULYSSES=${ULYSSES:-8}" in launcher
    assert '--lora-path "${lora_path}"' in launcher
    assert '--component-attention-backends "${COMPONENT_ATTENTION_BACKENDS}"' in launcher
    assert 'args+=(--attention-backend "${ATTENTION_BACKEND}")' not in launcher
    assert 'exec "${args[@]}"' in launcher


def test_optimization_stack_applies_to_the_eight_gpu_worker(monkeypatch, tmp_path):
    monkeypatch.setattr(MODULE, "detect_gpus", lambda: h20s(8))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate_compose.py",
            "--output-dir",
            str(tmp_path),
            "--data-root",
            "/srv/h3-data",
            "--advertise-host",
            "16.78.214.130",
            "--instance-id",
            "i-test",
            "--release-id",
            "release-test",
            "--sglang-image",
            "sglang:sage",
            "--sglang-sol-image",
            "sglang:sol",
            "--api-image",
            "api:test",
            "--optimization-stack-enabled",
            "--sol-component-attention-backends",
            "text_encoder=torch_sdpa,audio_vae=fa,video_vae=fa,transformer=sol_attn",
            "--sol-attention-backend-config",
            "dense_backend=sage_attn,dense_steps=0,kv_splits=auto,tau=1.5",
        ],
    )

    MODULE.main()
    compose = yaml.safe_load((tmp_path / "compose.yaml").read_text())
    config = json.loads((tmp_path / "instances.json").read_text())

    assert compose["services"]["h3-sglang-0"]["deploy"]["resources"]["reservations"][
        "devices"
    ][0]["device_ids"] == [str(index) for index in range(8)]
    for slot in (0,):
        worker = compose["services"][f"h3-sglang-{slot}"]
        assert worker["image"] == "${SGLANG_SOL_IMAGE}"
        env = worker["environment"]
        assert env["COMPONENT_ATTENTION_BACKENDS"].endswith("transformer=sol_attn}")
        assert "audio_vae=fa" in env["COMPONENT_ATTENTION_BACKENDS"]
        assert "video_vae=fa" in env["COMPONENT_ATTENTION_BACKENDS"]
        assert "dense_steps=0" in env["ATTENTION_BACKEND_CONFIG"]
        assert "tau=1.5" in env["ATTENTION_BACKEND_CONFIG"]
        assert env["ATTENTION_BACKEND"] == "sol_attn"
        assert env["SOL_ATTN_STRICT"] == "${SOL_ATTN_STRICT:-1}"
        assert env["WARMUP_STEPS"] == "${SOL_WARMUP_STEPS:-0}"
        assert env["QUANTIZATION"] == "${SOL_QUANTIZATION:-fp8}"
        assert env["ENABLE_TORCH_COMPILE"] == "${SOL_ENABLE_TORCH_COMPILE:-0}"
        assert env["LORA_MERGE_MODE"] == "${SOL_LORA_MERGE_MODE:-dynamic}"
        assert env["SGLANG_DIFFUSION_LORA_BEFORE_FP8"] == "${SOL_LORA_BEFORE_FP8:-0}"
        assert env["SGLANG_DIFFUSION_LORA_MERGE_FP32"] == "0"
        assert env["SGLANG_CACHE_DIT_ENABLED"] == "${SOL_CACHE_DIT_ENABLED:-true}"
        assert env["SGLANG_CACHE_DIT_WARMUP"] == "${SOL_CACHE_DIT_WARMUP:-1}"
        assert env["SGLANG_CACHE_DIT_RDT"] == "${SOL_CACHE_DIT_RDT:-0.12}"
        assert env["SGLANG_CACHE_DIT_MC"] == "${SOL_CACHE_DIT_MC:-3}"
        assert (
            compose["services"][f"h3-api-{slot}"]["environment"]["ATTENTION_BACKEND"]
            == "sol_attn"
        )
        assert (
            compose["services"][f"h3-api-{slot}"]["environment"]
            ["SGLANG_DIFFUSION_LORA_MERGE_FP32"]
            == "0"
        )
    assert [item["attention_profile"] for item in config["instances"]] == ["sol_attn"]
    assert [item["optimization_profile"] for item in config["instances"]] == [
        "sol_attn_fp8_cache_dit"
    ]
    stack = config["deployment"]["optimization_stack"]
    assert stack["enabled"] is True
    assert stack["quantization"] == "fp8"
    assert stack["torch_compile"] == "0"
    assert stack["lora_merge_mode"] == "dynamic"
    assert stack["lora_before_fp8"] == "0"
    assert stack["cache_dit"] == {
        "enabled": "true",
        "fn": "1",
        "bn": "0",
        "warmup": "1",
        "rdt": "0.12",
        "mc": "3",
    }


def test_allow_non_h20_only_relaxes_model_name_not_gpu_count():
    other = h20s(8)
    for gpu in other:
        gpu["name"] = "NVIDIA test accelerator"
    assert MODULE.partition_gpus(other, allow_non_h20=True) == [other]
    with pytest.raises(SystemExit, match="exactly 8 GPUs"):
        MODULE.partition_gpus(other[:4], allow_non_h20=True)


def test_optimization_stack_supports_exactly_one_eight_gpu_group(monkeypatch, tmp_path):
    monkeypatch.setattr(MODULE, "detect_gpus", lambda: h20s(8))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate_compose.py",
            "--output-dir",
            str(tmp_path),
            "--advertise-host",
            "16.78.214.130",
            "--instance-id",
            "i-test",
            "--release-id",
            "release-test",
            "--sglang-sol-image",
            "sglang:sol",
            "--optimization-stack-enabled",
        ],
    )

    MODULE.main()
    compose = yaml.safe_load((tmp_path / "compose.yaml").read_text())
    assert compose["services"]["h3-sglang-0"]["image"] == "${SGLANG_SOL_IMAGE}"
