from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_bernard_image_is_self_contained_and_preserves_all_patches():
    dockerfile = (ROOT / "docker/Dockerfile.bernard").read_text()

    assert "nightly-dev-20260812-c7c03ec5@sha256:d7538b2" in dockerfile
    assert "SGLANG_EXPECTED_COMMIT=c7c03ec53b" in dockerfile
    for patch in (
        "minimax-h3-short-edge.patch",
        "minimax-h3-request-optimization.patch",
        "minimax-h3-temporal-dense-prefix.patch",
    ):
        assert patch in dockerfile
    assert "TORCH_CUDA_ARCH_LIST=9.0" in dockerfile
    assert "SageAttention.git@${SAGEATTENTION_REVISION}" in dockerfile
    assert "Sana.git@${SOL_ATTENTION_REVISION}" in dockerfile
    assert "COMPONENT_ATTENTION_BACKENDS=text_encoder=torch_sdpa" in dockerfile
    assert "QUANTIZATION=fp8" in dockerfile
    assert "SGLANG_CACHE_DIT_ENABLED=true" in dockerfile
    assert "COPY api/app /opt/minimax-h3/api/app" in dockerfile
    assert "COPY scripts/bernard_healthcheck.sh /opt/tiger/csde/healthcheck.sh" in dockerfile


def test_bernard_entrypoint_fails_closed_to_one_eight_h20_worker():
    entrypoint = (ROOT / "scripts/start_bernard.sh").read_text()

    assert "exactly 8 GPUs" in entrypoint
    assert "must expose only NVIDIA H20" in entrypoint
    assert "export NUM_GPUS=8" in entrypoint
    assert "export TP=1" in entrypoint
    assert "export ULYSSES=8" in entrypoint
    assert "GPU_INDEXES=0,1,2,3,4,5,6,7" in entrypoint
    assert "/opt/minimax-h3/bin/launch_sglang.sh" in entrypoint
    assert "/opt/minimax-h3/api-venv/bin/uvicorn app.server:app" in entrypoint
    assert "wait -n" in entrypoint


def test_shared_launcher_keeps_the_complete_optimization_stack():
    launcher = (ROOT / "scripts/launch_sglang.sh").read_text()

    assert "NUM_GPUS=${NUM_GPUS:-8}" in launcher
    assert "TP=${TP:-1}" in launcher
    assert "ULYSSES=${ULYSSES:-8}" in launcher
    assert "transformer=sol_attn" in launcher
    assert "SOL_QUANTIZATION:-fp8" in launcher
    assert "SOL_LORA_MERGE_MODE:-dynamic" in launcher
    assert "SOL_CACHE_DIT_ENABLED:-true" in launcher
    assert "SOL_CACHE_DIT_RDT:-0.12" in launcher
    assert "--attention-backend-config" in launcher
    assert "--warmup-resolutions" in launcher
