from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_bernard_image_is_self_contained_and_preserves_all_patches():
    dockerfile = (ROOT / "docker/Dockerfile.bernard").read_text()

    assert "nightly-dev-20260812-c7c03ec5@sha256:d7538b2" in dockerfile
    assert "CSDE_TOOLING_IMAGE=aliyun-sin-hub.byted.org/base/csde@sha256:8b5d467" in dockerfile
    assert "COPY --from=csde-tooling /opt/tiger/hdfs_client /opt/tiger/hdfs_client" in dockerfile
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
    assert 'MODEL=""' in dockerfile
    assert "CSDE_MODEL_ROOT=/opt/tiger/csde/MiniMax-H3" in dockerfile
    assert 'resolved == "MiniMaxH3Pipeline"' in dockerfile
    assert "default_model" not in dockerfile
    assert "LORA_LOCAL_PATH=/cache/huggingface" in dockerfile
    assert "download_lora.py" in dockerfile
    assert "LORA_SHA256" in dockerfile
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
    assert 'HDFS_BIN=${HDFS_BIN:-/opt/tiger/hdfs_client/bin/hdfs}' in entrypoint
    assert '"${HDFS_BIN}" get -s -c 128 --ct 32 -t 8' in entrypoint
    assert "Refusing to manage unsafe CSDE_MODEL_ROOT" in entrypoint
    assert "model_path_identifies_minimax_h3" in entrypoint
    assert "CSDE_MODEL_ROOT must retain the MiniMax-H3 model identity" in entrypoint
    assert "default_model" not in entrypoint
    assert "model_is_complete" in entrypoint
    assert "prepare_model" in entrypoint
    assert "/opt/minimax-h3/bin/launch_sglang.sh" in entrypoint
    assert "/opt/minimax-h3/api-venv/bin/python -m uvicorn app.server:app" in entrypoint
    assert "/opt/minimax-h3/api-venv/bin/uvicorn app.server:app" not in entrypoint
    assert "wait -n" in entrypoint


def test_bernard_image_verifies_the_api_module_entrypoint():
    dockerfile = (ROOT / "docker/Dockerfile.bernard").read_text()

    assert "import fastapi, httpx, pydantic, uvicorn" in dockerfile
    assert "/opt/minimax-h3/api-venv/bin/python -m uvicorn --version" in dockerfile


def test_shared_launcher_keeps_the_complete_optimization_stack():
    launcher = (ROOT / "scripts/launch_sglang.sh").read_text()

    assert "NUM_GPUS=${NUM_GPUS:-8}" in launcher
    assert "TP=${TP:-1}" in launcher
    assert "ULYSSES=${ULYSSES:-8}" in launcher
    assert "--model-type diffusion" in launcher
    assert "transformer=sol_attn" in launcher
    assert 'args+=(--attention-backend "${ATTENTION_BACKEND}")' not in launcher
    assert "SOL_QUANTIZATION:-fp8" in launcher
    assert "SOL_LORA_MERGE_MODE:-dynamic" in launcher
    assert "SOL_CACHE_DIT_ENABLED:-true" in launcher
    assert "SOL_CACHE_DIT_RDT:-0.12" in launcher
    assert "--attention-backend-config" in launcher
    assert "--warmup-resolutions" in launcher


def test_shared_launcher_reuses_the_csde_localized_model():
    launcher = (ROOT / "scripts/launch_sglang.sh").read_text()

    assert "CSDE_MODEL_ROOT=${CSDE_MODEL_ROOT:-/opt/tiger/csde/MiniMax-H3}" in launcher
    assert "-n ${MODEL_PATH:-} && -d ${MODEL_PATH:-}" in launcher
    assert "MODEL=${CSDE_MODEL_ROOT}" in launcher
    assert "MODEL=MiniMaxAI/MiniMax-H3" in launcher
    for required_entry in (
        "modular_model_index.json",
        "FL2VA",
        "audio_vae",
        "processor",
        "scheduler",
        "text_encoder",
        "tokenizer",
        "transformer",
        "vae",
    ):
        assert required_entry in launcher
    assert "local MiniMax H3 model is incomplete" in launcher
    assert "local model path must retain the MiniMax-H3 identity" in launcher
    assert "default_model" not in launcher


def test_shared_launcher_forces_the_multimodal_dispatcher_for_local_model():
    launcher = (ROOT / "scripts/launch_sglang.sh").read_text()

    assert "sglang serve" in launcher
    assert "--model-type diffusion" in launcher
    assert launcher.index("--model-type diffusion") < launcher.index(
        '--model-path "${MODEL}"'
    )
