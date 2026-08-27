from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_sol_overlay_is_pinned_on_top_of_the_sage_capable_image():
    dockerfile = (ROOT / "docker/Dockerfile.sol-attn").read_text()
    assert "ARG SGLANG_BASE_IMAGE=" in dockerfile
    assert "5fe5febdf0f59fee1c0b44a5ce6665df0dabd247" in dockerfile
    assert "#subdirectory=techniques/sparse_backends" in dockerfile
    assert "import sol_attn" in dockerfile


def test_sglang_base_image_is_pinned_to_patch_commit():
    dockerfile = (ROOT / "docker/Dockerfile.sglang").read_text()
    env = (ROOT / "config/env.example").read_text()

    assert "nightly-dev-20260812-c7c03ec5@sha256:d7538b2" in dockerfile
    assert "SGLANG_EXPECTED_COMMIT=c7c03ec53b" in dockerfile
    assert 'git rev-parse --short=10 HEAD' in dockerfile
    assert "SGLANG_BASE_IMAGE=lmsysorg/sglang:dev" not in env


def test_request_optimization_patch_is_applied_to_sglang_image():
    dockerfile = (ROOT / "docker/Dockerfile.sglang").read_text()
    patch = (ROOT / "patches/minimax-h3-request-optimization.patch").read_text()
    temporal_patch = (
        ROOT / "patches/minimax-h3-temporal-dense-prefix.patch"
    ).read_text()
    static_lora_patch = (
        ROOT / "patches/minimax-h3-static-lora-before-fp8.patch"
    ).read_text()

    assert "minimax-h3-request-optimization.patch" in dockerfile
    assert "minimax-h3-temporal-dense-prefix.patch" in dockerfile
    assert "minimax-h3-static-lora-before-fp8.patch" in dockerfile
    assert "minimax_h3_optimization" in patch
    assert "request_sol_attn_config" in patch
    assert "max_continuous_cached_steps" in patch
    assert "current_signature != desired_signature" in patch
    assert "dense_prefix_seconds" in temporal_patch
    assert 'runtime.get("sink_conditioning", "exact_kv")' in temporal_patch
    assert 'runtime["_sink_tokens"] = target_start' in temporal_patch
    assert 'sink_conditioning == "exact_kv_and_rows"' in temporal_patch
    assert 'requested_duration <= prefix_seconds' in temporal_patch
    assert 'runtime["_force_dense"] = True' in temporal_patch
    assert 'runtime["_sink_tokens"] = dense_prefix_rows' in temporal_patch
    assert "_dense_prefix_varlen" in temporal_patch
    assert "_finalize_deferred_fp8_after_startup_lora" in static_lora_patch
    assert "SGLANG_DIFFUSION_LORA_BEFORE_FP8" in static_lora_patch
    assert "process_weights_after_loading(layer)" in static_lora_patch
    assert "startup LoRA must be fully merged before FP8" in static_lora_patch


def test_optimization_toggle_reinstalls_the_single_worker():
    script = (ROOT / "scripts/configure_sol_ab.sh").read_text()
    assert "set_env OPTIMIZATION_STACK_ENABLED 1" in script
    assert "set_env OPTIMIZATION_STACK_ENABLED 0" in script
    assert "on the single 8-H20 worker" in script
    assert "exec ./install.sh" in script


def test_sol_stack_verifier_fails_closed_on_all_three_optimizations():
    script = (ROOT / "scripts/verify_sol_stack.sh").read_text()
    assert "Using sol_attn attention backend" in script
    assert "--component-attention-backends" in script
    assert "parsed server_args are missing" in script
    assert '\"audio_vae\": \"fa\"' in script
    assert '\"video_vae\": \"fa\"' in script
    assert 'required_env QUANTIZATION "${SOL_QUANTIZATION}"' in script
    assert 'required_env ENABLE_TORCH_COMPILE "${SOL_ENABLE_TORCH_COMPILE}"' in script
    assert 'required_env LORA_MERGE_MODE "${SOL_LORA_MERGE_MODE}"' in script
    assert 'required_env SGLANG_DIFFUSION_LORA_BEFORE_FP8 "${SOL_LORA_BEFORE_FP8}"' in script
    assert 'required_env SGLANG_CACHE_DIT_ENABLED "${SOL_CACHE_DIT_ENABLED}"' in script
    assert 'required_env NUM_GPUS "8"' in script
    assert '"--model-type diffusion" in command' in script
    assert '"--num-gpus 8" in command' in script
    assert '"--tp-size 1" in command' in script
    assert '"--ulysses-degree 8" in command' in script
    assert "f\"--quantization {os.environ['QUANTIZATION']}\"" in script
    assert '"--enable-torch-compile" in command' in script
    assert "f\"--lora-merge-mode {os.environ['LORA_MERGE_MODE']}\"" in script
    assert "import cache_dit" in script
    assert "Fp8Config.get_name()" in script


def test_install_preserves_tuned_defaults_and_migrates_the_h200_identity():
    script = (ROOT / "install.sh").read_text()
    assert (
        "migrate_env_default SOL_ATTENTION_BACKEND_CONFIG "
        "dense_backend=sage_attn,dense_steps=2,kv_splits=auto,tau=1.0 "
        "dense_backend=sage_attn,dense_steps=1,kv_splits=auto,tau=1.25"
    ) in script
    assert "migrate_env_default SOL_CACHE_DIT_WARMUP 2 1" in script
    assert "migrate_env_default SOL_CACHE_DIT_RDT 0.04 0.08" in script
    assert "migrate_env_default SOL_CACHE_DIT_MC 1 2" in script
    assert (
        "migrate_env_default SOL_ATTENTION_BACKEND_CONFIG "
        "dense_backend=sage_attn,dense_steps=1,kv_splits=auto,tau=1.25 "
        "dense_backend=sage_attn,dense_steps=0,kv_splits=auto,tau=1.5"
    ) in script
    assert "migrate_env_default SOL_CACHE_DIT_RDT 0.08 0.12" in script
    assert "migrate_env_default SOL_CACHE_DIT_MC 2 3" in script
    assert "migrate_env_default SOL_LORA_MERGE_MODE dynamic merge" in script
    assert 'sglang_build_base_image} == "lmsysorg/sglang:dev"' in script
    assert "build_gpu_image docker/Dockerfile.sglang" in script
    assert "REBUILD_GPU_IMAGES=1 to rebuild" in script
    assert (
        "migrate_env_default SERVICE_ID "
        "Minimax-H3-AWS-H200 Minimax-H3-Lora-H20"
    ) in script
    assert (
        "migrate_env_default ULYSSES 4 8"
    ) in script
    assert (
        "migrate_env_default SGLANG_IMAGE "
        "minimax-h3-h200-sglang:20260824-v7 "
        "minimax-h3-h20-sglang:20260826-v1"
    ) in script
    assert (
        "migrate_env_default SGLANG_SOL_IMAGE "
        "minimax-h3-h200-sglang-sol:20260824-v4 "
        "minimax-h3-h20-sglang-sol:20260826-v1"
    ) in script
    assert 'if [[ ${TP} != "1" || ${ULYSSES} != "8" ]]' in script
    assert "exactly 8 GPUs are required" in script


def test_sol_toggle_wrappers_are_one_command_entrypoints():
    enable = (ROOT / "enable_sol_ab.sh").read_text()
    disable = (ROOT / "disable_sol_ab.sh").read_text()
    assert 'configure_sol_ab.sh" enable' in enable
    assert 'configure_sol_ab.sh" disable' in disable
