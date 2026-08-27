#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any

GPUS_PER_SERVICE = 8
GPU_MODEL_PATTERN = re.compile(r"\bH20\b", re.IGNORECASE)


def quote(value: object) -> str:
    return json.dumps(str(value))


def detect_gpus() -> list[dict[str, object]]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,uuid,name,memory.total",
        "--format=csv,noheader,nounits",
    ]
    output = subprocess.run(command, check=True, capture_output=True, text=True).stdout
    gpus: list[dict[str, object]] = []
    for line in output.splitlines():
        if not line.strip():
            continue
        index, uuid, name, memory = [part.strip() for part in line.split(",", 3)]
        gpus.append(
            {
                "index": int(index),
                "uuid": uuid,
                "name": name,
                "memory_mb": int(memory),
            }
        )
    if not gpus:
        raise SystemExit("nvidia-smi returned no GPUs")
    gpus.sort(key=lambda gpu: int(gpu["index"]))
    indexes = [gpu["index"] for gpu in gpus]
    if indexes != list(range(len(indexes))):
        raise SystemExit(f"GPU indexes must be contiguous from zero; got {indexes}")
    return gpus


def partition_gpus(
    gpus: list[dict[str, object]], allow_non_h20: bool = False
) -> list[list[dict[str, object]]]:
    if len(gpus) != GPUS_PER_SERVICE:
        raise SystemExit(
            f"exactly {GPUS_PER_SERVICE} GPUs are required for one 8-GPU service; "
            f"got {len(gpus)}"
        )
    if not allow_non_h20:
        invalid = [gpu for gpu in gpus if not GPU_MODEL_PATTERN.search(str(gpu["name"]))]
        if invalid:
            names = sorted({str(gpu["name"]) for gpu in invalid})
            raise SystemExit(f"all assigned GPUs must be H20; got {names}")
    return [gpus]


def detect_host_resources(group_count: int) -> tuple[int, int]:
    cpu_total = os.cpu_count() or group_count * 16
    memory_total_mb = 0
    meminfo = Path("/proc/meminfo")
    if meminfo.is_file():
        with meminfo.open() as source:
            for line in source:
                if line.startswith("MemTotal:"):
                    memory_total_mb = int(line.split()[1]) // 1024
                    break
    if memory_total_mb <= 0:
        memory_total_mb = group_count * 64 * 1024
    return max(4, cpu_total // group_count), max(16384, memory_total_mb // group_count)


def sglang_command() -> str:
    # Keep the self-managed Compose path and the Bernard image on exactly the
    # same launch arguments. Dollar escaping is unnecessary because this
    # wrapper contains no Compose-expanded variables.
    return "exec /opt/minimax-h3/bin/launch_sglang.sh\n"


def sglang_service(
    group_index: int,
    group: list[dict[str, object]],
    data_root: str,
    model_cache_root: str,
    sol_enabled: bool = False,
) -> list[str]:
    indexes = [int(gpu["index"]) for gpu in group]
    slot = f"{data_root}/slots/{group_index}"
    device_ids = ", ".join(quote(index) for index in indexes)
    image = "${SGLANG_SOL_IMAGE}" if sol_enabled else "${SGLANG_IMAGE}"
    service = [
        f"  h3-sglang-{group_index}:",
        f"    image: {image}",
        f"    container_name: minimax-h3-h20-sglang-{group_index}",
        "    restart: unless-stopped",
        "    init: true",
        "    ipc: host",
        "    shm_size: 64gb",
        "    env_file: ../.env",
        f'    command: ["bash", "-lc", {quote(sglang_command())}]',
        "    environment:",
        f"      NUM_GPUS: {GPUS_PER_SERVICE}",
        "      HF_HOME: /cache/huggingface",
        "      HF_HUB_CACHE: /cache/huggingface/hub",
        '      PYTORCH_CUDA_ALLOC_CONF: "${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:False}"',
        '      NCCL_P2P_DISABLE: "${NCCL_P2P_DISABLE:-0}"',
        '      NCCL_GRAPH_REGISTER: "${NCCL_GRAPH_REGISTER:-0}"',
        "      SGLANG_MINIMAX_H3_EXTRA_SHORT_EDGES: ${SHORT_EDGES:-480,704}",
    ]
    if sol_enabled:
        service.extend(
            [
                "      ATTENTION_BACKEND: sol_attn",
                '      COMPONENT_ATTENTION_BACKENDS: "${SOL_COMPONENT_ATTENTION_BACKENDS:-text_encoder=torch_sdpa,audio_vae=fa,video_vae=fa,transformer=sol_attn}"',
                '      ATTENTION_BACKEND_CONFIG: "${SOL_ATTENTION_BACKEND_CONFIG:-dense_backend=sage_attn,dense_steps=0,kv_splits=auto,tau=1.5}"',
                '      SOL_ATTN_STRICT: "${SOL_ATTN_STRICT:-1}"',
                '      WARMUP_STEPS: "${SOL_WARMUP_STEPS:-3}"',
                '      QUANTIZATION: "${SOL_QUANTIZATION:-fp8}"',
                '      ENABLE_TORCH_COMPILE: "${SOL_ENABLE_TORCH_COMPILE:-0}"',
                '      LORA_MERGE_MODE: "${SOL_LORA_MERGE_MODE:-dynamic}"',
                '      SGLANG_DIFFUSION_LORA_BEFORE_FP8: "${SOL_LORA_BEFORE_FP8:-0}"',
                '      SGLANG_DIFFUSION_LORA_MERGE_FP32: "0"',
                '      SGLANG_CACHE_DIT_ENABLED: "${SOL_CACHE_DIT_ENABLED:-true}"',
                '      SGLANG_CACHE_DIT_FN: "${SOL_CACHE_DIT_FN:-1}"',
                '      SGLANG_CACHE_DIT_BN: "${SOL_CACHE_DIT_BN:-0}"',
                '      SGLANG_CACHE_DIT_WARMUP: "${SOL_CACHE_DIT_WARMUP:-1}"',
                '      SGLANG_CACHE_DIT_RDT: "${SOL_CACHE_DIT_RDT:-0.12}"',
                '      SGLANG_CACHE_DIT_MC: "${SOL_CACHE_DIT_MC:-3}"',
            ]
        )
    service.extend(
        [
            "    volumes:",
            f"      - {model_cache_root}:/cache/huggingface",
            f"      - {slot}/output:/out/videos",
            "    healthcheck:",
            "      test: ['CMD-SHELL', 'curl -fsS http://127.0.0.1:30020/health >/dev/null']",
            "      interval: 10s",
            "      timeout: 5s",
            "      retries: 90",
            "      start_period: 120s",
            "    deploy:",
            "      resources:",
            "        reservations:",
            "          devices:",
            "            - driver: nvidia",
            f"              device_ids: [{device_ids}]",
            "              capabilities: [gpu]",
        ]
    )
    return service


def api_service(
    group_index: int,
    group: list[dict[str, object]],
    data_root: str,
    host: str,
    base_port: int,
    sol_enabled: bool = False,
) -> list[str]:
    port = base_port + group_index
    slot = f"{data_root}/slots/{group_index}"
    indexes = ",".join(str(gpu["index"]) for gpu in group)
    uuids = ",".join(str(gpu["uuid"]) for gpu in group)
    service = [
        f"  h3-api-{group_index}:",
        "    image: ${API_IMAGE}",
        f"    container_name: minimax-h3-h20-api-{group_index}",
        "    restart: unless-stopped",
        "    init: true",
        "    user: ${HOST_UID}:${HOST_GID}",
        "    env_file: ../.env",
        "    depends_on:",
        f"      h3-sglang-{group_index}:",
        "        condition: service_healthy",
        "    ports:",
        # Without HOST_IP Docker publishes on both 0.0.0.0 and [::].
        f"      - '{port}:30010'",
        "    environment:",
        f"      SGLANG_URL: http://h3-sglang-{group_index}:30020",
        "      DATA_ROOT: /data",
        f"      PUBLIC_BASE_URL: {quote(f'http://{host}:{port}')}",
        f"      GPU_GROUP_INDEX: {quote(group_index)}",
        f"      GPU_INDEXES: {quote(indexes)}",
        f"      GPU_UUIDS: {quote(uuids)}",
    ]
    if sol_enabled:
        service.extend(
            [
                "      ATTENTION_BACKEND: sol_attn",
                '      COMPONENT_ATTENTION_BACKENDS: "${SOL_COMPONENT_ATTENTION_BACKENDS:-text_encoder=torch_sdpa,audio_vae=fa,video_vae=fa,transformer=sol_attn}"',
                '      QUANTIZATION: "${SOL_QUANTIZATION:-fp8}"',
                '      ENABLE_TORCH_COMPILE: "${SOL_ENABLE_TORCH_COMPILE:-0}"',
                '      LORA_MERGE_MODE: "${SOL_LORA_MERGE_MODE:-dynamic}"',
                '      SGLANG_DIFFUSION_LORA_BEFORE_FP8: "${SOL_LORA_BEFORE_FP8:-0}"',
                '      SGLANG_DIFFUSION_LORA_MERGE_FP32: "0"',
                '      SGLANG_CACHE_DIT_ENABLED: "${SOL_CACHE_DIT_ENABLED:-true}"',
                '      SGLANG_CACHE_DIT_FN: "${SOL_CACHE_DIT_FN:-1}"',
                '      SGLANG_CACHE_DIT_BN: "${SOL_CACHE_DIT_BN:-0}"',
                '      SGLANG_CACHE_DIT_WARMUP: "${SOL_CACHE_DIT_WARMUP:-1}"',
                '      SGLANG_CACHE_DIT_RDT: "${SOL_CACHE_DIT_RDT:-0.12}"',
                '      SGLANG_CACHE_DIT_MC: "${SOL_CACHE_DIT_MC:-3}"',
            ]
        )
    service.extend(
        [
            "    volumes:",
            f"      - {slot}/api-data:/data",
            "    healthcheck:",
            '      test: [\'CMD-SHELL\', \'curl -fsS -H "Authorization: Bearer $$API_KEY" http://127.0.0.1:30010/healthz | grep -q "\\"ok\\":true"\']',
            "      interval: 10s",
            "      timeout: 5s",
            "      retries: 30",
            "      start_period: 30s",
        ]
    )
    return service


def build_config(
    groups: list[list[dict[str, object]]],
    host: str,
    instance_id: str,
    base_port: int,
    cpu_per_group: int,
    memory_per_group_mb: int,
    optimization_stack_enabled: bool = False,
) -> list[dict[str, Any]]:
    instances = []
    for group_index, group in enumerate(groups):
        instances.append(
            {
                "id": f"{instance_id}-8h20-{group_index}",
                "host": host,
                "port": base_port + group_index,
                "internal_url": f"http://h3-api-{group_index}:30010",
                "group_index": group_index,
                "gpu_indexes": [int(gpu["index"]) for gpu in group],
                "gpu_uuids": [str(gpu["uuid"]) for gpu in group],
                "gpu_names": [str(gpu["name"]) for gpu in group],
                "gpu_memory_mb": [int(gpu["memory_mb"]) for gpu in group],
                "cpu": cpu_per_group,
                "memory_mb": memory_per_group_mb,
                "attention_profile": (
                    "sol_attn" if optimization_stack_enabled else "sage_attn"
                ),
                "optimization_profile": (
                    "sol_attn_fp8_cache_dit"
                    if optimization_stack_enabled
                    else "sage_attn_bf16"
                ),
            }
        )
    return instances


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=".generated")
    parser.add_argument("--data-root", default="/srv/minimax-h3-8h20")
    parser.add_argument("--model-cache-root", default="")
    parser.add_argument("--advertise-host", required=True)
    parser.add_argument("--instance-id", required=True)
    parser.add_argument("--base-port", type=int, default=30010)
    parser.add_argument("--release-id", required=True)
    parser.add_argument("--sglang-image", default=os.getenv("SGLANG_IMAGE", ""))
    parser.add_argument("--sglang-sol-image", default=os.getenv("SGLANG_SOL_IMAGE", ""))
    parser.add_argument(
        "--optimization-stack-enabled",
        "--sol-ab-enabled",
        dest="optimization_stack_enabled",
        action="store_true",
    )
    parser.add_argument(
        "--sol-component-attention-backends",
        default=os.getenv(
            "SOL_COMPONENT_ATTENTION_BACKENDS",
            "text_encoder=torch_sdpa,audio_vae=fa,video_vae=fa,transformer=sol_attn",
        ),
    )
    parser.add_argument(
        "--sol-attention-backend-config",
        default=os.getenv(
            "SOL_ATTENTION_BACKEND_CONFIG",
            "dense_backend=sage_attn,dense_steps=0,kv_splits=auto,tau=1.5",
        ),
    )
    parser.add_argument("--api-image", default=os.getenv("API_IMAGE", ""))
    parser.add_argument("--allow-non-h20", action="store_true")
    args = parser.parse_args()
    if not args.model_cache_root:
        args.model_cache_root = f"{args.data_root}/hf-cache"

    repo_root = Path(__file__).resolve().parents[1]
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = (repo_root / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    gpus = detect_gpus()
    groups = partition_gpus(gpus, allow_non_h20=args.allow_non_h20)
    if args.optimization_stack_enabled and not args.sglang_sol_image:
        raise SystemExit(
            "--sglang-sol-image is required when the optimization stack is enabled"
        )
    cpu_per_group, memory_per_group_mb = detect_host_resources(len(groups))

    compose = ["name: minimax-h3-8h20", "", "services:"]
    for group_index, group in enumerate(groups):
        compose.extend(
            sglang_service(
                group_index,
                group,
                args.data_root,
                args.model_cache_root,
                sol_enabled=args.optimization_stack_enabled,
            )
        )
        compose.append("")
        compose.extend(
            api_service(
                group_index,
                group,
                args.data_root,
                args.advertise_host,
                args.base_port,
                sol_enabled=args.optimization_stack_enabled,
            )
        )
        compose.append("")

    compose.extend(
        [
            "  h3-cleaner:",
            "    image: ${API_IMAGE}",
            "    container_name: minimax-h3-h20-cleaner",
            "    restart: unless-stopped",
            "    init: true",
            "    env_file: ../.env",
            '    command: ["python", "-u", "-m", "app.cleanup"]',
            "    environment:",
            "      CLEANUP_ROOT: /slots",
            "      CLEANUP_STATE: /state/status.json",
            "    volumes:",
            f"      - {args.data_root}/slots:/slots",
            f"      - {args.data_root}/cleaner:/state",
            "",
            "  h3-reporter:",
            "    image: ${REPORTER_IMAGE}",
            "    container_name: minimax-h3-h20-reporter",
            "    restart: unless-stopped",
            "    init: true",
            "    user: ${HOST_UID}:${HOST_GID}",
            "    env_file: ../.env",
            "    environment:",
            "      REPORTER_CONFIG: /config/instances.json",
            "      REPORTER_STATE: /state/status.json",
            "    volumes:",
            f"      - {quote(str(output_dir / 'instances.json') + ':/config/instances.json:ro')}",
            f"      - {args.data_root}/reporter:/state",
            "",
            "  h3-watchdog:",
            "    image: ${WATCHDOG_IMAGE}",
            "    container_name: minimax-h3-h20-watchdog",
            "    restart: unless-stopped",
            "    init: true",
            "    env_file: ../.env",
            "    environment:",
            "      WATCHDOG_CONFIG: /config/instances.json",
            "      WATCHDOG_STATE: /state/status.json",
            "      WATCHDOG_SLOTS_ROOT: /slots",
            "    volumes:",
            f"      - {quote(str(output_dir / 'instances.json') + ':/config/instances.json:ro')}",
            f"      - {args.data_root}/slots:/slots:ro",
            f"      - {args.data_root}/watchdog:/state",
            "      - /var/run/docker.sock:/var/run/docker.sock",
        ]
    )

    instances = build_config(
        groups,
        args.advertise_host,
        args.instance_id,
        args.base_port,
        cpu_per_group,
        memory_per_group_mb,
        optimization_stack_enabled=args.optimization_stack_enabled,
    )
    model_lock = json.loads((repo_root / "config/models.lock.json").read_text())
    reporter_config = {
        "node": {
            "instance_id": args.instance_id,
            "host": args.advertise_host,
            "gpu_count": len(gpus),
            "assigned_gpu_count": len(groups) * GPUS_PER_SERVICE,
            "service_count": len(groups),
        },
        "deployment": {
            "release_id": args.release_id,
            "sglang_image": args.sglang_image,
            "optimization_stack": {
                "enabled": args.optimization_stack_enabled,
                "sglang_image": args.sglang_sol_image,
                "component_attention_backends": args.sol_component_attention_backends,
                "attention_backend_config": args.sol_attention_backend_config,
                "quantization": os.getenv("SOL_QUANTIZATION", "fp8"),
                "torch_compile": os.getenv("SOL_ENABLE_TORCH_COMPILE", "0"),
                "lora_merge_mode": os.getenv("SOL_LORA_MERGE_MODE", "dynamic"),
                "lora_before_fp8": os.getenv("SOL_LORA_BEFORE_FP8", "0"),
                "cache_dit": {
                    "enabled": os.getenv("SOL_CACHE_DIT_ENABLED", "true"),
                    "fn": os.getenv("SOL_CACHE_DIT_FN", "1"),
                    "bn": os.getenv("SOL_CACHE_DIT_BN", "0"),
                    "warmup": os.getenv("SOL_CACHE_DIT_WARMUP", "1"),
                    "rdt": os.getenv("SOL_CACHE_DIT_RDT", "0.12"),
                    "mc": os.getenv("SOL_CACHE_DIT_MC", "3"),
                },
            },
            "api_image": args.api_image,
            "model": model_lock,
        },
        "instances": instances,
    }
    (output_dir / "compose.yaml").write_text("\n".join(compose) + "\n")
    (output_dir / "instances.json").write_text(
        json.dumps(reporter_config, ensure_ascii=False, indent=2) + "\n"
    )
    (output_dir / "gpu-info.json").write_text(
        json.dumps(gpus, ensure_ascii=False, indent=2) + "\n"
    )
    print(f"generated one 8-H20 service from all {len(gpus)} GPUs in {output_dir}")


if __name__ == "__main__":
    main()
