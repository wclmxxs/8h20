#!/usr/bin/env bash
set -euo pipefail

container=${1:?"Usage: $0 CONTAINER"}

required_env() {
  local key=$1 expected=$2
  if ! grep -Fx "${key}=${expected}" <<<"${worker_env}" >/dev/null; then
    echo "${container}: expected ${key}=${expected}" >&2
    exit 1
  fi
}

worker_env=$(sudo docker inspect -f '{{range .Config.Env}}{{println .}}{{end}}' "${container}")
required_env ATTENTION_BACKEND sol_attn
required_env COMPONENT_ATTENTION_BACKENDS "${SOL_COMPONENT_ATTENTION_BACKENDS}"
required_env ATTENTION_BACKEND_CONFIG "${SOL_ATTENTION_BACKEND_CONFIG}"
required_env SOL_ATTN_STRICT "${SOL_ATTN_STRICT}"
required_env WARMUP_STEPS "${SOL_WARMUP_STEPS}"
required_env QUANTIZATION "${SOL_QUANTIZATION}"
required_env ENABLE_TORCH_COMPILE "${SOL_ENABLE_TORCH_COMPILE}"
required_env LORA_MERGE_MODE "${SOL_LORA_MERGE_MODE}"
required_env SGLANG_DIFFUSION_LORA_BEFORE_FP8 "${SOL_LORA_BEFORE_FP8}"
required_env SGLANG_DIFFUSION_LORA_MERGE_FP32 "1"
required_env SGLANG_CACHE_DIT_ENABLED "${SOL_CACHE_DIT_ENABLED}"
required_env SGLANG_CACHE_DIT_FN "${SOL_CACHE_DIT_FN}"
required_env SGLANG_CACHE_DIT_BN "${SOL_CACHE_DIT_BN}"
required_env SGLANG_CACHE_DIT_WARMUP "${SOL_CACHE_DIT_WARMUP}"
required_env SGLANG_CACHE_DIT_RDT "${SOL_CACHE_DIT_RDT}"
required_env SGLANG_CACHE_DIT_MC "${SOL_CACHE_DIT_MC}"
required_env NUM_GPUS "8"

sudo docker exec -i "${container}" python3 - <<'PY'
import cache_dit
import os
import sol_attn
from pathlib import Path
from sglang.multimodal_gen import envs
from sglang.multimodal_gen.runtime.layers.quantization.fp8 import Fp8Config

assert envs.SGLANG_CACHE_DIT_ENABLED is True
assert envs.SGLANG_CACHE_DIT_FN == int(os.environ["SGLANG_CACHE_DIT_FN"])
assert envs.SGLANG_CACHE_DIT_BN == int(os.environ["SGLANG_CACHE_DIT_BN"])
assert envs.SGLANG_CACHE_DIT_WARMUP == int(os.environ["SGLANG_CACHE_DIT_WARMUP"])
assert envs.SGLANG_CACHE_DIT_RDT == float(os.environ["SGLANG_CACHE_DIT_RDT"])
assert envs.SGLANG_CACHE_DIT_MC == int(os.environ["SGLANG_CACHE_DIT_MC"])
commands = []
for proc in Path("/proc").iterdir():
    if not proc.name.isdigit():
        continue
    try:
        commands.append((proc / "cmdline").read_bytes().replace(b"\0", b" ").decode())
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        pass
assert any(
    "sglang serve" in command
    and "--model-type diffusion" in command
    and "--num-gpus 8" in command
    and "--tp-size 1" in command
    and "--ulysses-degree 8" in command
    and f"--quantization {os.environ['QUANTIZATION']}" in command
    and "--enable-torch-compile" in command
    and f"--lora-merge-mode {os.environ['LORA_MERGE_MODE']}" in command
    and f"--component-attention-backends {os.environ['COMPONENT_ATTENTION_BACKENDS']}"
    in command
    for command in commands
), "live sglang process is missing the diffusion dispatcher, 8-GPU topology, component backends, FP8, or LoRA mode"
print("optimization imports OK:", sol_attn.__file__, cache_dit.__file__, Fp8Config.get_name())
PY

worker_logs=$(sudo docker logs "${container}" 2>&1)
grep -Fq 'Using sol_attn attention backend' <<<"${worker_logs}" || {
  echo "${container}: MiniMax H3 did not resolve its lazy DiT backend to sol_attn" >&2
  exit 1
}
grep -Fq 'server_args:' <<<"${worker_logs}" || {
  echo "${container}: server_args were not logged" >&2
  exit 1
}
for expected_backend in \
  '"text_encoder": "torch_sdpa"' \
  '"audio_vae": "fa"' \
  '"video_vae": "fa"' \
  '"transformer": "sol_attn"'; do
  grep -Fq "${expected_backend}" <<<"${worker_logs}" || {
    echo "${container}: parsed server_args are missing ${expected_backend}" >&2
    exit 1
  }
done
grep -Fq '"enable_torch_compile": true' <<<"${worker_logs}" || {
  echo "${container}: parsed server_args did not enable torch.compile" >&2
  exit 1
}
grep -Fq 'merge_mode=merge' <<<"${worker_logs}" || {
  echo "${container}: startup LoRA was not statically merged" >&2
  exit 1
}
grep -Fq 'online FP8 layers after statically merging' <<<"${worker_logs}" || {
  echo "${container}: FP8 was not finalized after the startup LoRA merge" >&2
  exit 1
}
if grep -Fq 'Could not merge layer' <<<"${worker_logs}"; then
  echo "${container}: at least one startup LoRA layer failed to merge" >&2
  exit 1
fi

echo "OPTIMIZATION_STACK_VERIFIED: Sol-Attn + static-LoRA-before-FP8 + torch.compile + Cache-DiT (${SOL_CACHE_DIT_WARMUP}/${SOL_CACHE_DIT_RDT}/${SOL_CACHE_DIT_MC})"
