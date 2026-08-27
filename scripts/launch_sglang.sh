#!/usr/bin/env bash
set -euo pipefail

NUM_GPUS=${NUM_GPUS:-8}
TP=${TP:-1}
ULYSSES=${ULYSSES:-8}
ENCODER_PARALLEL=${ENCODER_PARALLEL:-auto}
CSDE_MODEL_ROOT=${CSDE_MODEL_ROOT:-/opt/tiger/csde/MiniMax-H3}
MODEL_VARIANT=${MODEL_VARIANT:-fl2va}
SGLANG_HOST=${SGLANG_HOST:-0.0.0.0}
SGLANG_PORT=${SGLANG_PORT:-30020}
OUTPUT_PATH=${OUTPUT_PATH:-/out/videos}
OPTIMIZATION_STACK_ENABLED=${OPTIMIZATION_STACK_ENABLED:-1}

if [[ ${NUM_GPUS} != 8 || ${TP} != 1 || ${ULYSSES} != 8 ]]; then
  echo "MiniMax H3 H20 topology must be NUM_GPUS=8, TP=1, ULYSSES=8; got ${NUM_GPUS}/${TP}/${ULYSSES}" >&2
  exit 1
fi

# The Bernard entrypoint localizes the HDFS MODEL_PATH into a directory whose
# name retains the MiniMax-H3 identity. MODEL remains an explicit override; a
# local MODEL_PATH is also accepted for non-CSDE environments. The hdfs:// value
# itself is deliberately not passed to SGLang because --model-path expects a
# local directory or repo ID.
if [[ -z ${MODEL:-} ]]; then
  if [[ -n ${MODEL_PATH:-} && -d ${MODEL_PATH:-} ]]; then
    MODEL=${MODEL_PATH}
  elif [[ -d ${CSDE_MODEL_ROOT} ]]; then
    MODEL=${CSDE_MODEL_ROOT}
  else
    MODEL=MiniMaxAI/MiniMax-H3
  fi
fi

if [[ ${MODEL} == /* ]]; then
  model_identity=${MODEL,,}
  model_identity=${model_identity//-/}
  model_identity=${model_identity//_/}
  if [[ ${model_identity} != *minimaxh3* ]]; then
    echo "local model path must retain the MiniMax-H3 identity for SGLang native pipeline resolution: ${MODEL}" >&2
    exit 1
  fi
  required_model_entries=(
    modular_model_index.json
    FL2VA
    audio_vae
    processor
    scheduler
    text_encoder
    tokenizer
    transformer
    vae
  )
  for entry in "${required_model_entries[@]}"; do
    if [[ ! -e ${MODEL}/${entry} ]]; then
      echo "local MiniMax H3 model is incomplete: missing ${MODEL}/${entry}" >&2
      exit 1
    fi
  done
  echo "Using localized MiniMax H3 model: ${MODEL}"
fi

if [[ ${OPTIMIZATION_STACK_ENABLED} == 1 ]]; then
  ATTENTION_BACKEND=${ATTENTION_BACKEND:-sol_attn}
  COMPONENT_ATTENTION_BACKENDS=${COMPONENT_ATTENTION_BACKENDS:-${SOL_COMPONENT_ATTENTION_BACKENDS:-text_encoder=torch_sdpa,audio_vae=fa,video_vae=fa,transformer=sol_attn}}
  ATTENTION_BACKEND_CONFIG=${ATTENTION_BACKEND_CONFIG:-${SOL_ATTENTION_BACKEND_CONFIG:-dense_backend=sage_attn,dense_steps=0,kv_splits=auto,tau=1.5}}
  WARMUP_STEPS=${WARMUP_STEPS:-${SOL_WARMUP_STEPS:-0}}
  QUANTIZATION=${QUANTIZATION:-${SOL_QUANTIZATION:-fp8}}
  ENABLE_TORCH_COMPILE=${ENABLE_TORCH_COMPILE:-${SOL_ENABLE_TORCH_COMPILE:-0}}
  LORA_MERGE_MODE=${LORA_MERGE_MODE:-${SOL_LORA_MERGE_MODE:-dynamic}}
  SGLANG_DIFFUSION_LORA_BEFORE_FP8=${SGLANG_DIFFUSION_LORA_BEFORE_FP8:-${SOL_LORA_BEFORE_FP8:-0}}
  export SOL_ATTN_STRICT=${SOL_ATTN_STRICT:-1}
  export SGLANG_DIFFUSION_LORA_MERGE_FP32=0
  export SGLANG_CACHE_DIT_ENABLED=${SGLANG_CACHE_DIT_ENABLED:-${SOL_CACHE_DIT_ENABLED:-true}}
  export SGLANG_CACHE_DIT_FN=${SGLANG_CACHE_DIT_FN:-${SOL_CACHE_DIT_FN:-1}}
  export SGLANG_CACHE_DIT_BN=${SGLANG_CACHE_DIT_BN:-${SOL_CACHE_DIT_BN:-0}}
  export SGLANG_CACHE_DIT_WARMUP=${SGLANG_CACHE_DIT_WARMUP:-${SOL_CACHE_DIT_WARMUP:-1}}
  export SGLANG_CACHE_DIT_RDT=${SGLANG_CACHE_DIT_RDT:-${SOL_CACHE_DIT_RDT:-0.12}}
  export SGLANG_CACHE_DIT_MC=${SGLANG_CACHE_DIT_MC:-${SOL_CACHE_DIT_MC:-3}}
else
  ATTENTION_BACKEND=fa
  COMPONENT_ATTENTION_BACKENDS=transformer=sage_attn
  ATTENTION_BACKEND_CONFIG=
  WARMUP_STEPS=
  QUANTIZATION=
  ENABLE_TORCH_COMPILE=0
  LORA_MERGE_MODE=auto
  SGLANG_DIFFUSION_LORA_BEFORE_FP8=0
  export SGLANG_CACHE_DIT_ENABLED=false
fi

case ${ENABLE_TORCH_COMPILE} in
  0|1) ;;
  *)
    echo "ENABLE_TORCH_COMPILE must be 0 or 1; got ${ENABLE_TORCH_COMPILE}" >&2
    exit 1
    ;;
esac
case ${SGLANG_DIFFUSION_LORA_BEFORE_FP8} in
  0|1) ;;
  *)
    echo "SGLANG_DIFFUSION_LORA_BEFORE_FP8 must be 0 or 1; got ${SGLANG_DIFFUSION_LORA_BEFORE_FP8}" >&2
    exit 1
    ;;
esac
if [[ ${OPTIMIZATION_STACK_ENABLED} == 1 ]] \
  && [[ ${QUANTIZATION} != fp8 || ${LORA_MERGE_MODE} != dynamic \
    || ${SGLANG_DIFFUSION_LORA_BEFORE_FP8} != 0 ]]; then
  echo "the H20 optimization stack requires an FP8 base model with dynamic LoRA residuals (QUANTIZATION=fp8, LORA_MERGE_MODE=dynamic, SGLANG_DIFFUSION_LORA_BEFORE_FP8=0)" >&2
  exit 1
fi

export ATTENTION_BACKEND COMPONENT_ATTENTION_BACKENDS ATTENTION_BACKEND_CONFIG
export WARMUP_STEPS QUANTIZATION ENABLE_TORCH_COMPILE LORA_MERGE_MODE
export SGLANG_DIFFUSION_LORA_BEFORE_FP8
export SGLANG_MINIMAX_H3_EXTRA_SHORT_EDGES=${SGLANG_MINIMAX_H3_EXTRA_SHORT_EDGES:-${SHORT_EDGES:-480,704}}
export NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-0}
export NCCL_GRAPH_REGISTER=${NCCL_GRAPH_REGISTER:-0}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:False}

lora_path=${LORA_REPO:-larryvrh/MiniMax-H3-Turbo-Lora}
if [[ -n ${LORA_LOCAL_PATH:-} ]]; then
  lora_path=${LORA_LOCAL_PATH}
fi

mkdir -p "${OUTPUT_PATH}"
cd /sgl-workspace/sglang

args=(
  sglang serve
  # Keep an explicit top-level dispatcher guard in addition to the named model
  # directory so local deployment changes cannot fall back to the LLM parser.
  --model-type diffusion
  --model-path "${MODEL}"
  --model-variant "${MODEL_VARIANT}"
  --num-gpus "${NUM_GPUS}"
  --tp-size "${TP}"
  --ulysses-degree "${ULYSSES}"
  --performance-mode speed
  --encoder-parallel "${ENCODER_PARALLEL}"
  --lora-path "${lora_path}"
  --lora-weight-name "${LORA_WEIGHT:-minimax_h3_turbo_v4_step600_ema.safetensors}"
  --lora-nickname "${LORA_NICKNAME:-h3-turbo-v4}"
  --lora-scale "${LORA_SCALE:-1.0}"
  --lora-merge-mode "${LORA_MERGE_MODE}"
  --output-path "${OUTPUT_PATH}"
  --host "${SGLANG_HOST}"
  --port "${SGLANG_PORT}"
)

# MiniMax H3 resolves its DiT backend from the component map below.  Keep the
# profile marker out of the top-level backend flag: the model needs different
# compatible backends for its text encoder, VAEs, and transformer.
if [[ -n ${COMPONENT_ATTENTION_BACKENDS} ]]; then
  args+=(--component-attention-backends "${COMPONENT_ATTENTION_BACKENDS}")
fi
if [[ -n ${ATTENTION_BACKEND_CONFIG} ]]; then
  args+=(--attention-backend-config "${ATTENTION_BACKEND_CONFIG}")
fi
if [[ -n ${WARMUP_STEPS} ]]; then
  args+=(--warmup-steps "${WARMUP_STEPS}")
fi
if [[ -n ${QUANTIZATION} ]]; then
  args+=(--quantization "${QUANTIZATION}")
fi
if [[ ${ENABLE_TORCH_COMPILE} == 1 ]]; then
  args+=(--enable-torch-compile)
fi
if [[ -n ${WARMUP:-} ]]; then
  read -r -a warmup <<<"${WARMUP}"
  args+=(--warmup-resolutions "${warmup[@]}")
fi

exec "${args[@]}"
