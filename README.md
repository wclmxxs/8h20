# MiniMax H3 — 8×H20 单视频并行部署

这是从 `4h200` 完整迁移出的 8×H20 仓库。运行时严格要求恰好 8 张 NVIDIA H20，并只创建一个 SGLang worker：每个视频都使用 GPU `0..7`，拓扑固定为 `TP=1 × Ulysses=8`，不会再拆成两个 4 卡服务。H200 不会被 `H20` 检查误接受。

目标 Merlin 配置（只读核对于 2026-08-26）：

| 项目 | 值 |
| --- | --- |
| Service ID | `Minimax-H3-Lora-H20` |
| PSM | `capcut.ai_infra_minimax_h3.dreamina` |
| Deployment | `2601961` / `my-default-jkt` |
| 资源 | 128 CPU / 1 TiB / 8×`h20` |
| 服务类型 | Bernard regular service / custom-bind-icm |

## 保留的优化

- SGLang `fl2va`，支持 T2V、首帧、尾帧和首尾帧；明确拒绝 `ref2va`。
- 固定 SGLang commit `c7c03ec53b` 和 OCI digest，继续应用 short-edge、请求级优化、temporal dense prefix / exact KV sink 三个补丁。
- 主 DiT 保留 Sol-Attn；文本编码器用 Torch SDPA，Audio/Video VAE 用 FlashAttention，避免组件后端串用。
- 在线 FP8、动态 Turbo LoRA、Cache-DiT `Fn=1/Bn=0/W=1/R=0.12/MC=3` 全部保留。
- `sink_conditioning=exact_kv` 默认保持文本、首尾帧、参考素材和音频 conditioning KV 精确；可按请求启用 dense prefix。
- SM90 SageAttention 构建、allocator `expandable_segments`、480/704 short edge、warmup resolutions、请求级优化覆盖、SSRF 防护、任务清理和业务 API 兼容层全部保留。

共享入口 [scripts/launch_sglang.sh](scripts/launch_sglang.sh) 同时供 Compose 与 Bernard 使用，防止两个部署面漂移。它会拒绝任何不是 `NUM_GPUS=8, TP=1, ULYSSES=8` 的拓扑。

## Merlin / Bernard（目标部署方式）

[docker/Dockerfile.bernard](docker/Dockerfile.bernard) 是单容器镜像：构建完整优化栈，在容器内启动 8 卡 SGLang，并在 Bernard 注入的 `$PORT` 上启动业务 API。它还提供目标服务已配置的 `/opt/tiger/csde/healthcheck.sh` 兼容路径；SGLang 或 API 任一退出时整个容器退出，让 Bernard 负责重建 Pod。

本地只做镜像构建示例，不会变更 Merlin 部署：

```bash
docker build -f docker/Dockerfile.bernard -t minimax-h3-h20-bernard:20260826-v1 .
```

部署镜像前必须确认模型来源：启动器读取 `MODEL`（默认 `MiniMaxAI/MiniMax-H3`）和 `LORA_LOCAL_PATH`/`LORA_REPO`。目标服务当前的 `MODEL_PATH=hdfs://...` 属于原 CSDE 镜像约定，不会被 SGLang 自动当作本地目录；如果目标环境不能直连 Hugging Face，应先把基模和 LoRA 放入镜像或挂载目录，并显式设置 `MODEL` 与 `LORA_LOCAL_PATH`。不要只沿用当前 `MODEL_PATH` 后直接发版。

Bernard 模式不启动自注册 Reporter、Docker-socket Watchdog 或 cleaner 容器；平台负责实例注册/健康重建，业务 API 仍把视频与任务元数据写到 `DATA_ROOT`。

## 自管 Docker Compose（保留的备用方式）

`install.sh` 保留原项目的 AWS/DLAMI 自管能力，但已改成单个 8×H20 worker，并会强校验 `SERVICE_ID=Minimax-H3-Lora-H20`、GPU 数量、GPU 型号与 TP/Ulysses 拓扑：

```bash
git clone git@github.com:wclmxxs/8h20.git
cd 8h20
./install.sh
```

它通过 AWS IMDSv2 获取公网 IP/instance-id，构建 SGLang、API、Reporter、Watchdog 镜像，下载并校验 LoRA，启动唯一端口 `30010`，在 IPv4/IPv6 loopback 探活后注册。AMI 快速路径继续支持：

```bash
./prepare_ami.sh
./install.sh --from-ami
```

常用操作：

```bash
./status.sh
./smoke_test.sh
./update_api.sh
./stop.sh
./disable_sol_ab.sh
```

唯一拓扑：

| 对外端口 | GPU | SGLang 拓扑 | 优化 |
| --- | --- | --- | --- |
| `30010` | `0,1,2,3,4,5,6,7` | `TP=1, Ulysses=8` | Sol-Attn + FP8 + Cache-DiT + 动态 LoRA |

安装脚本会严格验证实际进程包含 `--num-gpus 8 --tp-size 1 --ulysses-degree 8`，并校验 Sol/Cache-DiT/FP8 模块、环境和 DiT Sol 启动日志。任何一项没有真正生效都会退出。Sol SM90 kernel 会按 token shape 专门化，首次请求可能包含 JIT；稳态测速应固定 seed 和请求参数，连续运行两次并取第二次，同时检查视频和音频质量。

自管模式日志：

```bash
sudo docker logs -f minimax-h3-h20-sglang-0
sudo docker logs -f minimax-h3-h20-api-0
sudo docker logs -f minimax-h3-h20-reporter
sudo docker logs -f minimax-h3-h20-watchdog
```

## 自管模式注册协议

Reporter 每 5 秒请求：

```http
POST /ic/capcut/edit_gateway/v1/report_catalog
Content-Type: application/json
X-Internal-Auth: bernard-edit-bridge-internal-call
```

请求体：

```json
{
  "psm": "capcut.ai_infra_minimax_h3.dreamina",
  "service_id": "Minimax-H3-Lora-H20",
  "instances_json": "[{\"id\":\"i-xxx-8h20-0\",\"host\":\"16.78.214.130\",\"ports\":[30010],\"state\":\"TASK_RUNNING\",\"healthCheckResults\":[{\"alive\":true}],\"containerInfos\":{\"h3-8h20-0\":{\"request\":{\"cpu\":128,\"memory\":1048576,\"nvidia.com/gpu\":8}}}}]"
}
```

整机只上报这一个实例。Reporter 从 Docker 内网探测 API 的 `/healthz`；SGLang 不健康时下一次上报 `alive=false`。

## 对外业务接口

接口结构与 RTX6000PRO 版本一致。

提交：

```http
POST /ic/capcut/edit_gateway/v2/video_generation
Content-Type: application/json
```

T2V 示例：

```json
{
  "model": "MiniMax-H3",
  "content": [
    {"type": "text", "text": "A cinematic sunrise over a quiet lake."}
  ],
  "resolution": "768P",
  "duration": 5,
  "ratio": "16:9",
  "num_inference_steps": 6,
  "seed": 42
}
```

`seed` 可选：省略或传 `null` 时，API 会为每个任务生成独立的 63 位随机 seed；显式传整数时保持可复现。实际使用的 seed 会保存在任务元数据中，并通过查询响应的 `task.seed` 返回。

Sol-Attn 和 Cache-DiT 可按请求覆盖。未传 `optimization`，或对象内省略某个字段时，继续使用 `.env` 中的部署默认值：

```json
{
  "optimization": {
    "sol_attn": {
      "enabled": true,
      "dense_steps": 2,
      "tau": 1.25,
      "sink_conditioning": "exact_kv",
      "dense_prefix_seconds": 3.0
    },
    "cache_dit": {
      "enabled": true,
      "warmup": 2,
      "rdt": 0.08,
      "max_continuous_cached_steps": 2
    }
  }
}
```

`sink_conditioning` 默认是 `exact_kv`，始终将目标视频之前的文本、首尾帧、参考素材和音频 conditioning KV 保持精确，避免 Sol 稀疏化削弱人物身份和提示词约束；可传 `exact_kv_and_rows` 进一步让这些 conditioning query 也走 Dense，或传 `off` 关闭。`dense_steps` 控制前几个扩散 step 使用完整 attention；`dense_prefix_seconds` 控制成片开头多少秒使用完整 attention。后者会在 conditioning sink 基础上继续把目标视频前缀保留为精确 KV sink，并用 Dense 结果覆盖对应 query。若请求视频时长小于或等于 `dense_prefix_seconds`，该请求整段使用 Dense，不调用 Sol-Attn。默认值为 `0`，保持全时段 Sol-Attn 行为。

`enabled=false` 可对单个请求关闭对应优化；下一条未传覆盖值的请求会自动恢复部署默认配置。FP8 量化和 attention backend 的加载方式仍是部署级配置，不能在请求中切换。业务接口和 `/v1/videos` 都接受同一个 `optimization` 对象。

FL2V 示例：

```json
{
  "model": "MiniMax-H3",
  "content": [
    {"type": "text", "text": "The camera slowly pushes toward the subject."},
    {
      "type": "image_url",
      "role": "first_frame",
      "image_url": {"url": "https://example.com/first.jpg"}
    },
    {
      "type": "image_url",
      "role": "last_frame",
      "image_url": {"url": "https://example.com/last.jpg"}
    }
  ],
  "resolution": "704P",
  "duration": 5,
  "ratio": "adaptive",
  "num_inference_steps": 6
}
```

返回：

```json
{"task_id": "video_xxx"}
```

查询：

```http
POST /ic/capcut/edit_gateway/v2/query/video_generation
Content-Type: application/json

{"model":"MiniMax-H3","task_id":"video_xxx"}
```

同步接口也保留：

- `POST /sync_infer`
- `POST /ic/capcut/edit_gateway/v2/sync_infer`
- `GET /ic/capcut/edit_gateway/v2/video_generation/{task_id}/content`

另外提供带 Bearer API Key 的 SGLang 兼容代理：

- `GET /healthz`
- `POST /v1/videos`
- `GET /v1/videos/{task_id}`
- `DELETE /v1/videos/{task_id}`
- `GET /v1/videos/{task_id}/content`

业务接口保持与现有 gateway 相同的无 API Key 调用方式；内部健康检查和 `/v1` 兼容接口使用 `.env` 中自动生成的 `API_KEY`。

## 主要配置

| 环境变量 | 默认值 | 说明 |
| --- | --- | --- |
| `SERVICE_ID` | `Minimax-H3-Lora-H20` | 自管 install 会强校验，防止注册到错误池子 |
| `API_BASE_PORT` | `30010` | 唯一 8 卡 worker 的对外端口 |
| `NUM_GPUS` | `8` | 每个请求使用的 GPU 总数；启动器拒绝其他值 |
| `TP` | `1` | Tensor Parallel 维度 |
| `ULYSSES` | `8` | Sequence Parallel 维度，让单个视频跨 8 张卡执行 |
| `MODEL` | `MiniMaxAI/MiniMax-H3` | 基模；部署命令始终显式传入 |
| `SGLANG_BASE_IMAGE` | `nightly-dev-20260812-c7c03ec5@sha256:d753…` | 与短边补丁匹配并锁定 digest 的 SGLang 基础镜像 |
| `REBUILD_GPU_IMAGES` | `0` | 已存在 GPU 镜像时复用；仅修改 GPU Dockerfile/补丁时显式设为 `1` |
| `LORA_REPO` | `larryvrh/MiniMax-H3-Turbo-Lora` | 静态 LoRA 仓库 |
| `LORA_REVISION` | `43a7455…` | 已验证的当前 LoRA commit |
| `LORA_WEIGHT` | `minimax_h3_turbo_v4_step600_ema.safetensors` | 当前 LoRA 文件 |
| `DEFAULT_NFE` | `6` | 业务接口默认实际去噪次数 |
| `SHORT_EDGES` | `480,704` | 在官方 768 之外额外启用的短边 |
| `WARMUP` | `864x480 1248x704 1344x768` | SGLang 启动预热规格 |
| `ATTENTION_BACKEND` | `fa` | 所有组件的安全基础后端，避免 Audio/Video VAE 使用不支持的 SageAttention |
| `COMPONENT_ATTENTION_BACKENDS` | `transformer=sage_attn` | 只把主去噪 transformer 切到 SageAttention |
| `OPTIMIZATION_STACK_ENABLED` | `1` | 是否给唯一 8 卡 worker 启用 Sol-Attn + FP8 + Cache-DiT |
| `SOL_COMPONENT_ATTENTION_BACKENDS` | `text_encoder=torch_sdpa,audio_vae=fa,video_vae=fa,transformer=sol_attn` | H3 DiT 使用 Sol；install 会自动补齐并强制保护文本编码器及 Audio/Video VAE，避免旧 `.env` 或自定义值误用 Sol |
| `SOL_ATTENTION_BACKEND_CONFIG` | `dense_backend=sage_attn,dense_steps=0,kv_splits=auto,tau=1.5` | Sol 激进稀疏配置；6 NFE 的全部 step 均进入稀疏路径 |
| `SOL_ATTN_STRICT` | `1` | 禁止 Sol kernel 异常时静默回退为 dense，避免产生虚假测速结果 |
| `SOL_WARMUP_STEPS` | `3` | 启动时执行 3 个 warmup step，覆盖 dense 和 sparse 两种 kernel 路径 |
| `SOL_QUANTIZATION` | `fp8` | 在线量化主 transformer |
| `SOL_LORA_MERGE_MODE` | `dynamic` | 动态应用 Turbo LoRA，不修改量化基模权重 |
| `SOL_CACHE_DIT_ENABLED` | `true` | 进程级启用 Cache-DiT |
| `SOL_CACHE_DIT_WARMUP` | `1` | 仅首个去噪 step 强制完整计算 |
| `SOL_CACHE_DIT_RDT` | `0.12` | 激进残差差异缓存阈值，允许更多复用 |
| `SOL_CACHE_DIT_MC` | `3` | 最多连续缓存 3 个 step |
| `REMOTE_MEDIA_HOST_ALLOWLIST` | `.byted.org` | 可访问的私网图片域名后缀；公网域名自动允许 |
| `VIDEO_RETENTION_HOURS` | `12` | 视频和对应任务元数据保留时间 |
| `PYTORCH_CUDA_ALLOC_CONF` | `expandable_segments:True` | 减少跨请求显存碎片和可恢复性 OOM |
| `WATCHDOG_STALL_SECONDS` | `300` | 有活跃任务但没有状态推进多久后重启对应 worker |
| `WATCHDOG_RESTART_COOLDOWN_SECONDS` | `300` | 自管 worker 两次自动重启之间的最短间隔 |
| `CLEANUP_INTERVAL_SECONDS` | `600` | 清理任务执行间隔；实际删除可能比 12 小时最多晚约 10 分钟 |
| `DATA_ROOT` | `/opt/dlami/nvme/minimax-h3-8h20` | 与 RTX6000PRO 仓完全分离 |
| `MODEL_CACHE_ROOT` | 空（解析为 `${DATA_ROOT}/hf-cache`） | Hugging Face 基模和 LoRA 缓存；AMI 复用时应指向 EBS |
| `STARTUP_TIMEOUT_SECONDS` | `1800` | SGLang 等待加载和 warmup 的最长秒数 |
| `STARTUP_PROGRESS_SECONDS` | `15` | 等待模型加载和 warmup 时输出一次进度的间隔 |

如果 SGLang 上游代码结构变化，构建阶段会因短边补丁不匹配而失败，不会静默启动一个只支持 768 的服务。

MiniMax H3 的 DiT attention backend 在第一次 forward 时延迟解析。优化 worker 因此使用全局 `sol_attn`，同时将 `text_encoder`、`audio_vae`、`video_vae` 显式覆盖为兼容后端；启动脚本会同时检查 DiT 实际解析为 Sol 和 Audio VAE 保持 FA，任一不满足都会退出。

SageAttention 会改变 transformer 的 attention 数值路径。需要让所有组件回退到 FlashAttention 时清空组件覆盖后重新执行安装：

```bash
sed -i 's/^ATTENTION_BACKEND=.*/ATTENTION_BACKEND=fa/' .env
sed -i 's/^COMPONENT_ATTENTION_BACKENDS=.*/COMPONENT_ATTENTION_BACKENDS=/' .env
./install.sh
```

## 本地验证

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements-dev.txt
.venv/bin/pytest
.venv/bin/ruff check .
find . -type f -name '*.sh' -exec bash -n {} +
find api reporter scripts tests watchdog -type f -name '*.py' -print0 | xargs -0 python3 -m py_compile
```
