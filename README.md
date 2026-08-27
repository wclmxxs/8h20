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
- 固定 SGLang commit `c7c03ec53b` 和 OCI digest，继续应用 short-edge、请求级优化、temporal dense prefix / exact KV sink，并使用 FP8 基模叠加 dynamic LoRA 残差。
- 主 DiT 在模型加载的组件上下文中立即解析为 Sol-Attn；文本编码器用 Torch SDPA，Audio/Video VAE 用 FlashAttention，避免延迟解析回退或组件后端串用。
- 基模 transformer 在线量化为 FP8，Turbo LoRA 保持为独立 dynamic 残差；随后启用 `torch.compile`。Ulysses/Ring 的动态 attention、每层 `all_to_all` 和最终 SP `all_gather` 保持 eager，避免不同视频 shape 的 collective 被 Inductor 专门化后触发 NCCL 错误；其余投影、归一化、残差和 MLP 继续编译，并保留 Cache-DiT `Fn=1/Bn=0/W=1/R=0.12/MC=3`。
- `sink_conditioning=exact_kv` 默认保持文本、首尾帧、参考素材和音频 conditioning KV 精确；可按请求启用 dense prefix。
- SM90 SageAttention 构建、allocator `expandable_segments`、480/704 short edge、warmup resolutions、请求级优化覆盖、SSRF 防护、任务清理和业务 API 兼容层全部保留。

共享入口 [scripts/launch_sglang.sh](scripts/launch_sglang.sh) 同时供 Compose 与 Bernard 使用，防止两个部署面漂移。它会拒绝任何不是 `NUM_GPUS=8, TP=1, ULYSSES=8` 的拓扑。

## Merlin / Bernard（目标部署方式）

[docker/Dockerfile.bernard](docker/Dockerfile.bernard) 是单容器镜像：构建完整优化栈，在容器内启动 8 卡 SGLang，并在 Bernard 注入的 `$PORT` 上启动业务 API。它还提供目标服务已配置的 `/opt/tiger/csde/healthcheck.sh` 兼容路径；SGLang 或 API 任一退出时整个容器退出，让 Bernard 负责重建 Pod。

本地只做镜像构建示例，不会变更 Merlin 部署：

```bash
docker build -f docker/Dockerfile.bernard -t minimax-h3-h20-bernard:20260826-v1 .
```

目标服务保留 `MODEL_PATH=hdfs://...`。Bernard 镜像从当前 CSDE 工具镜像的固定 digest 复制 `/opt/tiger/hdfs_client`，不依赖 Pod 运行时 SCM 挂载。每个新 Pod 从 HDFS 下载到 `/opt/tiger/csde/MiniMax-H3.partial`，校验 `modular_model_index.json`、FL2VA、Transformer、VAE、文本编码器、Tokenizer 等关键文件后再原子切换为 `/opt/tiger/csde/MiniMax-H3`。目录名刻意保留真实模型身份，使 SGLang 直接命中原生 `MiniMaxH3Pipeline`；构建阶段也会对该注册结果做断言。显式设置的本地 `MODEL` / `MODEL_PATH` 仍可覆盖自动选择，但路径同样必须包含 MiniMax-H3 身份；HDFS client 缺失或目录不完整时会直接退出，不会静默回退到在线基模。

Bernard 镜像在构建阶段下载并按固定 revision、大小和 SHA256 校验 Turbo LoRA，运行时通过 `LORA_LOCAL_PATH` 只读取镜像内缓存，不依赖 Pod 公网访问。

Bernard 模式不启动自注册 Reporter、Docker-socket Watchdog 或 cleaner 容器；平台负责实例注册/健康重建，业务 API 仍把视频与任务元数据写到 `DATA_ROOT`。

### 常驻 Pod 调试模式

短期反复验证 Python 改动时，为部署设置 `BERNARD_DEBUG_HOLD=1`。镜像仍会先把模型完整下载到 `/opt/tiger/csde/MiniMax-H3`，然后 PID 1 进入常驻的 child reaper，不自动启动 SGLang/API。调试健康检查固定成功，子进程退出不会触发容器重建或再次下载模型。该模式没有业务服务时也会显示健康，因此只能用于已隔离流量的调试实例。

首次进入 Pod WebShell 后，每条命令分别执行：

```bash
cd /tmp
git clone https://github.com/wclmxxs/8h20.git minimaxh3-8h20-hotpatch
cd /tmp/minimaxh3-8h20-hotpatch
bash scripts/hotpatch_current_bernard_pod.sh start
```

启动命令会定位实际导入的 `minimax_h3.py`，确保最终 SP `all_gather` 的 CUDA Graph eager 修复存在，并从当前 Git checkout 启动 SGLang 和 API。它立即返回，模型加载和 `torch.compile` 预热在后台继续：

```bash
bash scripts/hotpatch_current_bernard_pod.sh status
tail -f /tmp/minimax-h3-debug-sglang.log
```

修改并推送代码后，在同一个 Pod 中复用模型和镜像：

```bash
git pull --ff-only origin main
bash scripts/hotpatch_current_bernard_pod.sh restart
```

也可执行 `bash scripts/hotpatch_current_bernard_pod.sh stop` 释放 GPU。只有首次启用调试模式需要构建并部署一次包含该能力的镜像；之后的 Python/API/启动脚本迭代不再需要构建镜像。SGLang 的 Python 源码补丁由调试脚本直接落到当前容器；CUDA 扩展、系统依赖或基础镜像变化仍必须重新构建。

若启动日志在 MiniMax-H3 attention 的 `all_to_all_single` 报 NCCL
`unhandled cuda error`，先在当前 Pod 中禁用 NCCL GPU P2P，判断是否为节点
P2P transport 问题：

```bash
NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,GRAPH,COLL NCCL_P2P_DISABLE=1 ENABLE_TORCH_COMPILE=0 bash scripts/hotpatch_current_bernard_pod.sh restart
```

该诊断仍使用单请求 8 卡 Ulysses，只让 NCCL 绕开 GPU P2P transport。若启动
成功，再判断是保留该兼容模式，还是修复节点 P2P 后恢复默认 transport；随后再
单独恢复并测试 `ENABLE_TORCH_COMPILE=1`。当前固定版 SGLang 的原生 MiniMax-H3
attention 即使传入 `--kv-gather-degree` 仍会回退到 Ulysses all-to-all，因此不将
它作为 H20 transport 的规避方案。

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
| `30010` | `0,1,2,3,4,5,6,7` | `TP=1, Ulysses=8` | Sol-Attn + FP8 基模/dynamic LoRA + torch.compile + Cache-DiT |

安装脚本会严格验证实际进程包含 `--num-gpus 8 --tp-size 1 --ulysses-degree 8 --enable-torch-compile`，并校验基模使用 FP8、LoRA 以 dynamic 残差运行，以及 Sol/Cache-DiT 模块和 DiT Sol 启动日志。任何一项没有真正生效都会退出。torch.compile 与 Sol SM90 kernel 会按 token shape 专门化，首次启动/首个新 shape 会包含编译或 JIT；稳态测速应固定 seed 和请求参数，连续运行两次并取第二次，同时检查视频和音频质量。

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
| `LORA_REPO` | `larryvrh/MiniMax-H3-Turbo-Lora` | Turbo LoRA 仓库 |
| `LORA_REVISION` | `43a7455…` | 已验证的当前 LoRA commit |
| `LORA_WEIGHT` | `minimax_h3_turbo_v4_step600_ema.safetensors` | 当前 LoRA 文件 |
| `DEFAULT_NFE` | `6` | 业务接口默认实际去噪次数 |
| `SHORT_EDGES` | `480,704` | 在官方 768 之外额外启用的短边 |
| `WARMUP` | `864x480 1248x704 1344x768` | SGLang 启动预热规格 |
| `ATTENTION_BACKEND` | `fa` | 所有组件的安全基础后端，避免 Audio/Video VAE 使用不支持的 SageAttention |
| `COMPONENT_ATTENTION_BACKENDS` | `transformer=sage_attn` | 只把主去噪 transformer 切到 SageAttention |
| `OPTIMIZATION_STACK_ENABLED` | `1` | 是否给唯一 8 卡 worker 启用 Sol-Attn + FP8 基模/dynamic LoRA + torch.compile + Cache-DiT |
| `SOL_COMPONENT_ATTENTION_BACKENDS` | `text_encoder=torch_sdpa,audio_vae=fa,video_vae=fa,transformer=sol_attn` | H3 DiT 使用 Sol；install 会自动补齐并强制保护文本编码器及 Audio/Video VAE，避免旧 `.env` 或自定义值误用 Sol |
| `SOL_ATTENTION_BACKEND_CONFIG` | `dense_backend=sage_attn,dense_steps=0,kv_splits=auto,tau=1.5` | Sol 激进稀疏配置；6 NFE 的全部 step 均进入稀疏路径 |
| `SOL_ATTN_STRICT` | `1` | 禁止 Sol kernel 异常时静默回退为 dense，避免产生虚假测速结果 |
| `SOL_WARMUP_STEPS` | `3` | 启动时执行 3 个 warmup step，覆盖 dense 和 sparse 两种 kernel 路径 |
| `SOL_QUANTIZATION` | `fp8` | 在线量化基模 transformer，LoRA 不合入量化权重 |
| `SOL_ENABLE_TORCH_COMPILE` | `1` | 对 FP8 基模 transformer 开启 `torch.compile` |
| `SOL_LORA_MERGE_MODE` | `dynamic` | 保留 Turbo LoRA 为独立动态残差，避免静态合并后重新量化造成的画面发糊 |
| `SOL_LORA_BEFORE_FP8` | `0` | 禁止延迟 FP8 和静态 LoRA 合并路径 |
| `SOL_CACHE_DIT_ENABLED` | `true` | 进程级启用 Cache-DiT |
| `SOL_CACHE_DIT_WARMUP` | `1` | 仅首个去噪 step 强制完整计算 |
| `SOL_CACHE_DIT_RDT` | `0.12` | 激进残差差异缓存阈值，允许更多复用 |
| `SOL_CACHE_DIT_MC` | `3` | 最多连续缓存 3 个 step |
| `REMOTE_MEDIA_HOST_ALLOWLIST` | `.byted.org` | 可访问的私网图片域名后缀；公网域名自动允许 |
| `PUBLIC_BASE_URL` | 自动发现 | Bernard 默认优先选择实例的全局 IPv6，并结合动态 `PORT` 返回临时视频直链；显式设置时保持指定地址 |
| `PUBLIC_ADVERTISE_IP` | 空 | 可覆盖自动发现的实例地址；IPv6 会自动按 URL 规范添加方括号 |
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

Bernard 将 MiniMax H3 落盘为显式命名的 `/opt/tiger/csde/MiniMax-H3`，由 SGLang 注册表解析为原生 `MiniMaxH3Pipeline`；启动器同时固定传入官方 `--model-type diffusion`，防止顶层分流回退到 LLM 服务。MiniMax H3 的 DiT attention backend 在第一次 forward 时延迟解析。`ATTENTION_BACKEND=sol_attn` 在这里是优化 profile 标识；实际后端通过组件映射把 transformer 指向 Sol，并将 `text_encoder`、`audio_vae`、`video_vae` 显式覆盖为兼容后端；启动脚本会同时检查 DiT 实际解析为 Sol 和 Audio VAE 保持 FA，任一不满足都会退出。

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
