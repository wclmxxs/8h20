from __future__ import annotations

import asyncio
import json
import os
import re
import secrets
import tempfile
import time
from pathlib import Path
from typing import Any

import httpx
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.background import BackgroundTask

from . import storage

SGLANG_URL = os.getenv("SGLANG_URL", "http://127.0.0.1:30020").rstrip("/")
DATA_ROOT = Path(os.getenv("DATA_ROOT", "/data")).resolve()
JOB_ROOT = DATA_ROOT / "jobs"
UPLOAD_TMP_ROOT = DATA_ROOT / "tos-upload-tmp"
API_KEY = os.getenv("API_KEY", "")
MODEL_NAME = os.getenv("BUSINESS_MODEL", "MiniMax-H3")
GPU_GROUP_INDEX = int(os.getenv("GPU_GROUP_INDEX", "0"))
GPU_INDEXES = tuple(
    int(item) for item in os.getenv("GPU_INDEXES", "0,1,2,3,4,5,6,7").split(",") if item
)
GPU_UUIDS = tuple(item for item in os.getenv("GPU_UUIDS", "").split(",") if item)
RELEASE_ID = os.getenv("RELEASE_ID", "unknown")
LORA_REPO = os.getenv("LORA_REPO", "larryvrh/MiniMax-H3-Turbo-Lora")
LORA_REVISION = os.getenv("LORA_REVISION", "")
LORA_WEIGHT = os.getenv("LORA_WEIGHT", "minimax_h3_turbo_v4_step600_ema.safetensors")
LORA_NICKNAME = os.getenv("LORA_NICKNAME", "h3-turbo-v4")
LORA_SCALE = float(os.getenv("LORA_SCALE", "1.0"))
ATTENTION_BACKEND = os.getenv("ATTENTION_BACKEND", "fa")
COMPONENT_ATTENTION_BACKENDS = os.getenv(
    "COMPONENT_ATTENTION_BACKENDS", "transformer=sage_attn"
)
QUANTIZATION = os.getenv("QUANTIZATION", "")
LORA_MERGE_MODE = os.getenv("LORA_MERGE_MODE", "auto")
CACHE_DIT_ENABLED = os.getenv("SGLANG_CACHE_DIT_ENABLED", "false").lower() in {
    "1",
    "true",
    "yes",
    "on",
}
CACHE_DIT_CONFIG = {
    "fn": int(os.getenv("SGLANG_CACHE_DIT_FN", "1")),
    "bn": int(os.getenv("SGLANG_CACHE_DIT_BN", "0")),
    "warmup": int(os.getenv("SGLANG_CACHE_DIT_WARMUP", "4")),
    "rdt": float(os.getenv("SGLANG_CACHE_DIT_RDT", "0.24")),
    "mc": int(os.getenv("SGLANG_CACHE_DIT_MC", "3")),
}
UPSTREAM_TIMEOUT_SECONDS = float(os.getenv("UPSTREAM_TIMEOUT_SECONDS", "60"))
TASK_ID_PATTERN = re.compile(r"[A-Za-z0-9_-]{1,200}")
SEED_UPPER_BOUND = 1 << 63
TERMINAL_JOB_STATUSES = frozenset(
    {"completed", "succeeded", "failed", "deleted", "cancelled"}
)
ACTIVE_JOB_STATUSES = frozenset({"queued", "in_progress", "running"})
JOB_STATUS_RANK = {
    "unknown": 0,
    "queued": 1,
    "in_progress": 2,
    "running": 2,
    "completed": 3,
    "succeeded": 3,
    "failed": 3,
    "deleted": 3,
    "cancelled": 3,
}
UPLOAD_LOCKS: dict[str, asyncio.Lock] = {}


def validate_task_id(task_id: str) -> str:
    if not TASK_ID_PATTERN.fullmatch(task_id):
        raise HTTPException(status_code=400, detail="invalid task_id")
    return task_id


def job_file(task_id: str) -> Path:
    return JOB_ROOT / f"{validate_task_id(task_id)}.json"


def save_metadata(task_id: str, metadata: dict[str, Any]) -> None:
    JOB_ROOT.mkdir(parents=True, exist_ok=True)
    destination = job_file(task_id)
    temporary = destination.with_name(
        f".{destination.name}.{os.getpid()}.{secrets.token_hex(4)}.tmp"
    )
    try:
        temporary.write_text(json.dumps(metadata, ensure_ascii=False, indent=2))
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def load_metadata(task_id: str) -> dict[str, Any] | None:
    path = job_file(task_id)
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def record_job_status(
    task_id: str,
    status: object,
    metadata: dict[str, Any] | None = None,
    now: int | None = None,
) -> dict[str, Any]:
    """Persist status transitions so the external watchdog only polls live jobs."""
    current = metadata if metadata is not None else (load_metadata(task_id) or {})
    if not current:
        return current
    normalized = str(status or "unknown").lower()
    observed_at = int(time.time()) if now is None else int(now)
    previous = current.get("_watchdog") or {}
    previous_status = str(previous.get("status") or "unknown")
    if previous_status == normalized or JOB_STATUS_RANK.get(
        normalized, 0
    ) < JOB_STATUS_RANK.get(previous_status, 0):
        return current
    current["_watchdog"] = {
        "status": normalized,
        "status_changed_at": observed_at,
        "terminal": normalized in TERMINAL_JOB_STATUSES,
    }
    save_metadata(task_id, current)
    return current


def oldest_active_job_id() -> str | None:
    """Return the head of this single-flight worker's locally persisted queue."""
    candidates: list[tuple[int, int, str]] = []
    if not JOB_ROOT.is_dir():
        return None
    for path in JOB_ROOT.glob("*.json"):
        try:
            metadata = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        status = str((metadata.get("_watchdog") or {}).get("status") or "unknown")
        if status not in ACTIVE_JOB_STATUSES:
            continue
        task_id = str(metadata.get("id") or path.stem)
        if not TASK_ID_PATTERN.fullmatch(task_id):
            continue
        created_at = int(metadata.get("created_at") or 0)
        submitted_at_ns = int(metadata.get("submitted_at_ns") or created_at * 10**9)
        candidates.append((submitted_at_ns, created_at, task_id))
    return min(candidates, default=(0, 0, ""))[2] or None


def effective_job_status(
    task_id: str,
    upstream_status: object,
    metadata: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    """Expose the head of SGLang's single-flight queue as running.

    MiniMax H3's async API reports the active request as ``queued`` until it is
    terminal. Each deployed endpoint executes one request at a time, so the
    oldest live request is the one currently owned by the worker.
    """
    normalized = str(upstream_status or "unknown").lower()
    metadata = record_job_status(task_id, normalized, metadata=metadata)
    if normalized != "queued":
        return normalized, metadata

    local_status = str(
        (metadata.get("_watchdog") or {}).get("status") or "unknown"
    ).lower()
    if local_status in {"in_progress", "running"}:
        return "running", metadata
    if oldest_active_job_id() != task_id:
        return "queued", metadata

    metadata = record_job_status(task_id, "running", metadata=metadata)
    return "running", metadata


def with_resolved_seed(payload: dict[str, Any]) -> dict[str, Any]:
    resolved = dict(payload)
    if resolved.get("seed") is None:
        resolved["seed"] = secrets.randbelow(SEED_UPPER_BOUND)
    return resolved


def upstream_detail(response: httpx.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        return response.text[:2000] or f"HTTP {response.status_code}"
    if isinstance(payload, dict):
        detail = payload.get("detail") or payload.get("error") or payload.get("message")
        if detail:
            return (
                detail
                if isinstance(detail, str)
                else json.dumps(detail, ensure_ascii=False)
            )
    return json.dumps(payload, ensure_ascii=False)[:2000]


def raise_upstream(response: httpx.Response) -> None:
    if response.is_success:
        return
    status = response.status_code if 400 <= response.status_code < 500 else 502
    raise HTTPException(status_code=status, detail=upstream_detail(response))


async def submit_upstream(
    payload: dict[str, Any], business: dict[str, Any] | None = None
) -> dict[str, Any]:
    payload = with_resolved_seed(payload)
    try:
        async with httpx.AsyncClient(timeout=UPSTREAM_TIMEOUT_SECONDS) as client:
            response = await client.post(f"{SGLANG_URL}/v1/videos", json=payload)
    except httpx.HTTPError as exc:
        raise HTTPException(
            status_code=503, detail=f"SGLang is unavailable: {exc}"
        ) from exc
    raise_upstream(response)
    job = response.json()
    task_id = str(job.get("id") or "")
    if not task_id:
        raise HTTPException(
            status_code=502, detail="SGLang response did not contain id"
        )
    created_at = int(job.get("created_at") or time.time())
    metadata = {
        "id": task_id,
        "created_at": created_at,
        "submitted_at_ns": time.time_ns(),
        "request": payload,
        "business": business or {},
    }
    save_metadata(task_id, metadata)
    record_job_status(
        task_id,
        job.get("status") or "queued",
        metadata=metadata,
        now=created_at,
    )
    return job


async def retrieve_upstream(task_id: str) -> dict[str, Any]:
    validate_task_id(task_id)
    try:
        async with httpx.AsyncClient(timeout=UPSTREAM_TIMEOUT_SECONDS) as client:
            response = await client.get(f"{SGLANG_URL}/v1/videos/{task_id}")
    except httpx.HTTPError as exc:
        raise HTTPException(
            status_code=503, detail=f"SGLang is unavailable: {exc}"
        ) from exc
    raise_upstream(response)
    payload = response.json()
    metadata = load_metadata(task_id) or {}
    if metadata:
        status, metadata = effective_job_status(
            task_id, payload.get("status"), metadata=metadata
        )
        payload["status"] = status
        if status in {"completed", "succeeded"} and storage.describe()["enabled"]:
            metadata = await ensure_tos_output(task_id, metadata)
        payload["_deployment"] = metadata
        published_url = (metadata.get("_storage") or {}).get("url")
        if published_url:
            payload["output_url"] = published_url
    return payload


async def download_upstream_content(task_id: str, destination: Path) -> None:
    try:
        async with (
            httpx.AsyncClient(timeout=None) as client,
            client.stream(
                "GET", f"{SGLANG_URL}/v1/videos/{task_id}/content"
            ) as response,
        ):
            if not response.is_success:
                await response.aread()
                raise_upstream(response)
            with destination.open("wb") as output:
                async for chunk in response.aiter_bytes(4 * 1024 * 1024):
                    output.write(chunk)
    except HTTPException:
        raise
    except (httpx.HTTPError, OSError) as exc:
        raise HTTPException(
            status_code=502,
            detail=f"could not stage generated video for TOS: {type(exc).__name__}",
        ) from exc


async def ensure_tos_output(task_id: str, metadata: dict[str, Any]) -> dict[str, Any]:
    if (metadata.get("_storage") or {}).get("url"):
        return metadata

    lock = UPLOAD_LOCKS.setdefault(task_id, asyncio.Lock())
    async with lock:
        latest = load_metadata(task_id) or metadata
        if (latest.get("_storage") or {}).get("url"):
            return latest

        UPLOAD_TMP_ROOT.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{task_id}.", suffix=".mp4", dir=UPLOAD_TMP_ROOT
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        try:
            await download_upstream_content(task_id, temporary)
            try:
                published = await asyncio.to_thread(
                    storage.publish_file, temporary, task_id
                )
            except Exception as exc:
                raise HTTPException(
                    status_code=502,
                    detail=f"TOS upload failed: {type(exc).__name__}",
                ) from exc
            published["uploaded_at"] = int(time.time())
            latest["_storage"] = published
            save_metadata(task_id, latest)
            return latest
        finally:
            temporary.unlink(missing_ok=True)


async def delete_upstream(task_id: str) -> dict[str, Any]:
    validate_task_id(task_id)
    try:
        async with httpx.AsyncClient(timeout=UPSTREAM_TIMEOUT_SECONDS) as client:
            response = await client.delete(f"{SGLANG_URL}/v1/videos/{task_id}")
    except httpx.HTTPError as exc:
        raise HTTPException(
            status_code=503, detail=f"SGLang is unavailable: {exc}"
        ) from exc
    raise_upstream(response)
    payload = response.json()
    metadata = load_metadata(task_id) or {}
    if metadata:
        record_job_status(
            task_id,
            payload.get("status") or "cancelled",
            metadata=metadata,
        )
    return payload


async def stream_upstream(path: str) -> StreamingResponse:
    client = httpx.AsyncClient(timeout=None)
    try:
        request = client.build_request("GET", f"{SGLANG_URL}{path}")
        response = await client.send(request, stream=True)
    except httpx.HTTPError as exc:
        await client.aclose()
        raise HTTPException(
            status_code=503, detail=f"SGLang is unavailable: {exc}"
        ) from exc
    if not response.is_success:
        await response.aread()
        detail = upstream_detail(response)
        status = response.status_code if 400 <= response.status_code < 500 else 502
        await response.aclose()
        await client.aclose()
        raise HTTPException(status_code=status, detail=detail)

    headers = {}
    for name in ("content-length", "content-disposition", "etag", "last-modified"):
        if name in response.headers:
            headers[name] = response.headers[name]

    async def close() -> None:
        await response.aclose()
        await client.aclose()

    return StreamingResponse(
        response.aiter_bytes(1024 * 1024),
        status_code=response.status_code,
        media_type=response.headers.get("content-type", "video/mp4"),
        headers=headers,
        background=BackgroundTask(close),
    )


async def require_api_key(authorization: str | None = Header(default=None)) -> None:
    if not API_KEY:
        return
    if authorization != f"Bearer {API_KEY}":
        raise HTTPException(status_code=401, detail="invalid API key")


app = FastAPI(title="MiniMax H3 8xH20 gateway adapter")


@app.exception_handler(RequestValidationError)
async def request_validation_error_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    if request.url.path.startswith(("/ic/", "/sync_infer")):
        messages = []
        for error in exc.errors():
            location = ".".join(str(part) for part in error.get("loc", [])[1:])
            message = str(error.get("msg") or "invalid request")
            messages.append(f"{location}: {message}" if location else message)
        return JSONResponse(
            status_code=400,
            content={
                "error": {
                    "type": "invalid_request_error",
                    "message": "; ".join(messages),
                    "http_code": 400,
                }
            },
        )
    return JSONResponse(
        status_code=400, content=jsonable_encoder({"detail": exc.errors()})
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    if not request.url.path.startswith(("/ic/", "/sync_infer")):
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})
    error_type = {
        400: "invalid_request_error",
        401: "authentication_error",
        404: "not_found_error",
        409: "conflict_error",
        413: "payload_too_large_error",
        429: "rate_limit_error",
        502: "upstream_error",
        503: "upstream_unavailable_error",
        504: "timeout_error",
    }.get(exc.status_code, "internal_error")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "type": error_type,
                "message": str(exc.detail),
                "http_code": exc.status_code,
            }
        },
        headers=exc.headers,
    )


@app.get("/healthz")
async def healthz(_: None = Depends(require_api_key)) -> dict[str, Any]:
    healthy = False
    upstream: dict[str, Any] = {}
    error = None
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.get(f"{SGLANG_URL}/health")
        response.raise_for_status()
        upstream = response.json()
        healthy = True
    except Exception as exc:  # noqa: BLE001 - health must degrade instead of raising
        error = str(exc)
    return {
        "ok": healthy,
        "worker_count": 1,
        "healthy_workers": int(healthy),
        "workers": [{"id": GPU_GROUP_INDEX, "ok": healthy, "error": error}],
        "gpu_group_index": GPU_GROUP_INDEX,
        "gpu_indexes": list(GPU_INDEXES),
        "gpu_uuids": list(GPU_UUIDS),
        "upstream": upstream,
        "deployment": {
            "release_id": RELEASE_ID,
            "backend": "sglang",
            "model_variant": "fl2va",
            "lora_repo": LORA_REPO,
            "lora_revision": LORA_REVISION,
            "lora_weight": LORA_WEIGHT,
            "lora_nickname": LORA_NICKNAME,
            "lora_scale": LORA_SCALE,
            "attention_backend": ATTENTION_BACKEND,
            "component_attention_backends": COMPONENT_ATTENTION_BACKENDS,
            "quantization": QUANTIZATION or None,
            "lora_merge_mode": LORA_MERGE_MODE,
            "cache_dit_enabled": CACHE_DIT_ENABLED,
            "cache_dit_config": CACHE_DIT_CONFIG if CACHE_DIT_ENABLED else None,
            "output_storage": storage.describe(),
        },
    }


@app.post("/v1/videos")
async def create_video(
    request: Request, _: None = Depends(require_api_key)
) -> dict[str, Any]:
    payload = await request.json()
    if payload.get("task") == "ref2va":
        raise HTTPException(
            status_code=400, detail="ref2va is not deployed on this service"
        )
    return await submit_upstream(payload)


@app.get("/v1/videos/{task_id}")
async def retrieve_video(
    task_id: str, _: None = Depends(require_api_key)
) -> dict[str, Any]:
    return await retrieve_upstream(task_id)


@app.delete("/v1/videos/{task_id}")
async def delete_video(
    task_id: str, _: None = Depends(require_api_key)
) -> dict[str, Any]:
    return await delete_upstream(task_id)


@app.get("/v1/videos/{task_id}/content")
async def video_content(
    task_id: str, _: None = Depends(require_api_key)
) -> StreamingResponse:
    validate_task_id(task_id)
    return await stream_upstream(f"/v1/videos/{task_id}/content")


from .business import router as business_router

app.include_router(business_router)
