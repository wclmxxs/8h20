from __future__ import annotations

import asyncio
import ipaddress
import os
import socket
import time
from typing import Any, Literal
from urllib.parse import urlparse

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, model_validator

BUSINESS_MODEL = os.getenv("BUSINESS_MODEL", "MiniMax-H3")
PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "http://127.0.0.1:30010").rstrip("/")
DEFAULT_NFE = int(os.getenv("DEFAULT_NFE", "6"))
ALLOWED_NFE = frozenset(
    int(item) for item in os.getenv("ALLOWED_NFE", "4,6,8").split(",") if item
)
SYNC_INFER_TIMEOUT_SECONDS = int(os.getenv("SYNC_INFER_TIMEOUT_SECONDS", "1800"))
REMOTE_MEDIA_HOST_ALLOWLIST = tuple(
    item.strip().lower()
    for item in os.getenv("REMOTE_MEDIA_HOST_ALLOWLIST", ".byted.org").split(",")
    if item.strip()
)

router = APIRouter()


def core_module():
    # Imported lazily so contract tests and tooling can import this module directly.
    from . import main

    return main


def hostname_is_allowlisted(hostname: str) -> bool:
    normalized = hostname.rstrip(".").lower()
    return any(
        normalized == pattern.lstrip(".")
        or (pattern.startswith(".") and normalized.endswith(pattern))
        for pattern in REMOTE_MEDIA_HOST_ALLOWLIST
    )


def is_safe_remote_hostname(hostname: str) -> bool:
    if hostname_is_allowlisted(hostname):
        return True
    try:
        addresses = socket.getaddrinfo(hostname, None)
    except socket.gaierror:
        return False
    return all(ipaddress.ip_address(address[4][0]).is_global for address in addresses)


async def validate_media_hosts(request: GenerationRequest) -> None:
    for item in request.content:
        if item.type != "image_url" or item.image_url is None:
            continue
        hostname = urlparse(item.image_url.url).hostname
        if not hostname or not await asyncio.to_thread(
            is_safe_remote_hostname, hostname
        ):
            raise HTTPException(
                status_code=400,
                detail=(
                    "remote media host must be allowlisted or resolve only to public "
                    "IP addresses"
                ),
            )


class MediaURL(BaseModel):
    model_config = ConfigDict(extra="ignore")

    url: str

    @model_validator(mode="after")
    def validate_url(self) -> MediaURL:
        parsed = urlparse(self.url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("url must be an absolute http(s) URL")
        return self


class ContentItem(BaseModel):
    model_config = ConfigDict(extra="ignore")

    type: Literal["text", "image_url", "video_url", "audio_url"]
    role: str | None = None
    text: str | None = None
    image_url: MediaURL | None = None
    video_url: MediaURL | None = None
    audio_url: MediaURL | None = None

    @model_validator(mode="after")
    def validate_payload(self) -> ContentItem:
        values = {
            "text": self.text,
            "image_url": self.image_url,
            "video_url": self.video_url,
            "audio_url": self.audio_url,
        }
        populated = [name for name, value in values.items() if value is not None]
        if populated != [self.type]:
            raise ValueError(f"type={self.type!r} requires only the {self.type} field")
        if self.type == "text":
            if not self.text or not self.text.strip():
                raise ValueError("text must be non-empty")
            return self
        expected_roles = {
            "image_url": {"first_frame", "last_frame", "reference_image"},
            "video_url": {"reference_video"},
            "audio_url": {"reference_audio"},
        }
        if self.role not in expected_roles[self.type]:
            allowed = ", ".join(sorted(expected_roles[self.type]))
            raise ValueError(f"type={self.type!r} requires role in [{allowed}]")
        return self


class SolAttnOptimization(BaseModel):
    model_config = ConfigDict(extra="ignore", allow_inf_nan=False)

    enabled: bool | None = None
    dense_steps: int | None = Field(default=None, ge=0, le=100)
    tau: float | None = Field(default=None, gt=0)
    sink_conditioning: Literal["exact_kv", "exact_kv_and_rows", "off"] | None = None
    dense_prefix_seconds: float | None = Field(default=None, ge=0, le=15)


class CacheDitOptimization(BaseModel):
    model_config = ConfigDict(extra="ignore", allow_inf_nan=False)

    enabled: bool | None = None
    warmup: int | None = Field(default=None, ge=0, le=100)
    rdt: float | None = Field(default=None, ge=0, le=1)
    max_continuous_cached_steps: int | None = Field(default=None, ge=1, le=100)


class OptimizationConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    sol_attn: SolAttnOptimization | None = None
    cache_dit: CacheDitOptimization | None = None


class GenerationRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    model: str
    content: list[ContentItem] = Field(min_length=1)
    resolution: Literal["768P", "704P"]
    duration: int = Field(ge=4, le=15)
    ratio: Literal["adaptive", "21:9", "16:9", "4:3", "1:1", "3:4", "9:16"] | None = (
        None
    )
    num_inference_steps: int | None = Field(default=None, ge=1, le=50)
    seed: int | None = Field(default=None, ge=0, le=(1 << 63) - 1)
    optimization: OptimizationConfig | None = None

    @model_validator(mode="after")
    def validate_generation(self) -> GenerationRequest:
        if not any(
            item.type == "text" and item.text and item.text.strip()
            for item in self.content
        ):
            raise ValueError("content must contain at least one non-empty text item")
        nfe = self.num_inference_steps or DEFAULT_NFE
        if nfe not in ALLOWED_NFE:
            raise ValueError(
                f"num_inference_steps must be one of {sorted(ALLOWED_NFE)}"
            )

        media = [item for item in self.content if item.type != "text"]
        references = [
            item for item in media if item.role and item.role.startswith("reference_")
        ]
        if references:
            raise ValueError("reference media is not supported: ref2va is not deployed")
        if not media:
            if self.ratio is None or self.ratio == "adaptive":
                raise ValueError("text-only generation requires a non-adaptive ratio")
            return self
        if any(item.type != "image_url" for item in media):
            raise ValueError("fl2va accepts image_url keyframes only")
        if len(media) > 2:
            raise ValueError("fl2va accepts at most two keyframes")
        roles = [item.role for item in media]
        if len(set(roles)) != len(roles):
            raise ValueError("at most one first_frame and one last_frame are allowed")
        if self.ratio is None:
            self.ratio = "adaptive"
        return self


class QueryRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    model: str
    task_id: str

    @model_validator(mode="after")
    def validate_query(self) -> QueryRequest:
        if not self.task_id.strip():
            raise ValueError("task_id must be non-empty")
        return self


def media_url(item: ContentItem) -> str:
    assert item.image_url is not None
    return item.image_url.url


def to_upstream_request(request: GenerationRequest) -> dict[str, Any]:
    prompt = "\n".join(
        item.text.strip()
        for item in request.content
        if item.type == "text" and item.text
    )
    media = [item for item in request.content if item.type != "text"]
    conditions: list[dict[str, Any]] = []
    if not media:
        task = "t2va"
    else:
        task = "fl2va"
        ordered = sorted(media, key=lambda item: 0 if item.role == "first_frame" else 1)
        for item in ordered:
            conditions.append(
                {
                    "type": "image",
                    "uri": media_url(item),
                    "role": "keyframe",
                    "frame_index": 0 if item.role == "first_frame" else -1,
                }
            )
    nfe = request.num_inference_steps or DEFAULT_NFE
    payload: dict[str, Any] = {
        "model": BUSINESS_MODEL,
        "prompt": prompt,
        "seconds": request.duration,
        "task": task,
        "conditions": conditions,
        "target": {
            "short_edge": int(request.resolution.removesuffix("P")),
            "aspect_ratio": "auto" if request.ratio == "adaptive" else request.ratio,
            "duration_seconds": float(request.duration),
        },
        "num_outputs_per_prompt": 1,
        # SGLang counts the sigma grid including terminal zero; this API counts NFE.
        "num_inference_steps": nfe + 1,
        "flow_shift": 12.0,
        "audio_flow_shift": 3.0,
        "seed": request.seed,
    }
    if request.optimization is not None:
        payload["optimization"] = request.optimization.model_dump(exclude_none=True)
    return payload


def business_metadata(request: GenerationRequest) -> dict[str, Any]:
    return {
        "resolution": request.resolution,
        "duration": request.duration,
        "ratio": request.ratio,
        "nfe": request.num_inference_steps or DEFAULT_NFE,
    }


def task_payload(job: dict[str, Any]) -> dict[str, Any]:
    deployment = job.get("_deployment") or {}
    business = deployment.get("business") or {}
    upstream_request = deployment.get("request") or {}
    status = {
        "queued": "queued",
        "in_progress": "running",
        "running": "running",
        "completed": "succeeded",
        "succeeded": "succeeded",
        "failed": "failed",
        "deleted": "cancelled",
        "cancelled": "cancelled",
    }.get(str(job.get("status")), "failed")
    created_at = int(
        job.get("created_at") or deployment.get("created_at") or time.time()
    )
    status_changed_at = (deployment.get("_watchdog") or {}).get("status_changed_at")
    updated_at = max(
        int(value)
        for value in (
            created_at,
            status_changed_at,
            job.get("updated_at"),
            job.get("completed_at"),
        )
        if value is not None
    )
    task: dict[str, Any] = {
        "id": job["id"],
        "model": BUSINESS_MODEL,
        "status": status,
        "created_at": created_at,
        "updated_at": updated_at,
        "inference_time_s": (
            round(float(job["inference_time_s"]), 3)
            if job.get("inference_time_s") is not None
            else None
        ),
        "resolution": business.get("resolution", "768P"),
        "duration": business.get("duration", 5),
        "ratio": business.get("ratio", "16:9"),
        "task_type": "generation",
        "modality": "video",
    }
    if upstream_request.get("seed") is not None:
        task["seed"] = int(upstream_request["seed"])
    if status == "succeeded":
        task["content"] = {
            "url": f"{PUBLIC_BASE_URL}/ic/capcut/edit_gateway/v2/video_generation/{job['id']}/content"
        }
    elif status == "failed":
        raw_error = job.get("error") or {}
        message = (
            raw_error.get("message") if isinstance(raw_error, dict) else str(raw_error)
        )
        task["error"] = {
            "type": "upstream_error",
            "message": message or "Video generation failed",
            "http_code": 500,
        }
    return task


async def submit(request: GenerationRequest) -> str:
    await validate_media_hosts(request)
    core = core_module()
    job = await core.submit_upstream(
        to_upstream_request(request), business=business_metadata(request)
    )
    return str(job["id"])


@router.post("/ic/capcut/edit_gateway/v2/video_generation")
async def video_generation(request: GenerationRequest) -> dict[str, str]:
    return {"task_id": await submit(request)}


@router.post("/ic/capcut/edit_gateway/v2/query/video_generation")
async def query_video_generation(request: QueryRequest) -> dict[str, Any]:
    core = core_module()
    return {"task": task_payload(await core.retrieve_upstream(request.task_id))}


async def sync_impl(request: GenerationRequest) -> dict[str, Any] | JSONResponse:
    core = core_module()
    task_id = await submit(request)
    deadline = time.monotonic() + SYNC_INFER_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        task = task_payload(await core.retrieve_upstream(task_id))
        if task["status"] == "succeeded":
            return {"task": task}
        if task["status"] == "failed":
            return JSONResponse(status_code=500, content={"task": task})
        await asyncio.sleep(1)
    raise HTTPException(
        status_code=504,
        detail=f"Video generation exceeded {SYNC_INFER_TIMEOUT_SECONDS} seconds",
    )


@router.post("/sync_infer", response_model=None)
async def sync_infer(request: GenerationRequest) -> dict[str, Any] | JSONResponse:
    return await sync_impl(request)


@router.post("/ic/capcut/edit_gateway/v2/sync_infer", response_model=None)
async def namespaced_sync_infer(
    request: GenerationRequest,
) -> dict[str, Any] | JSONResponse:
    return await sync_impl(request)


@router.get("/ic/capcut/edit_gateway/v2/video_generation/{task_id}/content")
async def business_video_content(task_id: str) -> StreamingResponse:
    core = core_module()
    job = await core.retrieve_upstream(task_id)
    if job.get("status") not in {"completed", "succeeded"}:
        raise HTTPException(status_code=409, detail=f"task is {job.get('status')}")
    return await core.stream_upstream(f"/v1/videos/{task_id}/content")
