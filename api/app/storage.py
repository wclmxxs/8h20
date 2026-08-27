from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any
from urllib.parse import quote


@dataclass(frozen=True)
class TosConfig:
    bucket: str
    access_key: str
    secret_key: str
    key_prefix: str
    public_base_url: str
    service_name: str
    idc: str
    cluster: str
    addr_family: str

    @classmethod
    def from_env(cls) -> TosConfig:
        bucket = os.getenv("TOS_BUCKET", "").strip()
        return cls(
            bucket=bucket,
            access_key=os.getenv("TOS_ACCESS_KEY", "").strip(),
            secret_key=os.getenv("TOS_SECRET_KEY", "").strip(),
            key_prefix=os.getenv(
                "TOS_KEY_PREFIX", "minimax_h3_data_cache/outputs"
            ).strip("/"),
            public_base_url=os.getenv(
                "TOS_PUBLIC_BASE_URL",
                f"https://tosv.byted.org/obj/{bucket}" if bucket else "",
            ).rstrip("/"),
            service_name=os.getenv("TOS_SERVICE_NAME", "toutiao.tos.tosapi").strip(),
            idc=os.getenv("TOS_IDC", "").strip(),
            cluster=os.getenv("TOS_CLUSTER", "default").strip(),
            addr_family=os.getenv("TOS_ADDR_FAMILY", "dual-stack").strip(),
        )

    @property
    def enabled(self) -> bool:
        return bool(
            self.bucket
            and self.access_key
            and self.secret_key
            and self.public_base_url
            and self.service_name
        )

    def object_key(self, task_id: str) -> str:
        filename = f"{task_id}.mp4"
        return f"{self.key_prefix}/{filename}" if self.key_prefix else filename

    def public_url(self, object_key: str) -> str:
        return f"{self.public_base_url}/{quote(object_key, safe='/-_.~')}"


def describe() -> dict[str, Any]:
    config = TosConfig.from_env()
    return {
        "enabled": config.enabled,
        "provider": "tos" if config.enabled else None,
        "bucket": config.bucket or None,
        "key_prefix": config.key_prefix or None,
        "service_name": config.service_name or None,
        "idc": config.idc or None,
        "addr_family": config.addr_family or None,
    }


@lru_cache(maxsize=4)
def _filesystem(config: TosConfig):
    import tosfsspec

    options: dict[str, Any] = {
        "key": config.access_key,
        "secret": config.secret_key,
        "service_name": config.service_name,
        "cluster": config.cluster,
        "addr_family": config.addr_family,
        "enable_crc": True,
        "max_retry_count": 5,
    }
    if config.idc:
        options["idc"] = config.idc
    return tosfsspec.TosFileSystem(**options)


def publish_file(path: Path, task_id: str) -> dict[str, Any]:
    config = TosConfig.from_env()
    if not config.enabled:
        raise RuntimeError("TOS output publishing is not configured")
    object_key = config.object_key(task_id)
    size = path.stat().st_size
    if size <= 0:
        raise RuntimeError("refusing to upload an empty video")
    _filesystem(config).put_file(
        str(path),
        f"tos://{config.bucket}/{object_key}",
        ContentType="video/mp4",
    )
    return {
        "provider": "tos",
        "bucket": config.bucket,
        "key": object_key,
        "url": config.public_url(object_key),
        "size": size,
    }
