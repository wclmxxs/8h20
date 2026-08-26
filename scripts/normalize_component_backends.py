"""Normalize component attention backends required by the H20 stack."""

from __future__ import annotations

import sys

REQUIRED_BACKENDS = (
    ("text_encoder", "torch_sdpa"),
    ("audio_vae", "fa"),
    ("video_vae", "fa"),
    ("transformer", "sol_attn"),
)


def normalize_component_backends(spec: str) -> str:
    values: dict[str, str] = {}
    user_order: list[str] = []
    for raw_item in spec.split(","):
        item = raw_item.strip()
        if not item:
            continue
        key, separator, value = item.partition("=")
        key = key.strip()
        value = value.strip()
        if not separator or not key or not value:
            raise ValueError(
                "component attention backends must be comma-separated name=value pairs"
            )
        if key not in values:
            user_order.append(key)
        values[key] = value

    for key, value in REQUIRED_BACKENDS:
        values[key] = value

    required_names = [key for key, _ in REQUIRED_BACKENDS]
    output_order = required_names + [
        key for key in user_order if key not in required_names
    ]
    return ",".join(f"{key}={values[key]}" for key in output_order)


def main() -> int:
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} COMPONENT_BACKENDS", file=sys.stderr)
        return 2
    try:
        print(normalize_component_backends(sys.argv[1]))
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
