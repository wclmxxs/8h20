from __future__ import annotations

import importlib.util
import socket
from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parents[1] / "api/run_dual_stack.py"
SPEC = importlib.util.spec_from_file_location("run_dual_stack", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_create_listeners_binds_ipv4_and_ipv6_to_the_same_port():
    listeners = MODULE.create_listeners(0)
    try:
        assert [listener.family for listener in listeners] == [
            socket.AF_INET,
            socket.AF_INET6,
        ]
        assert listeners[0].getsockname()[1] == listeners[1].getsockname()[1]
        assert (
            listeners[1].getsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY) == 1
        )
    finally:
        for listener in listeners:
            listener.close()


def test_create_listeners_uses_loopback_behind_http_mesh():
    listeners = MODULE.create_listeners(0, mesh_ingress=True)
    try:
        assert len(listeners) == 1
        assert listeners[0].family == socket.AF_INET
        assert listeners[0].getsockname()[0] == "127.0.0.1"
    finally:
        for listener in listeners:
            listener.close()


def test_resolve_port_prefers_mesh_ingress_port(monkeypatch):
    monkeypatch.delenv("PORT0", raising=False)
    monkeypatch.setenv("MESH_INGRESS_PORT", "11288")
    monkeypatch.setenv("PORT", "30010")

    assert MODULE.resolve_port(mesh_ingress=True) == 11288
    assert MODULE.resolve_port(mesh_ingress=False) == 30010


def test_resolve_port_prefers_platform_port0(monkeypatch):
    monkeypatch.setenv("PORT0", "10111")
    monkeypatch.setenv("MESH_INGRESS_PORT", "11288")
    monkeypatch.setenv("PORT", "30010")

    assert MODULE.resolve_port(mesh_ingress=True) == 10111
    assert MODULE.resolve_port(mesh_ingress=False) == 10111


def test_http_mesh_ingress_enabled_accepts_platform_boolean(monkeypatch):
    monkeypatch.setenv("REQUIRE_HTTP_MESH", "True")

    assert MODULE.http_mesh_ingress_enabled() is True


def test_configure_public_base_url_prefers_the_global_ipv6(monkeypatch):
    monkeypatch.delenv("PUBLIC_BASE_URL", raising=False)
    monkeypatch.delenv("PUBLIC_ADVERTISE_IP", raising=False)
    monkeypatch.delenv("BYTED_HOST_IP", raising=False)
    monkeypatch.delenv("BYTED_HOST_IPV6", raising=False)
    monkeypatch.setattr(
        MODULE.subprocess,
        "check_output",
        lambda *args, **kwargs: (
            "26.33.135.130 2605:340:cd52:1900:e2bf:f442:842c:d69c "
            "fdbd:fdbd:fdbd:fdbd:ffff:ffff:0:1"
        ),
    )

    assert MODULE.configure_public_base_url(11637) == (
        "http://[2605:340:cd52:1900:e2bf:f442:842c:d69c]:11637"
    )


def test_configure_public_base_url_uses_byted_host_ipv6(monkeypatch):
    monkeypatch.delenv("PUBLIC_BASE_URL", raising=False)
    monkeypatch.delenv("PUBLIC_ADVERTISE_IP", raising=False)
    monkeypatch.setenv("BYTED_HOST_IP", "")
    monkeypatch.setenv(
        "BYTED_HOST_IPV6", "2605:340:cd52:1901:b4b2:9a04:bb74:ae3c"
    )

    assert MODULE.configure_public_base_url(10111) == (
        "http://[2605:340:cd52:1901:b4b2:9a04:bb74:ae3c]:10111"
    )


def test_configure_public_base_url_prefers_byted_host_ipv4(monkeypatch):
    monkeypatch.delenv("PUBLIC_BASE_URL", raising=False)
    monkeypatch.delenv("PUBLIC_ADVERTISE_IP", raising=False)
    monkeypatch.setenv("BYTED_HOST_IP", "10.20.30.40")
    monkeypatch.setenv(
        "BYTED_HOST_IPV6", "2605:340:cd52:1901:b4b2:9a04:bb74:ae3c"
    )

    assert MODULE.configure_public_base_url(10111) == "http://10.20.30.40:10111"


def test_configure_public_base_url_preserves_an_explicit_value(monkeypatch):
    monkeypatch.setenv("PUBLIC_BASE_URL", "https://video.example.test/root/")

    assert MODULE.configure_public_base_url(11637) == "https://video.example.test/root"
