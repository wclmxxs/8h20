#!/usr/bin/env python3
from __future__ import annotations

import ipaddress
import os
import socket
import subprocess

import uvicorn


def create_listeners(port: int) -> list[socket.socket]:
    listeners: list[socket.socket] = []
    try:
        ipv4 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listeners.append(ipv4)
        ipv4.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        ipv4.bind(("0.0.0.0", port))
        actual_port = int(ipv4.getsockname()[1])
        ipv4.listen(2048)
        ipv4.setblocking(False)

        ipv6 = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
        listeners.append(ipv6)
        ipv6.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        # Keep IPv6 separate from the explicit IPv4 socket so health checks
        # remain portable across different net.ipv6.bindv6only settings.
        ipv6.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 1)
        ipv6.bind(("::", actual_port))
        ipv6.listen(2048)
        ipv6.setblocking(False)
        return listeners
    except Exception:
        for listener in listeners:
            listener.close()
        raise


def discover_advertise_ip() -> ipaddress.IPv4Address | ipaddress.IPv6Address:
    explicit = os.getenv("PUBLIC_ADVERTISE_IP", "").strip().strip("[]")
    if explicit:
        return ipaddress.ip_address(explicit)

    output = subprocess.check_output(
        ["hostname", "-I"], text=True, timeout=5
    ).split()
    addresses: list[ipaddress.IPv4Address | ipaddress.IPv6Address] = []
    for value in output:
        try:
            address = ipaddress.ip_address(value)
        except ValueError:
            continue
        if not address.is_loopback and not address.is_link_local:
            addresses.append(address)

    global_ipv6 = next(
        (
            address
            for address in addresses
            if isinstance(address, ipaddress.IPv6Address) and address.is_global
        ),
        None,
    )
    if global_ipv6 is not None:
        return global_ipv6
    if addresses:
        return addresses[0]
    raise RuntimeError("no non-loopback address is available for PUBLIC_BASE_URL")


def configure_public_base_url(port: int) -> str:
    configured = os.getenv("PUBLIC_BASE_URL", "").strip().rstrip("/")
    if configured:
        os.environ["PUBLIC_BASE_URL"] = configured
        return configured

    address = discover_advertise_ip()
    host = f"[{address}]" if isinstance(address, ipaddress.IPv6Address) else str(address)
    base_url = f"http://{host}:{port}"
    os.environ["PUBLIC_BASE_URL"] = base_url
    return base_url


def main() -> None:
    port = int(os.getenv("PORT", os.getenv("TCE_SERVICE_PORT", "30010")))
    public_base_url = configure_public_base_url(port)
    listeners = create_listeners(port)
    print(
        f"API listening on IPv4 0.0.0.0:{port} and IPv6 [::]:{port}; "
        f"content base URL={public_base_url}",
        flush=True,
    )
    config = uvicorn.Config(
        "app.server:app",
        workers=1,
        timeout_keep_alive=120,
    )
    uvicorn.Server(config).run(sockets=listeners)


if __name__ == "__main__":
    main()
