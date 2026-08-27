#!/usr/bin/env python3
"""Keep a Bernard debug container alive and reap detached debug children."""

from __future__ import annotations

import os
import signal
import time


stopping = False


def request_stop(signum: int, _frame: object) -> None:
    global stopping
    print(f"DEBUG_HOLD: received signal {signum}; stopping", flush=True)
    stopping = True


def reap_children() -> None:
    while True:
        try:
            pid, status = os.waitpid(-1, os.WNOHANG)
        except ChildProcessError:
            return
        if pid == 0:
            return
        print(f"DEBUG_HOLD: reaped child pid={pid} status={status}", flush=True)


def main() -> None:
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    print(
        "DEBUG_HOLD_READY: PID 1 will stay alive and reap manually launched services",
        flush=True,
    )
    while not stopping:
        reap_children()
        time.sleep(1)
    reap_children()


if __name__ == "__main__":
    main()
