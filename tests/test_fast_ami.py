import importlib.util
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_download_module():
    path = ROOT / "scripts" / "download_lora.py"
    spec = importlib.util.spec_from_file_location("h3_download_lora", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_trust_existing_size_skips_hash(monkeypatch, tmp_path, capsys):
    download_lora = load_download_module()
    cache_root = tmp_path / "hf-cache"
    cached = cache_root / "hub" / "snapshots" / "revision" / "lora.safetensors"
    cached.parent.mkdir(parents=True)
    cached.write_bytes(b"trusted-ami-lora")
    calls = []

    huggingface_hub = types.ModuleType("huggingface_hub")

    def fake_download(**kwargs):
        calls.append(kwargs)
        assert kwargs["local_files_only"] is True
        return str(cached)

    huggingface_hub.hf_hub_download = fake_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", huggingface_hub)
    monkeypatch.setattr(
        download_lora,
        "sha256_file",
        lambda _: (_ for _ in ()).throw(AssertionError("SHA256 must be skipped")),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "download_lora.py",
            "--cache-root",
            str(cache_root),
            "--repo",
            "unused/repo",
            "--revision",
            "revision",
            "--filename",
            "lora.safetensors",
            "--sha256",
            "0" * 64,
            "--size",
            str(cached.stat().st_size),
            "--trust-existing-size",
        ],
    )

    download_lora.main()

    assert len(calls) == 1
    assert (
        capsys.readouterr()
        .out.strip()
        .endswith("/hub/snapshots/revision/lora.safetensors")
    )


def test_install_has_safe_ami_fast_path_and_parallel_workers():
    install = (ROOT / "install.sh").read_text()

    assert "--from-ami" in install
    assert install.index("docker stop minimax-h3-h20-watchdog") < install.index(
        "scripts/bootstrap_host.sh"
    )
    assert "detected_advertise_host=$(detect_imds public-ipv4" in install
    assert "detected_instance_id=$(detect_imds instance-id" in install
    assert 'image inspect "${image}"' in install
    assert "--trust-existing-size" in install
    assert 'worker_services+=("h3-sglang-${slot}")' in install
    assert 'up -d "${worker_services[@]}"' in install
    assert "Waiting for partition ${slot}" in install
    assert 'http://127.0.0.1:${port}/healthz' in install
    assert 'http://[::1]:${port}/healthz' in install
    assert "IPv4+IPv6" in install
    assert "Waiting for queue watchdog" in install
    assert "WATCHDOG_STARTUP_TIMEOUT_SECONDS" in install
    assert 'verify_sol_stack.sh "minimax-h3-h20-sglang-${slot}"' in install
    assert "instance-store NVMe" in install


def test_prepare_ami_stops_reporter_before_all_containers():
    script = (ROOT / "prepare_ami.sh").read_text()

    monitor_stop = script.index("stop h3-watchdog")
    reporter_stop = script.index("stop h3-reporter", monitor_stop + 1)
    all_stop = script.index('"${compose[@]}" stop', reporter_stop + 1)
    assert monitor_stop < reporter_stop < all_stop
    assert "AMI_READY" in script


def test_api_update_does_not_restart_gpu_workers():
    script = (ROOT / "update_api.sh").read_text()

    assert "--no-deps --force-recreate" in script
    assert '"${api_services[@]}"' in script
    assert "h3-sglang" not in script
    assert "--timeout-keep-alive" in script
    assert 'http://[::1]:${port}/healthz' in script
    assert "dual-stack" in script
