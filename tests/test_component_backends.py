import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts/normalize_component_backends.py"
SPEC = importlib.util.spec_from_file_location("normalize_component_backends", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_incomplete_legacy_config_is_repaired():
    assert MODULE.normalize_component_backends(
        "text_encoder=torch_sdpa,transformer=sol_attn"
    ) == (
        "text_encoder=torch_sdpa,audio_vae=fa,video_vae=fa,transformer=sol_attn"
    )


def test_required_backends_override_unsafe_values_and_preserve_extensions():
    assert MODULE.normalize_component_backends(
        "transformer=fa,audio_vae=sol_attn,custom_encoder=fa"
    ) == (
        "text_encoder=torch_sdpa,audio_vae=fa,video_vae=fa,"
        "transformer=sol_attn,custom_encoder=fa"
    )


@pytest.mark.parametrize("spec", ["audio_vae", "=fa", "audio_vae="])
def test_malformed_config_fails_before_containers_start(spec):
    with pytest.raises(ValueError, match="name=value"):
        MODULE.normalize_component_backends(spec)


def test_install_normalizes_before_loading_env():
    install = (ROOT / "install.sh").read_text()
    normalization = "python3 scripts/normalize_component_backends.py"
    assert normalization in install
    assert install.index(normalization) < install.index("source .env")
