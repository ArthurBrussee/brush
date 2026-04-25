"""
Run brush tests on a Modal GPU container.

Modal's free CI tier doesn't exist; this is paid. Pricing for the cheapest
GPU (T4) is roughly $0.59/hr at second-grain billing — a 5-minute CI run is
about $0.05.

Invocation (from CI):

    pip install modal
    python -m modal token set-from-env
    modal run ci/modal_runner.py::run_native
    modal run ci/modal_runner.py::run_wasm

Authentication: set `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET` as environment
variables (in GH Actions, drive these from repository secrets).

The container image is defined inline below. First run takes a few minutes
to build; subsequent runs reuse Modal's image cache (bumping the layer hash
forces a rebuild — keep changes small).
"""
import subprocess
import sys
from pathlib import Path

import modal

REPO_ROOT = Path(__file__).resolve().parent.parent

# Container image: NVIDIA CUDA base for the GPU drivers, then layer on
# Rust, Chrome, chromedriver, and the Vulkan/wgpu prereqs that brush
# needs at runtime.
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.0-runtime-ubuntu22.04",
        add_python="3.11",
    )
    .apt_install(
        "curl",
        "git",
        "build-essential",
        "pkg-config",
        "libssl-dev",
        "libxcb-xfixes0-dev",
        "libxkbcommon-dev",
        "libgtk-3-dev",
        "vulkan-tools",
        "libvulkan1",
        "mesa-vulkan-drivers",
        "libegl1",
        "libgles2",
        # Chrome runtime deps:
        "fonts-liberation",
        "libasound2",
        "libatk-bridge2.0-0",
        "libatk1.0-0",
        "libatspi2.0-0",
        "libnspr4",
        "libnss3",
        "libxcomposite1",
        "libxdamage1",
        "libxfixes3",
        "libxrandr2",
        "xdg-utils",
    )
    .run_commands(
        # Install Rust 1.93.0 with the wasm target.
        "curl -fsSL https://sh.rustup.rs "
        "| sh -s -- -y --default-toolchain 1.93.0 "
        "--target wasm32-unknown-unknown --profile minimal",
    )
    .env({"PATH": "/root/.cargo/bin:/usr/local/cuda/bin:${PATH}"})
    .run_commands(
        # Install Chrome for Testing + matching chromedriver via the same
        # python helper we used in the GH Actions runner setup.
        "pip install --no-cache-dir chromedriver-autoinstaller",
        # Pre-fetch Chrome stable so the image is ready to test.
        "curl -fsSL "
        "https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb "
        "-o /tmp/chrome.deb",
        "apt-get install -y /tmp/chrome.deb && rm /tmp/chrome.deb",
        # chromedriver-autoinstaller needs to write somewhere; pre-install.
        "python3 -c 'import chromedriver_autoinstaller; "
        "chromedriver_autoinstaller.install()'",
        # wasm-bindgen test runner.
        "cargo install wasm-bindgen-cli --version 0.2.108 --locked",
    )
)

app = modal.App("brush-ci")


def _ignore_workspace(path: Path) -> bool:
    """Skip build/git/node noise when uploading the workspace."""
    s = str(path)
    return "/target/" in s or "/.git/" in s or "/node_modules/" in s


# Attach the local workspace to the runtime image at /workspace. Modal
# de-dupes file blobs across runs, so subsequent uploads are quick.
image = image.add_local_dir(
    str(REPO_ROOT),
    "/workspace",
    ignore=_ignore_workspace,
)


def _run(cmd: list[str], env: dict[str, str] | None = None) -> None:
    """Stream subprocess output to Modal's logs and raise on non-zero exit."""
    print(f"+ {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd, cwd="/workspace", env=env)
    if result.returncode != 0:
        sys.exit(result.returncode)


@app.function(image=image, gpu="T4", timeout=60 * 60)
def run_native() -> None:
    import os

    print("::group::nvidia-smi", flush=True)
    subprocess.run(["nvidia-smi"], check=False)
    print("::endgroup::", flush=True)
    print("::group::vulkaninfo --summary", flush=True)
    subprocess.run(["vulkaninfo", "--summary"], check=False)
    print("::endgroup::", flush=True)

    env = {**os.environ, "RUSTFLAGS": "-D warnings"}
    _run(["cargo", "test", "--all", "--all-features"], env=env)
    _run(["cargo", "test", "--all", "--all-features", "--doc"], env=env)


@app.function(image=image, gpu="T4", timeout=60 * 60)
def run_wasm() -> None:
    import os

    print("::group::nvidia-smi", flush=True)
    subprocess.run(["nvidia-smi"], check=False)
    print("::endgroup::", flush=True)
    print("::group::vulkaninfo --summary", flush=True)
    subprocess.run(["vulkaninfo", "--summary"], check=False)
    print("::endgroup::", flush=True)

    # chromedriver-autoinstaller installs into the python user dir on the
    # baked image; resolve where it landed so wasm-bindgen-test can find it.
    driver_path = subprocess.check_output(
        [
            "python3",
            "-c",
            "import chromedriver_autoinstaller; "
            "print(chromedriver_autoinstaller.install())",
        ],
        text=True,
    ).strip().splitlines()[-1]
    print(f"chromedriver: {driver_path}", flush=True)

    env = {
        **os.environ,
        "RUSTFLAGS": '--cfg=getrandom_backend="wasm_js"',
        "CHROMEDRIVER": driver_path,
    }
    _run(
        [
            "cargo",
            "test",
            "-p",
            "brush-bench-test",
            "--target",
            "wasm32-unknown-unknown",
        ],
        env=env,
    )
