"""Image definition (modal's pythonic version of a Dockerfile)

What you'll want to change:
- the base image to a cuda and torch version that matches your method's requirements. Though our defaults have worked
for most methods so far.
- if you already have a Dockerfile and want to keep using it, replace the Image.from_registry() line with Image.from_dockerfile("Dockerfile").
You might also want to change the `workdir` to keep it consistent with the Dockerfile's.
- add your installation commands in the bottom section. Modal's syntax almost identically follows dockerfile's.
  - note that if your install commands needs access to a gpu that's possible
  - also, avoid using conda and use pip instead (installing and initializing conda in dockerfiles has caused us a lot of problems)

See their docs for more info: https://modal.com/docs/guide/images
"""

from pathlib import Path, PurePosixPath

from modal import Image, Volume

method_name = Path.cwd().name
assert method_name != "nvs-bench", (
    "nvs-bench must be called from the method's directory, not the nvs-bench subdirectory. Eg: `modal run nvs-bench/image.py`."
)

nvs_bench_volume = Volume.from_name("nvs-bench", create_if_missing=True)

modal_volumes: dict[str | PurePosixPath, Volume] = {
    "/nvs-bench": nvs_bench_volume,
}

image = (
    Image.from_registry("pytorch/pytorch:2.4.1-cuda12.1-cudnn9-devel")  # find others at: https://hub.docker.com/
    .env(
        {
            # Set Torch CUDA Compatbility to be for RTX 4090, T4, L40s, and A100
            # If using a different GPU, make sure its torch cuda architecture version is added to the list
            "TORCH_CUDA_ARCH_LIST": "7.5;8.0;8.9;9.0",
            # Set environment variable to avoid interactive prompts from installing packages
            "DEBIAN_FRONTEND": "noninteractive",
            "TZ": "America/New_York",
        }
    )
    # Install git and various other helper dependencies
    .run_commands(
        "apt-get update && apt-get install -y \
            openssh-server \
            git \
            wget \
            unzip \
            cmake \
            build-essential \
            ninja-build \
            libglew-dev \
            libassimp-dev \
            libboost-all-dev \
            libgtk-3-dev \
            libopencv-dev \
            libglfw3-dev \
            libavdevice-dev \
            libavcodec-dev \
            libeigen3-dev \
            libtbb-dev \
            libopenexr-dev \
            libxi-dev \
            libxrandr-dev \
            libxxf86vm-dev \
            libxxf86dga-dev \
            libxxf86vm-dev"
    )
    # Install gsutil (for downloading datasets the first time)
    .apt_install("curl", "ca-certificates", "gnupg")
    .run_commands(
        "curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add -",
        "echo 'deb https://packages.cloud.google.com/apt cloud-sdk main' | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list",
        "apt-get update && apt-get install -y google-cloud-cli",
    )
    # For tracking GPU usage
    .run_commands("pip install gpu_tracker")
    # Set the working dir
    .workdir(f"/root/{method_name}")
    ######## START OF YOUR CODE ########
    # Probably easiest to pull the repo from github, but you can also copy files from your local machine with .add_local_dir()
    # eg: .run_commands("git clone -b nvs-bench https://github.com/N-Demir/gaussian-splatting.git --recursive .")
    # Install (avoid conda installs because they don't work well in dockerfile situations)
    # Separating these on separate lines helps if there are errors (previous lines will be cached) especially on the large package installs
    # eg:
    # .run_commands("pip install submodules/diff-gaussian-rasterization")
    # .run_commands("pip install -e .")
    # Note: If your run_commands step needs access to a gpu it's actually possible to do that through "run_commands(gpu='L40S', ...)"
)
