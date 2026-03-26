# Headless Brush training. GPU drivers come from the host via
# nvidia-container-toolkit or by mounting /usr/share/vulkan/icd.d.
# Falls back to mesa software rendering if neither is available.
#
#   docker build -t brush -f extras/Dockerfile .
#   docker run --gpus all -v ./data:/data brush /data/my_scene

FROM rust:1.93 AS builder
WORKDIR /build
COPY . .
RUN cargo build --release -p brush-app

FROM ubuntu:24.04
ENV DEBIAN_FRONTEND=noninteractive
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=all

RUN apt-get update && apt-get install -y --no-install-recommends \
        libvulkan1 mesa-vulkan-drivers \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /build/target/release/brush /usr/local/bin/brush

RUN useradd -m brush && mkdir /data && chown brush:brush /data
USER brush
WORKDIR /data
ENTRYPOINT ["brush"]
