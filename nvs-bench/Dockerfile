# Simple docker file copied from extras/Dockerfile
# without the modal breaking changes (workdir and users)

FROM rust:1.88 AS builder
RUN git clone https://github.com/ArthurBrussee/brush
RUN cargo build --release --manifest-path ./brush/Cargo.toml

FROM ubuntu:22.04

# Install dependencies.
RUN apt-get update && DEBIAN_FRONTEND="noninteractive" apt-get install -y \
    build-essential \
    cmake \
    libvulkan1 \
    vulkan-tools \
    mesa-vulkan-drivers \
    nvidia-driver-570 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir /workdir
COPY --from=builder ./brush/target/release/brush_app /workdir
WORKDIR /workdir