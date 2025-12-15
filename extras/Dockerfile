# Simple docker file to run Brush. Note that as Brush has very few dependencies, you should be able to run
# it on any system already.
#
# This image is for easier server deployments. Nb: This dockerfile explicitly installs nvidia drivers,
# I haven't tested this on non nvidia GPUs. Alternatively, you should be able
# to run with host drivers.
FROM rust:1.86 AS builder
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
COPY --from=builder ./brush/target/release/brush /workdir
WORKDIR /workdir

# Below command creates home dir for 1000 UID user
RUN useradd -m -u 1000 clouduser
RUN chown -R 1000:root /workdir && chmod -R 775 /workdir

# Launch training.
USER clouduser
ENTRYPOINT ["./brush"]
