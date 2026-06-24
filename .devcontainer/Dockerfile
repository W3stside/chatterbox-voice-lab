# ROCm Python devcontainer for chatterbox-voice-lab — Ghost tower (RX 7900 XTX, gfx1100).
#
# Air-gap model: the ROCm USERSPACE + torch + chatterbox live HERE, in the container.
# The host keeps only the amdgpu kernel driver + /dev/kfd + /dev/dri (passed in via the
# runArgs in devcontainer.json). Nothing dev-related touches the host.
#
# Mirrors the validated native setup: plain Ubuntu base + pip torch 2.6.0+rocm6.2.4.
# The torch ROCm wheel is self-contained (bundles rocBLAS/MIOpen/HIP), so NO /opt/rocm
# system install is needed — exactly how it ran natively. Do NOT bump torch past 2.6:
# 2.7+ has a gfx1100 hipBLASLt regression (pytorch#150155).
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-venv python3-pip \
        ffmpeg libsndfile1 libnuma1 libdrm2 \
        git curl ca-certificates openssh-client \
    && rm -rf /var/lib/apt/lists/*

# Isolated venv (keeps the image's system python untouched; also dodges PEP 668).
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir --upgrade pip

# torch FIRST from the ROCm index, so chatterbox-tts (pinned torch==2.6.0) is already
# satisfied and never drags the CUDA build from PyPI (the trap we hit natively).
# `==2.6.0` is satisfied by the local `2.6.0+rocm6.2.4`.
RUN pip install --no-cache-dir torch==2.6.0 torchaudio==2.6.0 \
        --index-url https://download.pytorch.org/whl/rocm6.2.4

# App deps (requirements.txt does NOT pin torch — installed above).
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# gfx1100 override (matches host); harmless on a natively-supported card.
ENV HSA_OVERRIDE_GFX_VERSION=11.0.0
# HF + MIOpen caches land on volume mounts (see devcontainer.json) so model downloads
# and MIOpen tuning persist across rebuilds.
ENV HF_HOME=/root/.cache/huggingface

WORKDIR /workspaces/chatterbox-voice-lab
