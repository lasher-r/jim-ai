FROM nvcr.io/nvidia/l4t-pytorch:r32.7.1-pth1.10-py3

# nuke any Kitware refs anywhere under /etc/apt
RUN set -eux; \
    find /etc/apt -type f -name '*.list' -exec sed -i '/apt\.kitware\.com/d' {} +; \
    rm -f /etc/apt/sources.list.d/kitware* || true

# install OpenCV + v4l + Flask
RUN apt-get update && \
    apt-get install -y --no-install-recommends python3-opencv v4l-utils && \
    pip3 install --no-cache-dir flask && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
