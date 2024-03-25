FROM openvino/ubuntu18_dev:2020.3

ENV PROJECT=bsc_thesis
ENV PYTORCH_VERSION=1.8.1
ENV TORCHVISION_VERSION=0.9.1
ENV CUDNN_VERSION=7.6.5.32-1+cuda10.2
ENV NCCL_VERSION=2.7.8-1+cuda10.2
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

ARG python=3.6
ENV PYTHON_VERSION=${python}
ENV DEBIAN_FRONTEND=noninteractive

USER root

# Set default shell to /bin/bash
SHELL ["/bin/bash", "-cu"]

RUN apt-get update && apt-get install -y --allow-downgrades --allow-change-held-packages --no-install-recommends \
    build-essential \
    cmake \
    g++-4.8 \
    git \
    curl \
    docker.io \
    vim \
    wget \
    ca-certificates \
    libjpeg-dev \
    libpng-dev \
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-dev \
    python3-tk \
    librdmacm1 \
    libibverbs1 \
    ibverbs-providers \
    libgtk2.0-dev \
    unzip \
    bzip2 \
    htop \
    gnuplot \
    ffmpeg

RUN mkdir /workspace

RUN mkdir /workspace/${PROJECT}

WORKDIR /workspace/${PROJECT}
COPY . /workspace/${PROJECT}

ENV PYTHONPATH="/workspace/${PROJECT}:$PYTHONPATH"