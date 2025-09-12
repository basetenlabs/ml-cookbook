FROM nvidia/cuda:12.6.3-cudnn-devel-ubuntu22.04

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
RUN apt update && apt install git -y

ENV PATH="/bin:$PATH"

RUN apt install -y python3.10
RUN apt install -y python3-pip
RUN alias python=python3

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install -r requirements.txt

RUN git clone https://github.com/NVIDIA/apex
RUN cd apex
# APEX_CPP_EXT=1 APEX_CUDA_EXT=1 APEX_FAST_MULTIHEAD_ATTN=1 APEX_FUSED_CONV_BIAS_RELU=1 pip install -v --no-build-isolation . # to remove
RUN NVCC_APPEND_FLAGS="--threads 4" APEX_PARALLEL_BUILD=8 APEX_CPP_EXT=1 APEX_CUDA_EXT=1 APEX_FAST_MULTIHEAD_ATTN=1 APEX_FUSED_CONV_BIAS_RELU=1 pip install -v --no-build-isolation .
RUN cd ..