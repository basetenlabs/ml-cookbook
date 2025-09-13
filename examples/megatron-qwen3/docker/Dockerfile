FROM nvidia/cuda:12.6.3-cudnn-devel-ubuntu22.04

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
RUN apt update && apt install -y git build-essential

ENV PATH="/bin:$PATH"

RUN apt install -y python3.10 python3-pip
# Create symlinks instead of alias (aliases don't work in Dockerfile)
RUN ln -s /usr/bin/python3.10 /usr/bin/python
# RUN ln -s /usr/bin/pip3 /usr/bin/pip

# Set CUDA environment variables BEFORE installing packages
# ENV CUDA_HOME=/usr/local/cuda
# ENV PATH="$CUDA_HOME/bin:$PATH"
# ENV LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
# ENV NVCC_PREPEND_FLAGS="--forward-unknown-to-host-compiler"

# Verify CUDA installation
RUN nvcc --version
RUN echo $CUDA_HOME

# Copy requirements first for better caching
# COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install packaging setuptools wheel ninja
RUN pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
# RUN pip install torch==2.6.0+cu126 --index-url https://download.pytorch.org/whl/cu126

RUN pip install transformers==4.55.4
RUN pip install flash_attn==2.7.4.post1
# RUN pip install flash_attn-2.7.1.post1+cu12torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
RUN pip install ms-swift==3.7.2 hf_transfer==0.1.9
RUN pip install git+https://github.com/NVIDIA/TransformerEngine.git@stable
RUN pip install "numpy<2.0"
# RUN pip install -r requirements.txt

WORKDIR /tmp
RUN git clone https://github.com/NVIDIA/apex
WORKDIR /tmp/apex
RUN NVCC_APPEND_FLAGS="--threads 4" APEX_PARALLEL_BUILD=8 APEX_CPP_EXT=1 APEX_CUDA_EXT=1 APEX_FAST_MULTIHEAD_ATTN=1 APEX_FUSED_CONV_BIAS_RELU=1 pip install -v --no-build-isolation .

# Set working directory back to root or your app directory
WORKDIR /