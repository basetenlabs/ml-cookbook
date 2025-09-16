FROM --platform=linux/amd64 nvidia/cuda:12.6.3-cudnn-devel-ubuntu22.04

# Install system dependencies (excluding Python)
RUN apt update && apt install -y git build-essential curl && \
    apt clean && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
ENV PATH="/bin:$PATH"

# Install Python using uv
RUN uv python install 3.10
ENV PATH="/root/.local/bin:$PATH"

# Set CUDA environment variables
ENV CUDA_HOME=/usr/local/cuda
ENV PATH="$CUDA_HOME/bin:$PATH"
ENV LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
ENV NVCC_PREPEND_FLAGS="--forward-unknown-to-host-compiler"

# Set compilation optimization flags
# ENV CFLAGS="-O2"
# ENV CXXFLAGS="-O2"
# ENV CUDA_ARCH_LIST="60;61;62;70;75;80;86;90"

# Verify CUDA installation
# RUN nvcc --version

# Set working directory
WORKDIR /app

# Create virtual environment and activate it
RUN uv venv
ENV PATH="/app/.venv/bin:$PATH"

# Install torch first (needed for flash-attn build)
RUN uv pip install torch==2.7.1

# Install other base dependencies
RUN uv pip install transformers==4.55.4 ms-swift==3.7.2 hf-transfer==0.1.9 packaging==25.0 "numpy<2.0"

# Install flash-attn with no build isolation (needs torch to be available)
RUN uv pip install --no-build-isolation flash-attn==2.7.4.post1

# Install transformer-engine with proper build isolation settings
RUN uv pip install --no-build-isolation transformer-engine[pytorch]

# Add swap space to handle memory spikes during compilation
# RUN fallocate -l 4G /swapfile && \
#     chmod 600 /swapfile && \
#     mkswap /swapfile && \
#     swapon /swapfile

# Install NVIDIA Apex with memory-optimized settings
WORKDIR /tmp
RUN git clone https://github.com/NVIDIA/apex
WORKDIR /tmp/apex
RUN NVCC_APPEND_FLAGS="--threads 2" APEX_PARALLEL_BUILD=2 APEX_CPP_EXT=1 APEX_CUDA_EXT=1 APEX_FAST_MULTIHEAD_ATTN=1 APEX_FUSED_CONV_BIAS_RELU=1 MAX_JOBS=2 uv pip install -v --no-build-isolation .

# Clean up build artifacts and temporary files
# RUN rm -rf /tmp/apex && \
#     rm -f /swapfile && \
#     apt autoremove -y && \
#     apt autoclean

# Set working directory back to app directory
WORKDIR /app