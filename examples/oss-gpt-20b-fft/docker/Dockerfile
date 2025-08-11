FROM nvidia/cuda:12.8.0-devel-ubuntu22.04

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
RUN apt update && apt install git -y

ENV PATH="/bin:$PATH"

RUN uv venv gpt-oss --python 3.11 && uv venv
RUN uv pip install torch==2.8.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/test/cu128
RUN uv pip install git+https://github.com/triton-lang/triton.git@main#subdirectory=python/triton_kernels

# Copy requirements first for better caching
COPY requirements.txt .
RUN uv pip install -r requirements.txt
