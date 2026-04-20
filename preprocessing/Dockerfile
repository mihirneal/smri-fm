FROM freesurfer/freesurfer:7.4.1

# Install uv and use it to manage a clean Python env separate from FreeSurfer's
# stripped Python 3.8 (which is missing _sqlite3 and other stdlib modules).
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

ENV UV_PYTHON_INSTALL_DIR=/opt/uv-python
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="/opt/venv/bin:${PATH}"
ENV TEMPLATEFLOW_HOME=/opt/templateflow
ENV PYTHON_VERSION=3.11

RUN uv python install ${PYTHON_VERSION} && \
    uv venv --python ${PYTHON_VERSION} /opt/venv && \
    uv pip install --python /opt/venv \
        "numpy<2" \
        nibabel antspyx templateflow surfa voxelmorph neurite \
        "tensorflow[and-cuda]==2.15.*" \
        "tf-keras==2.15.*"

# Expose the venv's bundled CUDA libs so TF can find them at runtime on the
# cluster's H100 nodes (driver 535 / CUDA 12.2).
ENV LD_LIBRARY_PATH="/opt/venv/lib/python3.11/site-packages/nvidia/cuda_runtime/lib:/opt/venv/lib/python3.11/site-packages/nvidia/cudnn/lib:/opt/venv/lib/python3.11/site-packages/nvidia/cublas/lib:/opt/venv/lib/python3.11/site-packages/nvidia/cufft/lib:/opt/venv/lib/python3.11/site-packages/nvidia/curand/lib:/opt/venv/lib/python3.11/site-packages/nvidia/cusolver/lib:/opt/venv/lib/python3.11/site-packages/nvidia/cusparse/lib:/opt/venv/lib/python3.11/site-packages/nvidia/nccl/lib:/opt/venv/lib/python3.11/site-packages/nvidia/nvjitlink/lib:/usr/lib64:${LD_LIBRARY_PATH}"

COPY preprocessing/ /app/preprocessing/

WORKDIR /app

ENTRYPOINT ["/opt/venv/bin/python", "/app/preprocessing/pipeline.py"]
