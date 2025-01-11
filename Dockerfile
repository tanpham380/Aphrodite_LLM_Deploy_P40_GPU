FROM alpindale/aphrodite-openai:latest

# Keep existing CUDA environment settings
ENV CUDA_VERSION=12.1.1
ENV NV_CUDA_CUDART_VERSION=${CUDA_VERSION}-1
ENV NVIDIA_REQUIRE_CUDA=cuda>=12.1 brand=tesla,driver>=470,driver<471 brand=unknown,driver>=470,driver<471 brand=nvidia,driver>=470,driver<471 brand=nvidiartx,driver>=470,driver<471 brand=geforce,driver>=470,driver<471 brand=geforcertx,driver>=470,driver<471 brand=quadro,driver>=470,driver<471 brand=quadrortx,driver>=470,driver<471 brand=titan,driver>=470,driver<471 brand=titanrtx,driver>=470,driver<471 brand=tesla,driver>=525,driver<526 brand=unknown,driver>=525,driver<526 brand=nvidia,driver>=525,driver<526 brand=nvidiartx,driver>=525,driver<526 brand=geforce,driver>=525,driver<526 brand=geforcertx,driver>=525,driver<526 brand=quadro,driver>=525,driver<526 brand=quadrortx,driver>=525,driver<526 brand=titan,driver>=525,driver<526 brand=titanrtx,driver>=525,driver<526 brand=tesla,driver>=535,driver<536 brand=unknown,driver>=535,driver<536 brand=nvidia,driver>=535,driver<536 brand=nvidiartx,driver>=535,driver<536 brand=geforce,driver>=535,driver<536 brand=geforcertx,driver>=535,driver<536 brand=quadro,driver>=535,driver<536 brand=quadrortx,driver>=535,driver<536 brand=titan,driver>=535,driver<536 brand=titanrtx,driver>=535,driver<536

# Install dependencies
RUN pip install --no-cache-dir \
    partial-json-parser \
    einops \
    ray

# Copy package structure
COPY ./aphrodite-engine/aphrodite /usr/local/lib/python3.10/dist-packages/aphrodite

# Set permissions
RUN chmod -R 755 /usr/local/lib/python3.10/dist-packages/aphrodite

ENTRYPOINT ["python3", "-m", "aphrodite.endpoints.openai.api_server"]