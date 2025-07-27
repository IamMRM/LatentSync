FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel

# Install sudo
USER root

RUN apt-get update && \
    apt-get install -y sudo git && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
RUN chmod -R 777 /tmp

COPY . /workspace/

# Verify installation
CMD ["python3", "-c", "import torch; print(torch.cuda.is_available())"]