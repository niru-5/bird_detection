FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-devel
ARG DEBIAN_FRONTEND=noninteractive

ENV CUDA_HOME=/usr/local/cuda \
     TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 7.5 8.0 8.6+PTX" \
     SETUPTOOLS_USE_DISTUTILS=stdlib

RUN conda update conda -y

# Install libraries in the brand new image. 
RUN apt-get -y update && apt-get install -y --no-install-recommends \
         wget \
         build-essential \
         git \
         python3-opencv \
         ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Set the working directory for all the subsequent Dockerfile instructions.
WORKDIR /resources



RUN git clone https://github.com/IDEA-Research/GroundingDINO.git && \
     cd GroundingDINO && \
     mkdir weights && \
     cd weights && \
     wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth

# create a pip env 
RUN python -m venv pip_env && source pip_env/bin/activate && \ 
     pip install -U "numpy<2" ninja && \
     pip install "transformers<4.40" && \
     pip install "opencv-python<4.12" && \
     pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121 && \
     cd GroundingDINO/ && \
     pip install -e . --no-build-isolation


# RUN conda install -c "nvidia/label/cuda-12.1.1" cuda -y
# ENV CUDA_HOME=$CONDA_PREFIX

# ENV PATH=/usr/local/cuda/bin:$PATH

# RUN cd GroundingDINO/ && python -m pip install .

CMD sleep infinity

# COPY docker_test.py docker_test.py

# CMD [ "python", "docker_test.py" ]