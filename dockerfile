FROM nvidia/cuda:10.2-base-ubuntu18.04

RUN apt-get update && apt-get install -y curl ca-certificates sudo git bzip2 libx11-6 \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir /app
WORKDIR /app
RUN adduser --disabled-password --gecos '' --shell /bin/bash user \
    && chown -R user:user /app
RUN echo "user ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/90-user
USER user

ENV HOME=/home/user
RUN chmod 777 /home/user

# Install Miniconda and Python 3.8.3
ENV CONDA_AUTO_UPDATE_CONDA=false
ENV PATH=/home/user/miniconda/bin:$PATH
RUN curl -sLo ~/miniconda.sh https://repo.continuum.io/miniconda/Miniconda3-py38_4.8.3-Linux-x86_64.sh \
    && chmod +x ~/miniconda.sh \
    && ~/miniconda.sh -b -p ~/miniconda \
    && rm ~/miniconda.sh \
    && conda install -y python==3.8.3 \
    && conda clean -ya

# install pytorch torchvision and lightning
RUN conda install pytorch torchvision cudatoolkit=10.2 -c pytorch \
    && pip install pytorch-lightning \
    && conda clean -ya
