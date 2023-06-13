FROM ubuntu:22.10

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Lisbon \ DEBIAN_FRONTEND=noninteractive

RUN apt update && apt -y upgrade
RUN apt-get install -y wget
RUN apt-get update && apt-get install -y ffmpeg libsm6 libxext6
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda clean --all --yes
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PATH /opt/conda/bin:$PATH
RUN conda config --append channels conda-forge
RUN conda create -n mosquinha
SHELL ["conda", "run", "-n", "mosquinha", "/bin/bash", "-c"]
RUN pip install torchvision
RUN pip install torch~=2.0.0
RUN pip install scikit-learn~=1.2.2 pandas~=1.5.3
RUN pip install matplotlib~=3.7.1 seaborn~=0.12.2 tqdm~=4.65.0
RUN pip install PyYAML~=6.0 Pillow~=9.4.0 numpy~=1.24.2 opencv-python~=4.7.0.72
RUN pip install rembg

COPY Classify.py /opt
COPY Config.py /opt
COPY Learn.py /opt
COPY Performance.py /opt
COPY PreProcess.py /opt
COPY main.py /opt
COPY snr_contrast.py /opt

WORKDIR /data
