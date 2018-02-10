FROM ceshine/cuda-pytorch:0.3.0

MAINTAINER CeShine Lee <ceshine@ceshine.net>

RUN pip install -U tqdm joblib sacred tensorboardX tensorflow==1.5.0

RUN mkdir /home/docker/labs
COPY . /home/docker/labs
WORKDIR /home/docker/labs
