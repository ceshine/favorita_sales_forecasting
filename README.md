# (Simplified) Solution to Favorita Competition

Sorry, no CPU-only mode. You have to use an nvidia card to train models.

Test environment:
1. GTX 1070
2. 16 GB RAM + 8 GB Swap
3. At least 30 GB free disk space
  - (it can be less if you turn off some of the joblib disk caching)
4. Docker 17.12.0-ce
5. Nvidia-docker 2.0

## Acknowledgement

1. Transformer model comes from [Yu-Hsiang Huang's implementation](https://github.com/jadore801120/attention-is-all-you-need-pytorch). His repo is included in "*attention-is-all-you-need-pytorch*" folder via *git subtree*.
2. LSTNet model is largely inspired from [GUOKUN LAI's implementation](https://github.com/laiguokun/LSTNet).
3. The model structure is inspired by the work of  [Sean Vasquez](https://github.com/sjvasquez/web-traffic-forecasting) and [Arthur Suilin](https://github.com/Arturus/kaggle-web-traffic).

## Docker Usage

First build the image. Example command: `docker build -t favorita .`

Then spin up a docker container:
```
docker run --runtime=nvidia --rm -ti \
    -v /mnt/Data/favorita_cache:/home/docker/labs/cache \
    -v /mnt/Data/favorita_data:/home/docker/labs/data \
    -p 6006:6006 favorita bash
```

* It is recommended to manually mount the data and cache folder
* port 6006 is for running tensorboard inside the container

### Where to put the data
Download and extract the [data files from Kaggle](https://www.kaggle.com/c/favorita-grocery-sales-forecasting/data) into `data` folder.

We're going to assume you're using the BASH prompt inside the container in the rest of this README.

## Model Training

(Documentation WIP)
