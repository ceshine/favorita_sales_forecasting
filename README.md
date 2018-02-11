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

### Preprocessing

```
python prepare_seq_data.py
```

### Train Model

For now there are two types of model ready to be trained:
1. Transformer (fit_transformer.py)
2. LSTNet (fit_lstnet.py)

The training scripts use [Sacred](http://sacred.readthedocs.io/en/latest/) to manage experiments. It is recommended to set a seed explicitly via CLI:

```
python fit_transformer.py with seed=93102
```

You can also use Mongo to save experiment results and hyper-parameters for each run. Please refer to the Sacred documentation for more details.

### Prediction for Validation and Testing Dataset

The CSV output will be saved in `cache/preds/val/` and `cache/preds/test/` respectively.

### Tensorboard

Training and validation loss curves, and some of the embeddings are logged in tensorboard format. Launch tensorboad via:

```
tensorboard --logdir runs
```

Then visit http://localhost:6006 for the web interface.

## TODO (For now you need to figure them out yourself)

1. Ensembling script: I made some changes to the outputs of model training scripts so they are more readable. But that means ensembling script needs to be updated as well. (For those who want to try: the ground truth for validation set is stored in `cache/yval_seq.npy`.)
2. Encoder/Decoder and Encoder/MLP models with LSTM, GRU, QRNN, SRU units: I tried a lot of different stuffs for this competition. But I feel the code could use some refactoring, so they are removed for now.
3. Tabular data preparation and models: My GBM models is mediocre at best, so not really worth sharing here. But as I mentioned in the blog post. For those store/item combination that were removed by the 56-day nonzero filter, using a GBM model to predict values for them will give you a better score than predicting zeros.
