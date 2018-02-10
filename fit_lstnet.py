import os
import random
import logging
from datetime import datetime

import numpy as np
import torch
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import mean_squared_error
from sacred import Experiment

from dataset import read_dataset
from bots import LSTNetBot
from io_utils import export_validation, export_test

logging.basicConfig(level=logging.WARNING)

ex = Experiment('LSTNet')
ex.add_source_file("preprocess.py")
ex.add_source_file("prepare_seq_data.py")


@ex.named_config
def cnn_7():
    batch_size = 128
    ar_window_size = 28
    model_details = {
        "cnn_hidden_size": 256,
        "rnn_hidden_size": 256,
        "skip_hidden_size": 256,
        "skip": 7,
        "cnn_kernel": 7,
        "odrop": 0.25,
        "edrop": 0.25,
        "hdrop": 0.25,
        "unit_type": "GRU"
    }


@ex.config
def cnn_5():
    batch_size = 128
    ar_window_size = 28
    model_details = {
        "cnn_hidden_size": 256,
        "rnn_hidden_size": 256,
        "skip_hidden_size": 256,
        "skip": 7,
        "cnn_kernel": 5,
        "odrop": 0.25,
        "edrop": 0.25,
        "hdrop": 0.25,
        "unit_type": "GRU"
    }


@ex.named_config
def cnn_5_lstm():
    batch_size = 128
    ar_window_size = 28
    model_details = {
        "cnn_hidden_size": 256,
        "rnn_hidden_size": 256,
        "skip_hidden_size": 256,
        "skip": 7,
        "cnn_kernel": 5,
        "odrop": 0.25,
        "edrop": 0.25,
        "hdrop": 0.25,
        "unit_type": "LSTM"
    }


@ex.named_config
def cnn_3():
    batch_size = 128
    ar_window_size = 28
    model_details = {
        "cnn_hidden_size": 128,
        "rnn_hidden_size": 256,
        "skip_hidden_size": 128,
        "skip": 7,
        "cnn_kernel": 3,
        "odrop": 0.25,
        "edrop": 0.25,
        "hdrop": 0.1,
        "unit_type": "GRU"
    }


@ex.automain
def main(batch_size, ar_window_size, model_details, seed):
    train_dataset, val_dataset, test_dataset = read_dataset()

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    batches_per_epoch = len(train_dataset) // batch_size

    # Start Training
    bot = LSTNetBot(
        train_dataset, test_dataset, val_dataset=val_dataset,
        cnn_hidden_size=model_details["cnn_hidden_size"],
        rnn_hidden_size=model_details["rnn_hidden_size"],
        skip_hidden_size=model_details["skip_hidden_size"],
        skip=model_details["skip"], cnn_kernel=model_details["cnn_kernel"],
        hdrop=model_details["hdrop"], edrop=model_details["edrop"],
        odrop=model_details["odrop"], avg_window=500,
        ar_window_size=ar_window_size,
        clip_grad=10, unit_type=model_details["unit_type"])

    param_groups = [
        {
            "params": bot.model.parameters(), "lr": 5e-4
        }
    ]
    optimizer = optim.RMSprop(param_groups)
    scheduler = ReduceLROnPlateau(
        optimizer, factor=0.25, patience=4, cooldown=0,
        threshold=2e-4,
        min_lr=[x["lr"] * 0.25 ** 2 for x in param_groups]
    )

    _ = bot.train(
        optimizer, batch_size=batch_size, n_epochs=20,
        seed=seed, log_interval=batches_per_epoch // 50,
        snapshot_interval=batches_per_epoch // 50 * 5,
        early_stopping_cnt=15,
        scheduler=scheduler)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")

    val_pred = bot.predict_avg(is_test=False, k=8).cpu().numpy()
    weights = val_dataset.series_i[:, 0, -1] * .25 + 1
    score = mean_squared_error(val_dataset.y, val_pred, sample_weight=weights)
    export_validation("cache/preds/val/{}_{:.6f}_{}.csv".format(
        bot.name, score, timestamp), val_pred)

    test_pred = bot.predict_avg(is_test=True, k=8).cpu().numpy()
    export_test("cache/preds/test/{}_{:.6f}_{}.csv".format(
        bot.name, score, timestamp), test_pred)

    bot.logger.info("Score: {:.6f}".format(score))
    return score
