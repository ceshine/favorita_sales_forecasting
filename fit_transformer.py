"""Fit a transformer model"""
import random
import logging
from datetime import datetime

import numpy as np
import torch
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import mean_squared_error
from transformer.Optim import ScheduledOptim
from sacred import Experiment

from bots import TransformerBot
from dataset import read_dataset
from io_utils import export_validation, export_test

logging.basicConfig(level=logging.WARNING)

ex = Experiment('Transformer')
ex.add_source_file("preprocess.py")
ex.add_source_file("prepare_seq_data.py")


@ex.named_config
def no_tf_2l():
    batch_size = 128
    model_details = {
        "odrop": 0.25,
        "edrop": 0.25,
        "hdrop": 0.1,
        "d_model": 128,
        "d_inner_hid": 256,
        "n_layers": 2,
        "n_head": 4,
        "propagate": False
    }


@ex.config
def no_tf_2l_256():
    batch_size = 128
    model_details = {
        "odrop": 0.25,
        "edrop": 0.25,
        "hdrop": 0.25,
        "d_model": 256,
        "d_inner_hid": 256,
        "n_layers": 2,
        "n_head": 4,
        "propagate": False
    }


@ex.named_config
def no_tf_1l():
    batch_size = 128
    model_details = {
        "odrop": 0.25,
        "edrop": 0.25,
        "hdrop": 0.1,
        "d_model": 128,
        "d_inner_hid": 256,
        "n_layers": 1,
        "n_head": 8,
        "propagate": False
    }


@ex.named_config
def no_tf_3l():
    batch_size = 128
    model_details = {
        "odrop": 0.25,
        "edrop": 0.25,
        "hdrop": 0.1,
        "d_model": 128,
        "d_inner_hid": 256,
        "n_layers": 3,
        "n_head": 2,
        "propagate": False
    }


@ex.automain
def main(batch_size, model_details, seed):
    train_dataset, val_dataset, test_dataset = read_dataset()

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    batches_per_epoch = len(train_dataset) // batch_size

    # Start Training
    bot = TransformerBot(
        train_dataset, test_dataset, val_dataset=val_dataset,
        n_layers=model_details.get("n_layers", 6),
        n_head=model_details.get("n_head", 8),
        d_model=model_details.get("d_model", 512),
        d_inner_hid=model_details.get("d_inner_hid", 1024),
        d_k=model_details.get("d_k", 32),
        d_v=model_details.get("d_v", 32),
        propagate=model_details.get("propagate", False),
        hdrop=model_details.get("hdrop", 0),
        edrop=model_details.get("edrop", 0),
        odrop=model_details.get("odrop", 0),
        avg_window=500,
        clip_grad=10,
        tf_warmup=int(batches_per_epoch),
        tf_decay=0.1 ** (1 / 6),
        tf_steps=batches_per_epoch // 200 * 100,
        tf_min=0.1
    )

    param_groups = [
        {
            "params": bot.model.get_trainable_parameters(), "lr": 5e-4
        }
    ]

    optimizer = optim.Adam(param_groups)
    scheduler = ReduceLROnPlateau(
        optimizer, factor=0.25, patience=5, cooldown=0,
        threshold=2e-4,
        min_lr=[x["lr"] * 0.25 ** 2 for x in param_groups]
    )

    # optimizer = ScheduledOptim(
    #     optim.Adam(
    #         bot.model.get_trainable_parameters(), betas=(0.9, 0.98), eps=1e-09),
    #     model_details.get("d_model", 512),
    #     model_details.get("train_warmup", 2000))
    # scheduler = None

    _ = bot.train(
        optimizer, batch_size=batch_size, n_epochs=20,
        seed=seed, log_interval=batches_per_epoch // 50,
        snapshot_interval=batches_per_epoch // 50 * 5,
        early_stopping_cnt=15,
        scheduler=scheduler)

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
