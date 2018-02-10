import torch.nn as nn


def dummy_func():
    return


def weight_norm_rnn(rnn_layer, bidirectional=False):
    """Apply weight normalization on a single-layer RNN"""
    rnn_layer = nn.utils.weight_norm(
        rnn_layer, "weight_ih_l0")
    rnn_layer = nn.utils.weight_norm(
        rnn_layer, "weight_hh_l0")
    if bidirectional:
        rnn_layer = nn.utils.weight_norm(
            rnn_layer, "weight_hh_l0_reverse")
    rnn_layer.flatten_parameters = dummy_func
    return rnn_layer
