import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import transformer.Constants as Constants
from transformer.Layers import EncoderLayer, DecoderLayer
from transformer.Models import position_encoding_init, get_attn_padding_mask, get_attn_subsequent_mask

from locked_dropout import LockedDropout


class TransformerEncoder(nn.Module):
    ''' A encoder model with self attention mechanism. '''

    def __init__(self, n_max_seq, n_layers=6, n_head=8, d_k=64, d_v=64,
                 d_word_vec=512, d_model=512, d_inner_hid=1024, dropout=0.1):

        super(TransformerEncoder, self).__init__()

        n_position = n_max_seq + 1
        self.n_max_seq = n_max_seq
        self.d_model = d_model
        self.position_enc = nn.Embedding(
            n_position, d_word_vec, padding_idx=Constants.PAD)
        self.position_enc.weight.data = position_encoding_init(
            n_position, d_word_vec)
        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner_hid, n_head,
                         d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])

    def forward(self, src_seq, src_pos, return_attns=False):
        enc_input = src_seq
        # Position Encoding addition
        enc_input += self.position_enc(src_pos)
        if return_attns:
            enc_slf_attns = []
        enc_output = enc_input
        enc_slf_attn_mask = get_attn_padding_mask(
            src_seq[:, :, 0], src_seq[:, :, 0])
        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(
                enc_output, slf_attn_mask=enc_slf_attn_mask)
            if return_attns:
                enc_slf_attns += [enc_slf_attn]
        if return_attns:
            return enc_output, enc_slf_attns
        return enc_output,


class TransformerDecoder(nn.Module):
    ''' A decoder model with self attention mechanism. '''

    def __init__(
            self, n_max_seq, n_layers=6, n_head=8, d_k=64, d_v=64,
            d_word_vec=512, d_model=512, d_inner_hid=1024, dropout=0.1):

        super(TransformerDecoder, self).__init__()
        n_position = n_max_seq + 1
        self.n_max_seq = n_max_seq
        self.d_model = d_model
        self.position_enc = nn.Embedding(
            n_position, d_word_vec, padding_idx=Constants.PAD)
        self.position_enc.weight.data = position_encoding_init(
            n_position, d_word_vec)
        self.dropout = nn.Dropout(dropout)
        self.layer_stack = nn.ModuleList([
            DecoderLayer(d_model, d_inner_hid, n_head,
                         d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])

    def forward(self, tgt_seq, tgt_pos, src_seq, enc_output, return_attns=False):
        dec_input = tgt_seq
        # Position Encoding addition
        dec_input += self.position_enc(tgt_pos)
        # Decode
        dec_slf_attn_pad_mask = get_attn_padding_mask(
            tgt_seq[:, :, 0], tgt_seq[:, :, 0])
        dec_slf_attn_sub_mask = get_attn_subsequent_mask(tgt_seq[:, :, 0])
        dec_slf_attn_mask = torch.gt(
            dec_slf_attn_pad_mask + dec_slf_attn_sub_mask, 0)
        dec_enc_attn_pad_mask = get_attn_padding_mask(
            tgt_seq[:, :, 0], src_seq[:, :, 0])
        if return_attns:
            dec_slf_attns, dec_enc_attns = [], []
        dec_output = dec_input
        for dec_layer in self.layer_stack:
            dec_output, dec_slf_attn, dec_enc_attn = dec_layer(
                dec_output, enc_output,
                slf_attn_mask=dec_slf_attn_mask,
                dec_enc_attn_mask=dec_enc_attn_pad_mask)
            if return_attns:
                dec_slf_attns += [dec_slf_attn]
                dec_enc_attns += [dec_enc_attn]
        if return_attns:
            return dec_output, dec_slf_attns, dec_enc_attns
        return dec_output,
