"""End-to-end models
"""
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from components import (
    LockedDropout, TransformerEncoder, TransformerDecoder
)
from transformer.Modules import LayerNormalization
from weight_norm_rnn import weight_norm_rnn

TRAIN_PERIODS = 56


class BaseModel(nn.Module):
    def _create_layers(self, mlp=False):
        self.item_class_em = nn.Embedding(337, 8, max_norm=8, norm_type=2)
        self.item_family_em = nn.Embedding(33, 8, max_norm=8, norm_type=2)
        self.store_type_em = nn.Embedding(5, 2, max_norm=2, norm_type=2)
        self.store_cluster_em = nn.Embedding(17, 5, max_norm=5, norm_type=2)
        self.store_city_em = nn.Embedding(22, 5, max_norm=5, norm_type=2)
        self.store_em = nn.Embedding(54, 5, max_norm=5, norm_type=2)
        self.weekday_em = nn.Embedding(7, 3, max_norm=3, norm_type=2)
        self.day_em = nn.Embedding(31, 10, max_norm=10, norm_type=2)
        self.month_em = nn.Embedding(6, 2, max_norm=2, norm_type=2)
        self.year_em = nn.Embedding(3, 2, max_norm=2, norm_type=2)
        if not mlp:
            self.step_one_network = nn.Sequential(
                torch.nn.utils.weight_norm(nn.Linear(
                    self.hidden_size, self.hidden_size)),
                nn.ReLU(),
                nn.Dropout(self.odrop),
                torch.nn.utils.weight_norm(nn.Linear(self.hidden_size, 1))
            )

    def __init__(self, *, bidirectional, edrop, odrop, propagate, min_length, y_scale_by, steps=15):
        super(BaseModel, self).__init__()
        # numeric + derived + onehot + embeddings
        self.propagate = propagate
        self.input_dim = 4 + 6 + 4 + 8 + 8 + 2 + 5 + 5 + 5 + 3 + 10 + 2 + 2
        self.decode_dim = (
            self.input_dim + steps - 1 - int(not self.propagate))
        self.odrop = odrop
        self.bidirectional = bidirectional
        self.edrop = edrop
        self.min_length = min_length
        self.steps = steps
        self.y_scale_by = y_scale_by
        self.edropout = LockedDropout(batch_first=True)

    def init_weights(self, mlp=False):
        nn.init.orthogonal(self.store_type_em.weight)
        nn.init.orthogonal(self.store_cluster_em.weight)
        nn.init.orthogonal(self.store_city_em.weight)
        nn.init.orthogonal(self.item_class_em.weight)
        nn.init.orthogonal(self.item_family_em.weight)
        nn.init.orthogonal(self.store_em.weight)
        nn.init.orthogonal(self.month_em.weight)
        nn.init.orthogonal(self.day_em.weight)
        nn.init.orthogonal(self.year_em.weight)
        if not mlp:
            for m in self.step_one_network:
                if isinstance(m, nn.Linear):
                    nn.init.xavier_normal(m.weight_g)
                    nn.init.orthogonal(m.weight_v)
                    nn.init.constant(m.bias, 0)

    def preprocess_x(self, x, x_d, x_i):
        embeddings = torch.cat([
            self.item_class_em(x_i[:, :, 4].long()),
            self.item_family_em(x_i[:, :, 5].long()),
            self.store_type_em(x_i[:, :, 6].long()),
            self.store_cluster_em(x_i[:, :, 7].long()),
            self.store_em(x_i[:, :, 8].long()),
            self.store_city_em(x_i[:, :, 9].long()),
            self.day_em((x_i[:, :, 10]).long()),
            self.month_em((x_i[:, :, 11]).long()),
            self.year_em((x_i[:, :, 12]).long()),
            self.weekday_em((x_i[:, :, 13]).long()),
        ], dim=2)
        if self.edrop:
            embeddings = self.edropout(embeddings, self.edrop)
        if self.training and self.min_length != TRAIN_PERIODS:
            start = random.randint(0, TRAIN_PERIODS - self.min_length + 1)
        else:
            start = 0
        x_encode = torch.cat([
            x[:, start:TRAIN_PERIODS, :2],
            # Store Cluster Sales
            x[:, start:TRAIN_PERIODS, 2:4],
            # Item class sales
            # x[:, start:TRAIN_PERIODS, 4:6],
            x_d.unsqueeze(1).expand(
                x_d.size()[0], TRAIN_PERIODS - start, x_d.size()[1]),
            # Local promo
            x_i[:, start:TRAIN_PERIODS, :2].float(),
            # Global promo
            x_i[:, start:TRAIN_PERIODS, 2:4].float(),
            # x_i[:, start:TRAIN_PERIODS, -2:-1].float()
            embeddings[:, start:TRAIN_PERIODS, :],
        ], dim=2)
        step_idx = torch.cuda.FloatTensor(
            x.size()[0], self.steps, self.steps).zero_()
        step_idx.scatter_(
            2, torch.arange(0, self.steps).cuda().long(
            ).unsqueeze(0).expand(x.size()[0], self.steps).unsqueeze(2), 1)
        x_decode = torch.cat([
            x[:, TRAIN_PERIODS:, :].index_select(
                2, Variable(torch.cuda.LongTensor(
                    [0, 1, 3] if self.propagate else [1, 3]))),
            x_d.unsqueeze(1).expand(x_d.size()[0], 15, x_d.size()[1]),
            embeddings[:, TRAIN_PERIODS:, :],
            # Local Promo
            x_i[:, TRAIN_PERIODS:, :2].float(),
            # Global Promo
            x_i[:, TRAIN_PERIODS:, 2:4].float(),
            Variable(step_idx)
        ], dim=2)
        return x_encode, x_decode


class LSTNet(BaseModel):
    def __init__(
            self, cnn_hidden_size, rnn_hidden_size, skip_hidden_size,
            hdrop, edrop, odrop, skip=7, cnn_kernel=3, *,
            y_scale_by, steps=15, rnn_type="GRU", min_length=TRAIN_PERIODS,
            ar_window_size=0):
        super(LSTNet, self).__init__(
            bidirectional=False, edrop=edrop, odrop=odrop, propagate=False,
            min_length=min_length, y_scale_by=y_scale_by, steps=steps
        )
        self.hdrop = hdrop
        self.hdropout = LockedDropout(batch_first=False)
        self.ar_window_size = ar_window_size
        self.skip = skip
        self.skip_seq_len = (TRAIN_PERIODS - cnn_kernel) // self.skip
        self._create_layers(mlp=True)
        self.cnn = nn.utils.weight_norm(
            nn.Conv1d(self.input_dim, cnn_hidden_size, cnn_kernel))
        if rnn_type == "GRU":
            self.rnn = nn.GRU(
                cnn_hidden_size, rnn_hidden_size, batch_first=False)
            self.rnn = weight_norm_rnn(self.rnn)
        else:
            self.rnn = nn.LSTM(
                cnn_hidden_size, rnn_hidden_size, batch_first=False)
        if skip:
            if rnn_type == "GRU":
                self.skiprnn = nn.GRU(
                    cnn_hidden_size, skip_hidden_size, batch_first=False)
                self.skiprnn = weight_norm_rnn(self.skiprnn)
            elif rnn_type == "SRU":
                self.skiprnn = nn.SRU(cnn_hidden_size, skip_hidden_size)
            else:
                self.skiprnn = nn.LSTM(
                    cnn_hidden_size, skip_hidden_size, batch_first=False)
            output_dim = rnn_hidden_size + \
                skip_hidden_size * skip + steps * 7
        else:
            output_dim = rnn_hidden_size + steps * 7
        self.output_network = nn.Sequential(
            torch.nn.utils.weight_norm(
                nn.Linear(output_dim, rnn_hidden_size)),
            nn.ReLU(),
            nn.Dropout(self.odrop),
            torch.nn.utils.weight_norm(
                nn.Linear(rnn_hidden_size, steps + 1))
        )
        if self.ar_window_size:
            self.ar_highway = torch.nn.utils.weight_norm(
                nn.Linear(ar_window_size, steps + 1))
        self.init_mlp_weights()

    def init_mlp_weights(self):
        """Class specific initialization"""
        super(LSTNet, self).init_weights(mlp=True)
        nn.init.orthogonal(self.cnn.weight)
        nn.init.constant(self.cnn.bias, 0)
        for rnn in (self.skiprnn, self.rnn):
            if "weight_ig_l0_g" in rnn._parameters:
                print("Initializing WN LSTN/GRU...")
                nn.init.xavier_normal(rnn.weight_ih_l0_g)
                nn.init.orthogonal(rnn.weight_ih_l0_v)
                nn.init.xavier_normal(rnn.weight_hh_l0_g)
                nn.init.orthogonal(rnn.weight_hh_l0_v)
            else:
                nn.init.orthogonal(rnn.weight_ih_l0)
                nn.init.orthogonal(rnn.weight_hh_l0)
            nn.init.constant(rnn.bias_ih_l0, 0)
            nn.init.constant(rnn.bias_hh_l0, 0)
        for m in self.output_network:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal(m.weight_g)
                nn.init.orthogonal(m.weight_v)
                nn.init.constant(m.bias, 0)
        if self.ar_window_size:
            nn.init.xavier_normal(self.ar_highway.weight_g)
            nn.init.orthogonal(self.ar_highway.weight_v)
            nn.init.constant(self.ar_highway.bias, 0)

    def forward(self, x, x_d, x_i, **ignore):
        x_enc, _ = self.preprocess_x(x, x_d, x_i)
        batch_size = x_enc.size()[0]
        # CNN
        cnn_output = F.relu(
            self.cnn(x_enc.permute(0, 2, 1)).contiguous().permute(2, 0, 1))
        if self.hdrop:
            cnn_output = self.hdropout(cnn_output, self.hdrop)

        # RNN
        _, rnn_hidden = rnn_output = self.rnn(cnn_output)
        if isinstance(rnn_hidden, tuple):
            # LSTM
            rnn_hidden = rnn_hidden[0]
        rnn_output = rnn_hidden.squeeze(0)
        if self.hdrop:
            rnn_output = F.dropout(rnn_output, self.hdrop)

        # Skip RNN
        if self.skip:
            skip_input = cnn_output[
                (-self.skip_seq_len * self.skip):, :, :].contiguous()
            skip_input = skip_input.view(
                self.skip_seq_len, self.skip * batch_size, skip_input.size(2))
            _, skip_hidden = self.skiprnn(skip_input)
            if isinstance(skip_hidden, tuple):
                # LSTM
                skip_hidden = skip_hidden[0]
            skip_output = skip_hidden.squeeze(0)
            skip_output = skip_output.view(batch_size, -1)
            if self.hdrop:
                skip_output = F.dropout(skip_output, self.hdrop)
            rnn_output = torch.cat([rnn_output, skip_output], dim=1)

        reg_input = torch.cat([
            rnn_output,
            # Previous year sales
            x[:, TRAIN_PERIODS:, 1].float(),
            # Previous year cluster sales
            x[:, TRAIN_PERIODS:, 3].float(),
            # Previous year item class sales
            x[:, TRAIN_PERIODS:, 5].float(),
            # Promo this year
            x_i[:, TRAIN_PERIODS:, 0].float(),
            # Promo last year
            x_i[:, TRAIN_PERIODS:, 1].float(),
            # Global promo this year
            x_i[:, TRAIN_PERIODS:, 2].float(),
            # Global promo last year
            x_i[:, TRAIN_PERIODS:, 3].float(),
        ], dim=1)
        reg_output = self.output_network(reg_input)

        # AR Highway
        if self.ar_window_size > 0:
            hw_input = F.dropout(
                x_enc[:, -self.ar_window_size:, 0], self.odrop)
            hw_output = self.ar_highway(hw_input)
            reg_output = reg_output + hw_output

        return reg_output


class TransformerModel(BaseModel):
    def __init__(
            self, n_max_seq, *,  y_scale_by, steps, min_length, n_layers=6, n_head=8,
            d_word_vec=512, d_model=512, d_inner_hid=1024, d_k=64, d_v=64,
            edrop=0.25, odrop=0.25, hdrop=0.1, propagate=False):
        super(TransformerModel, self).__init__(
            bidirectional=False, edrop=edrop, odrop=odrop, propagate=propagate,
            min_length=min_length, y_scale_by=y_scale_by, steps=steps
        )
        self.hidden_size = d_model
        self._create_layers(mlp=True)
        self.encoder = TransformerEncoder(
            n_max_seq, n_layers=n_layers, n_head=n_head,
            d_word_vec=d_word_vec, d_model=d_model, d_k=d_k, d_v=d_v,
            d_inner_hid=d_inner_hid, dropout=hdrop)
        self.decoder = TransformerDecoder(
            n_max_seq, n_layers=n_layers, n_head=n_head,
            d_word_vec=d_word_vec, d_model=d_model, d_k=d_k, d_v=d_v,
            d_inner_hid=d_inner_hid, dropout=hdrop)
        self.encoder_mapping = nn.Linear(self.input_dim, d_word_vec)
        self.decoder_mapping = nn.Linear(self.decode_dim, d_word_vec)
        self.step_one_network = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            LayerNormalization(self.hidden_size),
            nn.Linear(self.hidden_size, 1)
        )
        self.output_network = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            LayerNormalization(self.hidden_size),
            nn.Linear(self.hidden_size, 1)
        )
        self.init_linear_weights()

        assert d_model == d_word_vec
        'To facilitate the residual connections, \
         the dimensions of all module output shall be the same.'

    def init_linear_weights(self):
        """Class specific initialization"""
        super(TransformerModel, self).init_weights(mlp=True)
        nn.init.orthogonal(self.encoder_mapping.weight)
        nn.init.constant(self.encoder_mapping.bias, 0)
        nn.init.orthogonal(self.decoder_mapping.weight)
        nn.init.constant(self.decoder_mapping.bias, 0)
        for submodel in (self.encoder, self.decoder):
            for m in submodel.parameters():
                if isinstance(m, (nn.Linear, nn.Conv1d)):
                    nn.init.orthogonal(m.weight)
                    nn.init.constant(m.bias, 0)
        for submodel in (self.output_network, self.step_one_network):
            for m in submodel:
                if isinstance(m, nn.Linear):
                    nn.init.orthogonal(m.weight)
                    nn.init.constant(m.bias, 0)

    def get_trainable_parameters(self):
        ''' Avoid updating the position encoding '''
        enc_freezed_param_ids = set(
            map(id, self.encoder.position_enc.parameters()))
        dec_freezed_param_ids = set(
            map(id, self.decoder.position_enc.parameters()))
        freezed_param_ids = enc_freezed_param_ids | dec_freezed_param_ids
        return (p for p in self.parameters() if id(p) not in freezed_param_ids)

    def forward(self, x, x_d, x_i, *, tf_ratio=0, **ignore):
        x_encode, x_decode = self.preprocess_x(x, x_d, x_i)
        src_pos = Variable(torch.arange(0, x_encode.size()[1]).cuda().long())
        tgt_pos = Variable(torch.arange(0, x_decode.size()[1]).cuda().long())
        x_encode = self.encoder_mapping(x_encode)
        enc_output, *_ = self.encoder(x_encode, src_pos)
        step_one = self.step_one_network(
            F.dropout(enc_output[:, -1, :], self.odrop))
        if self.propagate:
            output = Variable(
                torch.cuda.FloatTensor(x.size()[0], self.steps + 1).zero_())
            previous = step_one
            output[:, 0] = previous[:, 0]
            for j in range(self.steps):
                if random.random() >= tf_ratio:
                    x_decode[:, j, 0] = previous[:, 0].detach()
                dec_output, *_ = self.decoder(
                    self.decoder_mapping(
                        x_decode[:, :(j + 1), :]), tgt_pos[:(j + 1)],
                    x_encode, enc_output)
                previous = self.output_network(
                    F.dropout(dec_output[:, -1, :], self.odrop))
                output[:, j + 1] = previous[:, 0]
            return output
        x_decode = self.decoder_mapping(x_decode)
        dec_output, *_ = self.decoder(x_decode, tgt_pos, x_encode, enc_output)
        reg_output = self.output_network(
            F.dropout(dec_output, self.odrop)).squeeze(2)
        return torch.cat([step_one, reg_output], dim=1)
