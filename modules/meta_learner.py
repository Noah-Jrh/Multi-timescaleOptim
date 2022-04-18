# -*- coding: utf-8 -*-
# @Time    : 2019/7/31 上午10:07
# @Author  : Noah
# @Email   : 15520816169@163.com
# @File    : meta_learner.py
# @Software: PyCharm
"""Instruction:
    Version_1.0:
"""

import torch
import torch.nn as nn
import torch.optim as optim
# import numpy as np


# like switch-case
def set_loss(var):
    return {
        'ce': nn.CrossEntropyLoss(),
        'mse': nn.MSELoss(),
    }.get(var, 'error')


# like switch-case
def set_optimizer(var, model, lr):
    return {
        'sgd': optim.SGD(model.parameters(), lr=lr),
        'adam': optim.Adam(model.parameters(), lr=lr)
    }.get(var, 'error')


class MetaLSTMCell(nn.Module):
    """
        C_t = f_t * C_{t-1} + i_t * \tilde{C_t}
    """
    def __init__(self, input_size, hidden_size, params, lstm_bias):
        super(MetaLSTMCell, self).__init__()
        """Args:
            input_size (int): cell input size, default = 20
            hidden_size (int): should be 1
            n_learner_params (int): number of learner's parameters
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_learner_params = params.size(0)
        self.WF = nn.Parameter(torch.Tensor(input_size + 2, hidden_size))
        self.WI = nn.Parameter(torch.Tensor(input_size + 2, hidden_size))
        self.c0 = nn.Parameter(torch.Tensor(self.n_learner_params, 1))
        self.bI = nn.Parameter(torch.Tensor(1, hidden_size))
        self.bF = nn.Parameter(torch.Tensor(1, hidden_size))
        self.reset_parameters(lstm_bias)
        self.c0.data.copy_(params.unsqueeze(1))

    def reset_parameters(self, lstm_bias):
        for weight in self.parameters():
            nn.init.uniform_(weight, -0.01, 0.01)
        nn.init.uniform_(self.bF, lstm_bias[0], lstm_bias[1])
        nn.init.uniform_(self.bI, lstm_bias[2], lstm_bias[3])

    def forward(self, inputs, hx=None):
        x_all, grad = inputs
        batch, _ = x_all.size()
        if hx is None:
            f_prev = torch.zeros((batch, self.hidden_size)).to(self.WF.device)
            i_prev = torch.zeros((batch, self.hidden_size)).to(self.WI.device)
            c_prev = self.c0
            hx = [f_prev, i_prev, c_prev]
        f_prev, i_prev, c_prev = hx
        # f_t = sigmoid(W_f * [grad_t, loss_t, theta_{t-1}, f_{t-1}] + b_f)
        f_next = torch.mm(torch.cat((x_all, c_prev, f_prev), 1), self.WF) + self.bF.expand_as(f_prev)
        # i_t = sigmoid(W_i * [grad_t, loss_t, theta_{t-1}, i_{t-1}] + b_i)
        i_next = torch.mm(torch.cat((x_all, c_prev, i_prev), 1), self.WI) + self.bI.expand_as(i_prev)
        # next cell/params
        c_next = torch.sigmoid(f_next).mul(c_prev) - torch.sigmoid(i_next).mul(grad)
        # print('max=', torch.sigmoid(i_next).max(), 'min=', torch.sigmoid(i_next).min())

        return c_next, [f_next, i_next, c_next]

    def extra_repr(self):
        s = '{input_size}, {hidden_size}, {n_learner_params}'
        return s.format(**self.__dict__)


class MetaLearner(nn.Module):

    def __init__(self, args, params):
        super(MetaLearner, self).__init__()
        """Args:
            input_size (int): for the first LSTM layer, default = 4
            hidden_size (int): for the first LSTM layer, default = 20
            n_learner_params (int): number of learner's parameters
        """
        self.input_size = args.cfg_lstm[0]
        self.hidden_size = args.cfg_lstm[1]
        self.batch_size = args.batchSize
        self.lstm_bias = args.lstm_bias
        self.n_learner_params = params.size(0)
        self.net = nn.ModuleDict({
            'lstm': nn.LSTMCell(self.input_size, self.hidden_size),
            'metalstm': MetaLSTMCell(self.hidden_size, 1, params, self.lstm_bias),
        })

    def forward(self, inputs, hs=None):
        loss, grad_prep, grad = inputs
        loss = loss.expand_as(grad_prep)
        inputs = torch.cat((loss, grad_prep), 1)  # [n_learner_params, 4]
        if hs is None:
            hs = [None, None]
        lstmhx, lstmcx = self.net['lstm'](inputs, hs[0])
        flat_learner_unsqzd, metalstm_hs = self.net['metalstm']([lstmhx, grad], hs[1])
        return flat_learner_unsqzd.squeeze(), [(lstmhx, lstmcx), metalstm_hs]


class MetaLSTMCellv6(nn.Module):
    def __init__(self, input_size, hidden_size, flat_params, lstm_bias):
        super(MetaLSTMCellv6, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_learner_params = flat_params.size(0)
        self.WF = nn.Parameter(torch.Tensor(input_size + 2, hidden_size))
        self.WI = nn.Parameter(torch.Tensor(input_size + 2, hidden_size))
        self.c0 = nn.Parameter(torch.Tensor(self.n_learner_params, 1))
        self.bI = nn.Parameter(torch.Tensor(1, hidden_size))
        self.bF = nn.Parameter(torch.Tensor(1, hidden_size))
        self.reset_parameters(lstm_bias)
        self.c0.data.copy_(flat_params.unsqueeze(1))

    def reset_parameters(self, lstm_bias):
        for weight in self.parameters():
            nn.init.uniform_(weight, -0.01, 0.01)
        nn.init.uniform_(self.bF, lstm_bias[0], lstm_bias[1])
        nn.init.uniform_(self.bI, lstm_bias[2], lstm_bias[3])

    def forward(self, inputs, hx=None):
        x_all, grad = inputs
        batch, _ = x_all.size()
        if hx is None:
            f_prev = torch.zeros((batch, self.hidden_size)).to(self.WF.device)
            i_prev = torch.zeros((batch, self.hidden_size)).to(self.WI.device)
            c_prev = self.c0
            hx = [f_prev, i_prev, c_prev]

        f_prev, i_prev, c_prev = hx

        # f_t = sigmoid(W_f * [grad_t, loss_t, theta_{t-1}, f_{t-1}] + b_f)
        f_next = torch.mm(torch.cat((x_all, c_prev, f_prev), 1), self.WF) + self.bF.expand_as(f_prev)
        # i_t = sigmoid(W_i * [grad_t, loss_t, theta_{t-1}, i_{t-1}] + b_i)
        i_next = torch.mm(torch.cat((x_all, c_prev, i_prev), 1), self.WI) + self.bI.expand_as(i_prev)
        # next cell/params
        c_next = torch.sigmoid(f_next).mul(c_prev) - torch.sigmoid(i_next).mul(grad)
        # c_next = torch.sigmoid(f_next).mul(c_prev) - torch.sigmoid(i_next).mul(grad)
        # print('max=', torch.sigmoid(i_next).max(), 'min=', torch.sigmoid(i_next).min())
        return c_next, [f_next, i_next, c_next]

    def extra_repr(self):
        s = '{input_size}, {hidden_size}, {n_learner_params}'
        return s.format(**self.__dict__)


class MetaLearnerv6(nn.Module):

    def __init__(self, args, params):
        super(MetaLearnerv6, self).__init__()
        self.input_size = args.cfg_lstm[0]
        self.hidden_size = args.cfg_lstm[1]
        self.lstm_bias = args.lstm_bias
        self.net = nn.ModuleDict({
            'lstm': nn.LSTMCell(self.input_size, self.hidden_size),
            'metalstm': MetaLSTMCellv6(self.hidden_size, 1, params, self.lstm_bias),
        })

    def forward(self, inputs, hs=None):
        loss, grad_prep, grad = inputs
        loss = loss.expand_as(grad_prep)
        inputs = torch.cat((loss, grad_prep), 1)  # [n_learner_params, 4]
        if hs is None:
            hs = [None, None]

        lstmhx, lstmcx = self.net['lstm'](inputs, hs[0])
        flat_learner_unsqzd, metalstm_hs = self.net['metalstm']([lstmhx, grad], hs[1])

        return flat_learner_unsqzd.squeeze(), [(lstmhx, lstmcx), metalstm_hs]
