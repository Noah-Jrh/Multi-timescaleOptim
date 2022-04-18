# -*- coding: utf-8 -*-
# @Time    : 2019/8/6 下午10:34
# @Author  : Noah
# @Email   : 15520816169@163.com
# @File    : function.py
# @Software: PyCharm
# @Update   :
# @Version  : 1.0
"""Instruction:
    Version_1.0:
"""

import os
import torch
import numpy as np
import logging
import scipy.io as io
import random


class Logger:
    def __init__(self, args, log_path, flag):
        self.log_path = log_path
        self.logger = logging.getLogger('')
        if args.train:
            filename = os.path.join(self.log_path, 'train_' + flag + '.log')
            # file handler
            handler = logging.FileHandler(filename=filename, mode="w")
            handler.setLevel(logging.INFO)
            handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s: %(message)s'))

            # console handler
            console = logging.StreamHandler()
            console.setLevel(logging.INFO)
            console.setFormatter(logging.Formatter('%(message)s'))

            self.logger.setLevel(logging.INFO)
            self.logger.addHandler(handler)
            self.logger.addHandler(console)
            self.logger.info("Logger created at {}".format(filename))
        else:
            filename = os.path.join(self.log_path, 'test_' + flag + '.log')
            # file handler
            handler = logging.FileHandler(filename=filename, mode="w")
            handler.setLevel(logging.INFO)
            handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s: %(message)s'))

            # console handler
            console = logging.StreamHandler()
            console.setLevel(logging.INFO)
            console.setFormatter(logging.Formatter('%(message)s'))

            self.logger.setLevel(logging.INFO)
            self.logger.addHandler(handler)
            self.logger.addHandler(console)
            self.logger.info("Logger created at {}".format(filename))

    def debug(self, strout):
        return self.logger.debug(strout)

    def info(self, strout):
        return self.logger.info(strout)


# 图像脉冲编码
def img2spike(vec, time_window):
    """Instruction:
        vec: data tensor
        T: Time Window Size
    """
    # Latency Encoding 时延编码
    firing_time = ((torch.ones_like(vec).float() - vec) * time_window).view(-1).int()  # [784]
    # spike_trans = torch.zeros(vec.size(1), time_window+1)
    # for t in range(time_window+1):
    #     spike_trans[:, t] = firing_time == t
    return firing_time


def bernoulli(datum: torch.Tensor, time, dt: float = 1.0, **kwargs) -> torch.Tensor:
    max_prob = kwargs.get('max_prob', 1.0)

    assert 0 <= max_prob <= 1, 'Maximum firing probability must be in range [0, 1]'
    assert (datum >= 0).all(), 'Inputs must be non-negative'

    shape, size = datum.shape, datum.numel()
    datum = datum.view(-1)

    if time is not None:
        time = int(time / dt)

    # Normalize inputs and rescale (spike probability proportional to normalized intensity).
    if datum.max() > 1.0:
        datum /= datum.max()

    # Make spike data from Bernoulli sampling.
    if time is None:
        spikes = torch.bernoulli(max_prob * datum)
        spikes = spikes.view(*shape)
    else:
        spikes = torch.bernoulli(max_prob * datum.repeat([time, 1]))
        spikes = spikes.view(time, *shape)

    return spikes.byte()


def fly_hash(vec, args, path):
    vec = vec.view(vec.size(0), -1)
    batch = vec.size(0)
    in_size = vec.size(1)
    out_size = in_size * args.expansion
    k = int(args.wta * out_size)
    savepath = os.path.join(path, 'RandomWts.mat')
    if not os.path.isfile(savepath):
        weight = torch.zeros(in_size, out_size).float()
        for j in range(out_size):
            idx = random.sample(list(range(in_size)), int(args.sparse * in_size))
            for i in idx:
                weight[i, j] = 1.0
        io.savemat(savepath, {'Wts': weight.cpu().numpy()})
    else:
        data = io.loadmat(savepath)
        tmp = data['Wts']
        weight = torch.from_numpy(tmp).float()
    weight = weight.cuda()
    outpt = vec.mm(weight)
    value, _ = outpt.max(dim=1)
    for p in range(batch):
        outpt[p, :] = outpt[p, :] / value[p]
    # print(outpt[0, :])
    # fired_s, idx = torch.sort(outpt, descending=True)
    # for p in range(batch):
    #     tmp = outpt[p, :]
    #     tmp[tmp < fired_s[p, k]] = 0
    #     tmp[tmp >= fired_s[p, k]] = 1
    return outpt


def accuracy(output, target, args):
    # pred_1 = np.argmax(output.detach().cpu().numpy(), axis=1)
    output = output.view(output.size(0), args.n_way, args.output_size // args.n_way)
    vote_sum = torch.sum(output, dim=2)
    pred = torch.argmax(vote_sum, dim=1)
    correct = torch.eq(pred, target.long()).sum().item()
    # tmp = pred_1 - target.cpu().numpy()
    # print('after:', correct)
    # print('before:', np.sum(tmp == 0))
    # if correct != np.sum(tmp == 0):
    #     print(output)
    #     print(vote_sum)
    #     print(pred_1)
    #     print(pred)
    #     raise NameError('wrong')
    return correct


def confusion_matrix(output, target, args):
    cm = torch.zeros(args.n_way, args.n_way)
    output = output.view(output.size(0), args.n_way, args.output_size // args.n_way)
    vote_sum = torch.sum(output, dim=2)
    pred = torch.argmax(vote_sum, dim=1)
    for p, t in zip(pred, target):
        cm[p, t] += 1
    return cm.numpy().T


def preprocess_grad_loss(x):
    p = 10
    indicator = (x.abs() >= np.exp(-p)).to(torch.float32)

    # preproc1
    x_proc1 = indicator * torch.log(x.abs() + 1e-8) / p + (1 - indicator) * -1
    # preproc2
    x_proc2 = indicator * torch.sign(x) + (1 - indicator) * np.exp(p) * x
    return torch.stack((x_proc1, x_proc2), 1)


# Dacay learning_rate
def lr_scheduler(optimizer, epoch, lr_decay_epoch=1, lr_decay=0.1):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    if epoch % lr_decay_epoch == 0 and epoch > 1:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * lr_decay
    return optimizer
