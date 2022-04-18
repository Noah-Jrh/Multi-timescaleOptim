# coding=utf-8
'''
   @Author       : Noah
   @Version      : v1.0.0
   @Date         : 2020-08-21 22:13:25
   @LastEditors  : Please set LastEditors
   @LastEditTime : 2020-11-26 10:42:06
   @CopyRight (c): 2019 NCRC, SCU. All rights reserved.
   @Description  : Please add descriptioon
'''

import torch
import torch.nn as nn
# import torch.nn.functional as fun
import numpy as np
# from modules import neuron_model as nm
import modules.neuron_model as nm
from functools import reduce
import math


class SpkNet(nn.Module):
    '''
        Spiking Neural Network
    '''

    def __init__(self, args, cfg):
        super(SpkNet, self).__init__()
        self.args = args
        self.net = self._make_layers(cfg)
        # self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.uniform_(0, 1)
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.uniform_(0, 1)

    def _make_layers(self, cfg):
        layers = []
        if self.args.structure[0:3] == 'SNN':
            in_features = self.args.input_size
        else:
            in_channels = 1  # decided by the dataSet
            feature_map = [self.args.img_size, self.args.img_size]
            in_features = in_channels * reduce(lambda x, y: x * y, feature_map)   # 列表中所有元素的乘积
        for i, (name, param) in enumerate(cfg):
            if name == 'conv2d':
                layers += [nn.Conv2d(in_channels, param[0], kernel_size=param[1], stride=param[2], padding=param[3], bias=self.args.learner_bias)]
                feature_map = [(x - param[1] + 2 * param[3]) // param[2] + 1 for x in feature_map]  # W = (W - K + 2P) / S + 1
                in_channels = param[0]
            elif name == 'bn2':
                layers += [nn.BatchNorm2d(in_channels, eps=param[0], momentum=param[1])]
            elif name == 'a_lif':
                if isinstance(layers[-1], (nn.BatchNorm2d)) or isinstance(layers[-1], (nn.Conv2d)):
                    layers += [nm.A_LIF(self.args, [in_channels] + feature_map)]
                elif isinstance(layers[-1], (nn.Linear)):
                    layers += [nm.A_LIF(self.args, in_features)]
            elif name == 'lif_v1':
                if isinstance(layers[-1], (nn.BatchNorm2d)) or isinstance(layers[-1], (nn.Conv2d)):
                    layers += [nm.LIF_v1(self.args, [in_channels] + feature_map)]
                elif isinstance(layers[-1], (nn.Linear)):
                    layers += [nm.LIF_v1(self.args, in_features)]
            elif name == 'lif_v2':
                if isinstance(layers[-1], (nn.BatchNorm2d)) or isinstance(layers[-1], (nn.Conv2d)):
                    layers += [nm.LIF_v2(self.args, [in_channels] + feature_map)]
                elif isinstance(layers[-1], (nn.Linear)):
                    layers += [nm.LIF_v2(self.args, in_features)]
            elif name == 'ap':
                layers += [nn.AvgPool2d(kernel_size=param[0], stride=param[1], padding=param[2])]
                feature_map = [(x - param[0] + 2 * param[2]) // param[1] + 1 for x in feature_map]  # W = (W - K + 2P) / S + 1
            elif name == 'flatten':
                in_features = in_channels * reduce(lambda x, y: x * y, feature_map)   # 列表中所有元素的乘积
            elif name == 'linear':
                layers += [nn.Linear(in_features, param[0], bias=self.args.learner_bias)]
                in_features = param[0]
            elif name == 'output':
                layers += [nn.Linear(in_features, param[0], bias=self.args.learner_bias)]
                if self.args.loss_fun == 'ce':
                    break
                elif self.args.loss_fun == 'mse':
                    layers += [nm.A_LIF(self.args, param)]
                else:
                    raise NameError("Loss Functions {} not recognized".format(name))
            else:
                raise NameError("Components {} not recognized".format(name))
        # nn.ModuleDict({'features': nn.Sequential(*layers)})
        net = nn.Sequential(*layers) return net

    def forward_v0(self, inpt):
        # forward for BP in all time windows
        batch = inpt[0, :].size(0)
        opt_train = []
        for step in range(self.args.T):  # simulation time steps
            x = inpt[step, :]
            for i in range(len(self.net)):
                if isinstance(self.net[i], (nn.Conv2d)) or isinstance(self.net[i], (nn.AvgPool2d)) or isinstance(self.net[i], nn.BatchNorm2d):
                    x = self.net[i](x)
                elif isinstance(self.net[i], (nn.Linear)):
                    x = x.view(batch, -1)
                    x = self.net[i](x)
                elif isinstance(self.net[i], (nm.A_LIF)):
                    x = self.net[i](x, step)
            opt_train.append(x)
        # Voltage output: Suit Voltage-based loss (CE)
        if self.args.loss_fun == 'ce':
            outputs = np.sum(opt_train, 0)
        # Spike output: Suit Spike-based loss (MSE)
        if self.args.loss_fun == 'mse':
            outputs = np.sum(opt_train, 0) / self.args.T
        return outputs

    def forward_v1(self, x, step):
        # forward for EBP or SBP
        batch = x.size(0)
        for i in range(len(self.net)):
            if isinstance(self.net[i], (nn.Conv2d)) or isinstance(self.net[i], (nn.AvgPool2d)) or isinstance(self.net[i], nn.BatchNorm2d):
                x = self.net[i](x)
            elif isinstance(self.net[i], (nn.Linear)):
                x = x.view(batch, -1)
                x = self.net[i](x)
            elif isinstance(self.net[i], (nm.A_LIF)) or isinstance(self.net[i], (nm.LIF_v1)) or isinstance(self.net[i], (nm.LIF_v2)):
                x = self.net[i](x, step)
        return x

    def forward(self, inpt, step):
        # return self.forward_v0(inpt)
        return self.forward_v1(inpt, step)

    def reset_batch_stats(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.reset_running_stats()


class ConvNet(nn.Module):
    '''
        Spiking Convolutional Neural Network
    '''

    def __init__(self, args, cfg):
        super(ConvNet, self).__init__()
        self.args = args
        self.net = self._make_layers(cfg)
        # self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.uniform_(0, 1)
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.uniform_(0, 1)

    def _make_layers(self, cfg):
        layers = []
        in_channels = 1  # decided by the dataSet
        feature_map = [self.args.img_size, self.args.img_size]
        in_features = in_channels * reduce(lambda x, y: x * y, feature_map)   # 列表中所有元素的乘积

        for i, (name, param) in enumerate(cfg):
            if name == 'conv2d':
                layers += [nn.Conv2d(in_channels, param[0], kernel_size=param[1], stride=param[2], padding=param[3], bias=self.args.learner_bias)]
                # layers += [nn.BatchNorm2d(param[0], 1e-3, 0.95)]
                feature_map = [(x - param[1] + 2 * param[3]) // param[2] + 1 for x in feature_map]  # W = (W - K + 2P) / S + 1
                in_channels = param[0]
            elif name == 'relu':
                layers += [nn.ReLU()]
            elif name == 'ap':
                layers += [nn.AvgPool2d(kernel_size=param[0], stride=param[1], padding=param[2])]
                feature_map = [(x - param[0] + 2 * param[2]) // param[1] + 1 for x in feature_map]  # W = (W - K + 2P) / S + 1
            elif name == 'flatten':
                in_features = in_channels * reduce(lambda x, y: x * y, feature_map)   # 列表中所有元素的乘积
            elif name == 'linear':
                layers += [nn.Linear(in_features, param[0], bias=self.args.learner_bias)]
                in_features = param[0]
            elif name == 'output':
                layers += [nn.Linear(in_features, param[0], bias=self.args.learner_bias)]
            else:
                raise NameError("Components {} not recognized".format(name))
        # nn.ModuleDict({'features': nn.Sequential(*layers)})
        net = nn.Sequential(*layers)
        return net

    def forward(self, x):
        batch = x.size(0)
        for i in range(len(self.net)):
            if isinstance(self.net[i], (nn.Conv2d)) or isinstance(self.net[i], (nn.AvgPool2d)) or isinstance(
                    self.net[i], (nn.ReLU)) or isinstance(self.net[i], (nn.MaxPool2d)) or isinstance(self.net[i], nn.BatchNorm2d) or isinstance(self.net[i], nn.BatchNorm1d):
                x = self.net[i](x)
            elif isinstance(self.net[i], (nn.Linear)):
                x = x.view(batch, -1)
                x = self.net[i](x)
        return x

    def reset_batch_stats(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                m.reset_running_stats()


def get_flat_params(net):
    # Get the net.parameters
    return torch.cat([p.view(-1) for p in net.parameters()], 0)


def copy_flat_params(net, c0):
    # Init the net.parameters with tensors from c0
    idx = 0
    for p in net.parameters():
        plen = p.view(-1).size(0)
        p.data.copy_(c0[idx: idx + plen].view_as(p))
        idx += plen


def transfer_params(net, net2, c0):
    # Use load_state_dict only to copy the running mean/var in batchnorm,
    net.load_state_dict(net2.state_dict())
    # Replace nn.Parameters with tensors from c0
    idx = 0
    for m in net.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear) or isinstance(m, nn.BatchNorm2d):
            wlen = m._parameters['weight'].view(-1).size(0)
            m._parameters['weight'] = c0[idx: idx + wlen].view_as(m._parameters['weight']).clone()
            idx += wlen
            # print('m:', m)
            # print('wlen:', wlen)
            # print('idx:', idx)
            if m._parameters['bias'] is not None:
                blen = m._parameters['bias'].view(-1).size(0)
                m._parameters['bias'] = c0[idx: idx + blen].view_as(m._parameters['bias']).clone()
                idx += blen
                # print('blen:', blen)
                # print('idx:', idx)
        if isinstance(m, nm.A_LIF):
            tlen = m._parameters['v_th'].view(-1).size(0)
            m._parameters['v_th'] = c0[idx: idx + tlen].view_as(m._parameters['v_th']).clone()
            idx += tlen
            dlen = m._parameters['v_decay'].view(-1).size(0)
            m._parameters['v_decay'] = c0[idx: idx + dlen].view_as(m._parameters['v_decay']).clone()
            idx += dlen
            # print('m:', m)
            # print('wlen:', wlen)
            # print('idx:', idx)


# class CNN(nn.Module):
#     """
#         artificial nerual networks:
#     """

#     def __init__(self, args):
#         super(CNN, self).__init__()
#         self.args = args
#         if args.structure == 'fc2':
#             self.cfg_fc = [args.input_size, 100, args.n_way]
#             self.net = nn.Sequential(OrderedDict([
#                 ('fc1', nn.Linear(self.cfg_fc[0], self.cfg_fc[1])),
#                 ('fc2', nn.Linear(self.cfg_fc[1], self.cfg_fc[2]))
#             ]))
#         elif args.structure == 'conv3-fc1':
#             self.args = args
#             self.cfg_cnn = [[1, 32, 3, 1, 1],
#                             [32, 32, 3, 1, 1],
#                             [32, 32, 3, 1, 1]]
#             self.cfg_kernel = [28, 14, 7, 3]
#             in_feature = self.cfg_kernel[-1] * \
#                 self.cfg_kernel[-1] * self.cfg_cnn[-1][1]
#             self.cfg_fc = [in_feature, args.n_way]
#             in_planes1, out_planes1, kernel_size1, stride1, padding1 = self.cfg_cnn[0]
#             in_planes2, out_planes2, kernel_size2, stride2, padding2 = self.cfg_cnn[1]
#             in_planes3, out_planes3, kernel_size3, stride3, padding3 = self.cfg_cnn[2]
#             self.net = nn.ModuleDict({
#                 'cnn': nn.Sequential(OrderedDict([
#                     ('conv1', nn.Conv2d(in_planes1, out_planes1, kernel_size=kernel_size1,
#                                         stride=stride1, padding=padding1)),
#                     # ('norm1', nn.BatchNorm2d(32, bn_eps, bn_momentum)),
#                     ('relu1', nn.ReLU(inplace=False)),
#                     ('pool1', nn.AvgPool2d(2)),

#                     ('conv2', nn.Conv2d(in_planes2, out_planes2, kernel_size=kernel_size2,
#                                         stride=stride2, padding=padding2)),
#                     # ('norm2', nn.BatchNorm2d(32, bn_eps, bn_momentum)),
#                     ('relu2', nn.ReLU(inplace=False)),
#                     ('pool2', nn.AvgPool2d(2)),

#                     ('conv3', nn.Conv2d(in_planes3, out_planes3, kernel_size=kernel_size3,
#                                         stride=stride3, padding=padding3)),
#                     # ('norm3', nn.BatchNorm2d(32, bn_eps, bn_momentum)),
#                     ('relu3', nn.ReLU(inplace=False)),
#                     ('pool3', nn.AvgPool2d(2))])),
#                 'fc1': nn.Linear(self.cfg_fc[0], self.cfg_fc[-1])}
#             )
#         else:
#             raise NameError(
#                 "Structure {} not recognized".format(self.args.structure))

#     def forward(self, x):
#         if self.args.structure == 'fc2':
#             x = x.view(x.size(0), -1)
#             outputs = self.net(x)

#         elif self.args.structure == 'conv3-fc1':
#             x = self.net['cnn'](x)
#             feature = x.view(x.size(0), -1)
#             outputs = self.net['fc1'](feature)
#         else:
#             raise NameError(
#                 "Structure {} not recognized".format(self.args.structure))
#         return outputs

#     def get_flat_params(self):
#         return torch.cat([p.view(-1) for p in self.net.parameters()], 0)

#     def copy_flat_params(self, ci):
#         idx = 0
#         for p in self.net.parameters():
#             plen = p.view(-1).size(0)
#             p.data.copy_(ci[idx: idx+plen].view_as(p))
#             idx += plen

#     def transfer_params(self, learner_w_grad, c0):
#         # Use load_state_dict only to copy the running mean/var in batchnorm, the values of the parameters
#         #  are going to be replaced by cI
#         self.load_state_dict(learner_w_grad.state_dict())
#         #  replace nn.Parameters with tensors from cI (NOT nn.Parameters anymore).
#         idx = 0
#         for m in self.net.modules():
#             if isinstance(m, nn.Conv2d) or isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.Linear):
#                 wlen = m._parameters['weight'].view(-1).size(0)
#                 m._parameters['weight'] = c0[idx: idx +
#                                              wlen].view_as(m._parameters['weight']).clone()
#                 idx += wlen
#                 if m._parameters['bias'] is not None:
#                     blen = m._parameters['bias'].view(-1).size(0)
#                     m._parameters['bias'] = c0[idx: idx +
#                                                blen].view_as(m._parameters['bias']).clone()
#                     idx += blen
