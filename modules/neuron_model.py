# coding=utf-8
'''
   @Author       : Noah
   @Version      : v1.0.0
   @Date         : 2020-01-13 11:30:44
   @LastEditors  : Please set LastEditors
   @LastEditTime : 2020-11-11 15:14:10
   @FilePath     : /workspace/Meta_Synaptic/Modules/neuron_model.py
   @CopyRight (c): 2019 NCRC, SCU. All rights reserved.
   @Description  : neuron_model.py
'''
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
pi = torch.tensor(math.pi)


class Linear(torch.autograd.Function):
    '''
    Here we use the piecewise-linear surrogate gradient as was done
    in Bellec et al. (2018).
    '''
    gamma = 0.3  # Controls the dampening of the piecewise-linear surrogate gradient

    @staticmethod
    def forward(self, inpt):
        self.save_for_backward(inpt)
        return inpt.gt(0).float()

    @staticmethod
    def backward(self, grad_output):
        inpt, = self.saved_tensors
        grad_input = grad_output.clone()
        sur_grad = Linear.gamma * F.threshold(1.0 - torch.abs(inpt), 0, 0)
        return grad_input * sur_grad.float()


class Rectangle(torch.autograd.Function):
    '''
    Here we use the Rectangle surrogate gradient as was done
    in Yu et al. (2018).
    '''
    # beta = 1.0  # Controls the dampening of the piecewise-linear surrogate gradient

    @staticmethod
    def forward(self, inpt):
        self.save_for_backward(inpt)
        return inpt.gt(0).float()

    @staticmethod
    def backward(self, grad_output):
        inpt, = self.saved_tensors
        grad_input = grad_output.clone()
        sur_grad = (torch.abs(inpt) < 0.5).float()
        return grad_input * sur_grad


class PDF(torch.autograd.Function):

    alpha = 0.1
    beta = 0.1

    @staticmethod
    def forward(self, inpt):
        self.save_for_backward(inpt)
        return inpt.gt(0).float()

    @staticmethod
    def backward(self, grad_output):
        inpt, = self.saved_tensors
        grad_input = grad_output.clone()
        sur_grad = PDF.alpha * torch.exp(-PDF.beta * torch.abs(inpt))
        return sur_grad * grad_input


class ActFun(torch.autograd.Function):
    ''' Adaptive threshold: neurons have same threshold in same channel
    Input size: [batch, 32, 28, 28], [batch, 32, 14, 14], [batch, 32, 7, 7], [batch, 5]
    Threshold size: [32], [32], [32], [5] reshape operating inside
    Forward: (u-u_th)<=0? H(u-u_th)=0:1
    Backward:   grad_u = mathcal{Spike} * H'
                grad_u_th = -mathcal{Spike} * H'
    '''
    @staticmethod
    def forward(self, inpt, u_th):
        if inpt.size(1) == 32:
            tmp = u_th.unsqueeze(1).unsqueeze(1).unsqueeze(0).expand_as(inpt)
        else:
            tmp = u_th.unsqueeze(0).expand_as(inpt)
        u_delta = inpt - tmp
        self.save_for_backward(inpt, u_th)
        return u_delta.gt(0).float()

    @staticmethod
    def backward(self, grad_output):
        inpt, u_th = self.saved_tensors
        grad_input = grad_output.clone()
        if inpt.size(1) == 32:
            tmp = u_th.unsqueeze(1).unsqueeze(1).unsqueeze(0).expand_as(inpt)
        else:
            tmp = u_th.unsqueeze(0).expand_as(inpt)
        h_deriva = abs(inpt - tmp) < 0.5
        grad_u = grad_input * h_deriva.float()
        grad_u_th = -grad_input * h_deriva.float()
        if inpt.size(1) == 32:
            grad_u_th = grad_u_th.sum(0).sum(2).sum(1)
        else:
            grad_u_th = grad_u_th.sum(0)
        return grad_u, grad_u_th


# class STDB(torch.autograd.Function):

#     alpha = ''
#     beta = ''

#     @staticmethod
#     def forward(ctx, input, last_spike):

#         ctx.save_for_backward(last_spike)
#         out = torch.zeros_like(input).cuda()
#         out[input > 0] = 1.0
#         return out

#     @staticmethod
#     def backward(ctx, grad_output):

#         last_spike, = ctx.saved_tensors
#         grad_input = grad_output.clone()
#         grad = STDB.alpha * torch.exp(-1 * last_spike)**STDB.beta
#         return grad * grad_input, None


class A_LIF(nn.Module):
    # Shared adaptative threshold and Leaky in each layer
    def __init__(self, args, in_feature):
        super(A_LIF, self).__init__()
        self.args = args
        self.in_feature = in_feature
        self.v_th = nn.Parameter(torch.tensor(args.v_th))
        self.v_decay = nn.Parameter(torch.tensor(args.v_decay))
        if args.sur_grad == 'linear':
            self.act_fun = Linear().apply
        elif args.sur_grad == 'rectangle':
            self.act_fun = Rectangle().apply
        elif args.sur_grad == 'pdf':
            self.act_fun = PDF().apply

    def reset_parameters(self, inpt):
        self.membrane = inpt

    def forward_cts(self, inpt, step):
        # for backwrad only at current time steps
        if step == 0:   # reset
            self.reset_parameters(inpt)
        else:
            self.membrane = self.v_decay * self.prev_membrane * (1. - self.prev_spike) + inpt

        if self.v_th > 0:
            mem_thr = self.membrane / (self.v_th + 1e-8) - 1.0
        elif self.v_th <= 0:
            mem_thr = 1.0 - self.membrane / (self.v_th + 1e-8)
        # mem_thr = self.membrane - self.v_th

        self.spike = self.act_fun(mem_thr)
        self.prev_membrane = self.membrane.detach()
        self.prev_spike = self.spike.detach()
        return self.spike.clone()

    def forward_pts(self, inpt, step):
        # for backwrad through previous time steps
        if step == 0:   # reset
            self.reset_parameters(inpt)
        else:
            self.membrane = self.v_decay * self.membrane * (1. - self.spike) + inpt

        if self.v_th > 0:
            mem_thr = self.membrane / (self.v_th + 1e-8) - 1.0
        elif self.v_th < 0:
            mem_thr = 1.0 - self.membrane / (self.v_th + 1e-8)
        mem_thr = self.membrane - self.v_th

        self.spike = self.act_fun(mem_thr)
        return self.spike.clone()

    def forward(self, inpt, step):
        return self.forward_cts(inpt, step)
        # return self.forward_pts(inpt, step)

    def extra_repr(self):
        return 'args={}, in_feature={}'.format(self.args is not None, self.in_feature)


class A_LIF_v1(nn.Module):
    # Shared adaptative threshold and Leaky in same channals, like bias
    def __init__(self, args, in_feature):
        super(A_LIF_v1, self).__init__()
        self.args = args
        self.in_feature = in_feature
        self.v_th = nn.Parameter(torch.FloatTensor(torch.full([in_feature[0]], args.v_th)))
        self.v_decay = nn.Parameter(torch.FloatTensor(torch.full([in_feature[0]], args.v_decay)))
        if args.sur_grad == 'linear':
            self.act_fun = Linear().apply
        elif args.sur_grad == 'rectangle':
            self.act_fun = Rectangle().apply
        elif args.sur_grad == 'pdf':
            self.act_fun = PDF().apply

    def reset_parameters(self, inpt):
        self.membrane = inpt

    def forward(self, inpt, step):
        # # linear norm
        # self.v_decay.data = 0.1 + (self.v_decay.data - self.v_decay.min().data) / (self.v_decay.max().data - self.v_decay.min().data) * 0.9
        # self.v_th.data = 0.1 + (self.v_th.data - self.v_th.min().data) / (self.v_th.max().data - self.v_th.min().data) * 0.9
        if inpt.size(1) == 32:
            decay = self.v_decay.unsqueeze(1).unsqueeze(1).unsqueeze(0).expand_as(inpt)
            th = self.v_th.unsqueeze(1).unsqueeze(1).unsqueeze(0).expand_as(inpt)
        else:
            decay = self.v_decay.expand_as(inpt)
            th = self.v_th.expand_as(inpt)

        if step == 0:   # reset
            self.reset_parameters(inpt)
        else:
            # self.membrane = self.v_decay * self.membrane * (1. - self.spike) + inpt
            # self.membrane = self.v_decay * self.prev_membrane * (1. - self.prev_spike) + inpt
            self.membrane = torch.sigmoid(decay) * self.membrane * (1. - self.spike) + inpt

        mem_thr = self.membrane / torch.sigmoid(th) - 1.0
        # mem_thr = self.membrane - self.v_th
        self.spike = self.act_fun(mem_thr)
        # self.prev_membrane = self.membrane.detach()
        # self.prev_spike = self.spike.detach()
        # self.spike = self.act_fun(self.membrane, self.v_th)
        return self.spike.clone()

    def extra_repr(self):
        return 'args={}, in_feature={}'.format(self.args is not None, self.in_feature)


class LIF_v1(nn.Module):
    '''
        Forward Return: spikes in each time step
        Backward: backprop previous time step, membrane like eligibility traces
    '''

    def __init__(self, args, in_feature):
        super(LIF_v1, self).__init__()
        self.args = args
        self.in_feature = in_feature
        self.v_th = args.v_th
        self.v_decay = args.v_decay
        if args.sur_grad == 'linear':
            self.act_fun = Linear().apply
        elif args.sur_grad == 'rectangle':
            self.act_fun = Rectangle().apply
        elif args.sur_grad == 'pdf':
            self.act_fun = PDF().apply
        # # Record List in a time window
        # self.spike_trains = []          # Spike output in each step
        # self.current_trains = []        # Currents in each step
        # self.membrane_trains = []       # Membrane potential in each step

    def reset_parameters(self, inpt):
        self.membrane = inpt

    def forward(self, inpt, step):
        if step == 0:   # reset
            self.reset_parameters(inpt)
        else:
            self.membrane = self.v_decay * self.prev_membrane * (1. - self.prev_spike) + inpt
        mem_thr = self.membrane / self.v_th - 1.0
        # mem_thr = self.membrane - self.v_th
        self.spike = self.act_fun(mem_thr)
        # self.spike = self.act_fun(self.membrane, self.v_th)
        self.prev_membrane = self.membrane.detach()
        self.prev_spike = self.spike.detach()
        return self.spike.clone()

    def extra_repr(self):
        return 'args={}, in_feature={}'.format(self.args is not None, self.in_feature)


class LIF_v2(nn.Module):
    '''
        Forward Return: spikes in each time step
        Backward: only in current time step, membrane like eligibility traces
    '''

    def __init__(self, args, in_feature):
        super(LIF_v2, self).__init__()
        self.args = args
        self.in_feature = in_feature
        self.v_th = args.v_th
        self.v_decay = args.v_decay
        if args.sur_grad == 'linear':
            self.act_fun = Linear().apply
        elif args.sur_grad == 'rectangle':
            self.act_fun = Rectangle().apply
        elif args.sur_grad == 'pdf':
            self.act_fun = PDF().apply
        # # Record List in a time window
        # self.spike_trains = []          # Spike output in each step
        # self.current_trains = []        # Currents in each step
        # self.membrane_trains = []       # Membrane potential in each step

    def reset_parameters(self, inpt):
        self.membrane = inpt

    def forward(self, inpt, step):
        if step == 0:   # reset
            self.reset_parameters(inpt)
        else:
            self.membrane = self.v_decay * self.prev_membrane * (1. - self.prev_spike) + inpt
        mem_thr = self.membrane / self.v_th - 1.0
        # mem_thr = self.membrane - self.v_th
        self.spike = self.act_fun(mem_thr)
        self.prev_membrane = self.membrane.detach()
        self.prev_spike = self.spike.detach()
        # self.spike = self.act_fun(self.membrane, self.v_th)
        return self.spike.clone()

    def extra_repr(self):
        return 'args={}, in_feature={}'.format(self.args is not None, self.in_feature)
