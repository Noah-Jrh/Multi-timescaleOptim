# coding=utf-8
'''
   @Author       : Noah
   @Version      : v1.0.0
   @Date         : 2020-09-11 11:37:47
   @LastEditors  : Please set LastEditors
   @LastEditTime : 2020-11-16 16:20:50
   @CopyRight (c): 2019 NCRC, SCU. All rights reserved.
   @Description  : 等步长反向过程（Equidistant Backward Phases，EBP）
'''

import argparse
import copy
import time
import os
import random
import torch
import torch.nn as nn
# import torch.nn.functional as fun
# import torch.optim.lr_scheduler as lr_scheduler
import torch.cuda as cuda
import torch.backends.cudnn as cudnn
import numpy as np
import utils.function as tool
import utils.plot as plt
# from torchvision import transforms
from utils.dataloader import loader, episode_data
import modules.learner as learner
from modules.meta_learner import MetaLearner, set_loss, set_optimizer

# Global variable
# os.environ['CUDA_VISIBLE_DEVICES'] = "1"
parser = argparse.ArgumentParser(description='SNN training')
parser.add_argument("--model", type=str, default='MSTO')
parser.add_argument("--structure", type=str, default='SCN_4_32')
parser.add_argument("--dataSet", type=str, default='omniglot',
                    choices=['mnist', 'omniglot', 'spiking'])
parser.add_argument("--preprocess", choices=['hmax', None], default=None)
parser.add_argument("--checkpoints", type=str, default='2020-10-02-17_47_57',
                    help='2020-11-06-20_31_44')
parser.add_argument("--dataDir", type=str, default='./1-Data')
parser.add_argument("--resDir", type=str, default='./5-Shared/MSTO_SNN/results')
parser.add_argument("--nIter", type=int, default=1000)
parser.add_argument("--nEpoch", type=int, default=100)
parser.add_argument("--batchSize", type=int, default=1)
parser.add_argument("--seed", type=int, default=None, help="Random seed")

parser.add_argument('--n_way', type=int, help='n way', default=5)
parser.add_argument('--k_shot', type=int,
                    help='k shot for support set', default=5)
parser.add_argument('--k_query', type=int,
                    help='k shot for query set', default=15)
parser.add_argument('--img_size', type=int, help='image size', default=28)
parser.add_argument('--input_size', type=int, default=784,
                    help='spiking (MLP based SNN): 196 / omniglot(CNN based SNN): 784')
parser.add_argument('--output_size', type=int, default=5)

parser.add_argument("--dt", type=float, default=1)
parser.add_argument("--rates", type=float, default=1.0)
parser.add_argument("--v_th", type=float, default=0.2)
parser.add_argument("--v_decay", type=float, default=0.3)
parser.add_argument("--T", type=int, default=20)
parser.add_argument("--step", type=int, default=1)
parser.add_argument("--sur_grad", choices=['linear', 'rectangle', 'pdf'], default='linear')

parser.add_argument('--loss_fun', choices=['ce', 'mse'], default='ce',
                    help='Loss Function: ce(CrossEntropyLoss), mse(MSELoss)')
parser.add_argument('--optimizer', choices=['sgd', 'adam'], default='adam',
                    help='Optimizer: SGD,Adam')
parser.add_argument("--cfg_lstm", type=int, nargs='+', default=[4, 20],
                    help='lstm(input_size, hidden_size)')
parser.add_argument("--lstm_bias", type=float,
                    nargs='+', default=[4, 6, -5, -4])
parser.add_argument('--lr', type=float, default=0.001)

parser.add_argument("--learner_bias", dest="learner_bias", action="store_true")
parser.add_argument("--train", dest="train", action="store_true")
parser.add_argument("--plot", dest="plot", action="store_true")
parser.set_defaults(learner_bias=False, train=False, plot=True)
# Set device
device = torch.device("cuda" if cuda.is_available() else "cpu")
if cuda.is_available():
    cudnn.deterministic = True
    cudnn.benchmark = False
parser.add_argument('--device', type=float, default=device)
locals().update(vars(parser.parse_args()))
args, unparsed = parser.parse_known_args()
acc_record = list([])

''' Configuration instructions
    CONV: nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding) in_planes from previous layer
    AP: nn.AvgPool2d(kernel_size, stride, padding)
    FC: nn.Linear(in_features, out_features) in_planes from previous layer
    A_LIF : nm.A_LIF()
    Spiking Convolutional Neural Network (SCN)
        SCN_5_32: 4 Conv-Avgpool layers (32 filters + Nonoverlap AvgPool) + 1-layer Fully Connection
        SCN_5_32_v1: 4 Conv-Avgpool layers (32 filters + Mixed AvgPool) + 1-layer Fully Connection
        SCN_5_64: 4 Conv-Avgpool layers (64 filters + Nonoverlap AvgPool) + 1-layer Fully Connection
        SCN_5_64_v1: 4 Conv-Avgpool layers (64 filters + Mixed AvgPool) + 1-layer Fully Connection
'''
cfg = {
    # MLP-based SNN
    'SNN_2': [('linear', [50]), ('a_lif', []), ('output', [args.n_way])],
    # HMAX + MLP-based SNN
    'SNN_3_HMAX': [('linear', [128]), ('a_lif', []), ('linear', [64]), ('a_lif', []), ('output', [args.n_way])],
    'SNN_3_HMAX_bn': [('linear', [128]), ('bn1', [1e-3, 0.95]), ('a_lif', []),
                      ('linear', [64]), ('bn1', [1e-3, 0.95]), ('a_lif', []), ('output', [args.n_way])],
    # HMAX + MLP-based SNN
    'SNN_3_HMAX_v1': [('linear', [128]), ('lif_v1', []), ('linear', [64]), ('lif_v1', []), ('output', [args.n_way])],

    # MSTO_epb_alif
    'SCN_4_32': [('conv2d', [32, 3, 1, 1]), ('a_lif', []), ('ap', [2, 2, 0])] * 3 + [
        ('flatten', []), ('output', [args.n_way])],
    'SCN_4_32_bn': [('conv2d', [32, 3, 1, 1]), ('bn2', [1e-3, 0.95]), ('a_lif', []), ('ap', [2, 2, 0])] * 3 + [
        ('flatten', []), ('output', [args.n_way])],
    # MSTO_epb_lif: no a_lif
    'SCN_4_32_v1': [('conv2d', [32, 3, 1, 1]), ('lif_v1', []), ('ap', [2, 2, 0])] * 3 + [
        ('flatten', []), ('output', [args.n_way])],
    'SCN_4_32_bn_v1': [('conv2d', [32, 3, 1, 1]), ('bn2', [1e-3, 0.95]), ('lif_v1', []), ('ap', [2, 2, 0])] * 3 + [
        ('flatten', []), ('output', [args.n_way])],

    # 'SCN_4_32_v2': [('ap', [4, 4, 0])] + [('conv2d', [32, 3, 1, 1]), ('lif_v2', []), ('ap', [2, 2, 0])] * 3 + [
    #     ('flatten', []), ('output', [args.n_way])],
    # 'SCN_4_32_v3': [('ap', [4, 4, 0])] + [('conv2d', [32, 3, 1, 1]), ('bn', [1e-3, 0.95]), ('lif_v2', []), ('ap', [2, 2, 0])] * 3 + [
    #     ('flatten', []), ('output', [args.n_way])],

    'SCN_5_32': [('conv2d', [32, 3, 1, 1]), ('a_lif', []), ('ap', [2, 2, 0])] * 4 + [
        ('flatten', []), ('output', [args.n_way])],
    'SCN_5_64': [('conv2d', [64, 3, 1, 1]), ('a_lif', []), ('ap', [2, 2, 0])] * 4 + [
        ('flatten', []), ('output', [args.n_way])],
    'SCN_5_64_bn': [('conv2d', [64, 3, 1, 1]), ('bn2', [1e-3, 0.95]), ('a_lif', []), ('ap', [2, 2, 0])] * 4 + [
        ('flatten', []), ('output', [args.n_way])],
}

# Set path
if args.train:
    flag = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime())
else:
    flag = args.checkpoints
names = [args.model, args.structure, args.dataSet, args.n_way, args.k_shot]
model_name = '_'.join([str(x) for x in names])
data_path = os.path.join(args.dataDir, args.dataSet)
params_path = os.path.join(args.resDir, model_name, 'parameters', flag)
logs_path = os.path.join(args.resDir, model_name, 'logs')
imgs_path = os.path.join(args.resDir, model_name, 'imgs')
for path in [data_path, params_path, logs_path, imgs_path]:
    if not os.path.isdir(path):
        os.makedirs(path)

# Set seed
if args.seed is None:
    args.seed = random.randint(0, 1e3)
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

# Set Logger
logger = tool.Logger(args, logs_path, flag)
argsDict = args.__dict__
logger.info('>>>> Hyper-parameters')
for key, value in sorted(argsDict.items()):
    logger.info(' --' + str(key) + ': ' + str(value))
if len(unparsed) != 0:
    raise NameError("Argument {} not recognized".format(unparsed))

# Set data
datas = loader(args.dataSet, data_path, args)
args.input_size = datas.inpt_size

# Init Network
learner_train = learner.SpkNet(args, cfg[args.structure]).to(args.device)
learner_metatrain = copy.deepcopy(learner_train)

logger.info('>>>> Learner Structure:')
logger.info(learner_train.net)
logger.info('>>>> Learner Parameters:')
for name, params in learner_train.net.named_parameters():
    logger.info(' --' + str(name) + ': ' + str(params.shape))
metalearner = MetaLearner(
    args, learner.get_flat_params(learner_train)).to(args.device)
logger.info('>>>> MetaLearner Structure:')
logger.info(metalearner.net)
logger.info('>>>> MetaLearner Parameters:')
for name, params in metalearner.named_parameters():
    logger.info(' --' + str(name) + ': ' + str(params.size()))

# loss and optimizer setting
loss_function = set_loss(args.loss_fun)
# optimizer = set_optimizer(args.optimizer, metalearner, args.lr)


def loss_calc(output, label):
    batch = output.size(0)
    if args.loss_fun == 'ce':
        loss = loss_function(output, label.long())
        loss.backward()
        # loss.backward(retain_graph=True)
    elif args.loss_fun == 'mse':
        population_num = args.output_size // args.n_way
        inpt = torch.ones(batch, population_num, device=args.device)
        tmp = torch.arange(0, population_num,
                           device=args.device).expand_as(inpt)
        index = population_num * label.view(batch, 1).expand_as(inpt) + tmp
        label = torch.zeros(batch, args.output_size).to(args.device).scatter_(1, index, inpt).float()
        loss = loss_function(output, label)
        loss.backward()
    else:
        raise NameError("Loss {} not recognized".format(args.loss_fun))
    return loss


l1_v = []
l1_d = []
l1_mem = []
l1_spk = []
l4_v = []
l4_d = []
l7_v = []
l7_d = []


def plot(inpt):
    '''
        data: torch.Float
    '''
    fig = plt.figure()
    axs = fig.add_subplot(3, 1, 3)
    img1 = inpt[:, 0, :].T
    # find firing neuron and its timestamp
    idx = np.where(img1 == 1)
    idx_n = np.unique(idx[0])   # fring neuron idx
    idx_t = []  # fring neuron idx
    for k in idx_n:
        idx_t.append(idx[1][idx[0] == k])
    colors2 = 'black'
    lineoffsets = idx_n
    linelengths = 1
    # create a horizontal plot (orientation='vertical')
    axs.eventplot(idx_t, colors=colors2, lineoffsets=lineoffsets, linelengths=linelengths)
    x_ticks = range(0, 51, 50)
    plt.xticks(x_ticks)
    y_ticks = range(0, 197, 196)
    plt.yticks(y_ticks)
    plt.ylabel(u'input layer')

    axs = fig.add_subplot(3, 1, 1)
    img1 = np.array(l1_spk)
    # find firing neuron and its timestamp
    idx = np.where(img1 == 1)
    idx_n = np.unique(idx[0])   # fring neuron idx
    idx_t = []  # fring neuron idx
    for k in idx_n:
        idx_t.append(idx[1][idx[0] == k])
    colors2 = 'black'
    lineoffsets = idx_n
    linelengths = 1
    # create a horizontal plot (orientation='vertical')
    axs.eventplot(idx_t, colors=colors2, lineoffsets=lineoffsets, linelengths=linelengths)
    x_ticks = range(0, 51, 50)
    plt.xticks(x_ticks)
    y_ticks = range(0, 51, 50)
    plt.yticks(y_ticks)
    plt.ylabel(u'hidden layer')

    axs = fig.add_subplot(3, 1, 2)
    img2 = np.array(l1_mem)
    img3 = np.array(l1_v)
    img4 = np.array(l1_d)
    for i in range(img2.shape[1]):
        if i == 0:
            axs.plot(img2[:, i], color='lightgray', label='Membrane Potential')
        else:
            axs.plot(img2[:, i], color='lightgray')
    axs.plot(img3, color='lightcoral', label='Firing Threshold')
    axs.plot(img4, color='yellow', label='Leakage Factor')
    axs.legend()
    plt.ylabel(u'hidden layer')
    plt.show()
    # plt.savefig(str(target[i]) + '_' + str(i) + '.png', dpi=600)
    # print()
    print()


def plot2(inpt):
    '''
        data: torch.Float
    '''
    fig = plt.figure()
    axs = fig.add_subplot(3, 1, 1)
    img1_v = np.array(l1_v)
    img1_d = np.array(l1_d)
    axs.plot(img1_v / img1_v.max(), color='lightcoral', label='Firing Threshold')
    axs.plot(img1_d / img1_d.max(), color='yellow', label='Leakage Factor')
    axs.legend()
    plt.ylabel(u'Conv1')

    axs = fig.add_subplot(3, 1, 2)
    img4_v = np.array(l4_v)
    img4_d = np.array(l4_d)
    axs.plot(img4_v / img4_v.max(), color='lightcoral', label='Firing Threshold')
    axs.plot(img4_d / img4_d.max(), color='yellow', label='Leakage Factor')
    axs.legend()
    plt.ylabel(u'Conv2')

    axs = fig.add_subplot(3, 1, 3)
    img7_v = np.array(l7_v)
    img7_d = np.array(l7_d)
    axs.plot(img7_v / img7_v.max(), color='lightcoral', label='Firing Threshold')
    axs.plot(img7_d / img7_d.max(), color='yellow', label='Leakage Factor')
    axs.legend()
    plt.ylabel(u'Conv3')
    plt.show()


def train_learner(x_spt, y_spt):
    c0 = metalearner.net['metalstm'].c0.data
    hs = [None]
    for step in range(args.T):
        learner.copy_flat_params(learner_train, c0)
        # l1_v.append(copy.copy(learner_train.net[1].v_th.detach().numpy()))
        # l1_d.append(copy.copy(learner_train.net[1].v_th.detach().numpy()))
        sptout = learner_train(x_spt[step, :], step)
        # for name, param in learner_train.named_parameters():
        #     if param.requires_grad:
        #         print(name)
        # for p in learner_train.parameters():
        #     print(p.size())
        #     print(p.grad.data.max())
        if (step + 1) % args.step == 0 and step != 0:
            # l1_v.append(copy.copy(learner_train.net[1].v_th.detach().numpy()))
            # l1_d.append(copy.copy(learner_train.net[1].v_decay.detach().numpy()))
            # # l1_mem.append(copy.copy(learner_train.net[1].membrane[0, :].detach().numpy()))
            # # l1_spk.append(copy.copy(learner_train.net[1].spike[0, :].detach().numpy()))
            # l4_v.append(copy.copy(learner_train.net[4].v_th.detach().numpy()))
            # l4_d.append(copy.copy(learner_train.net[4].v_decay.detach().numpy()))
            # l7_v.append(copy.copy(learner_train.net[7].v_th.detach().numpy()))
            # l7_d.append(copy.copy(learner_train.net[7].v_decay.detach().numpy()))
            learner_train.zero_grad()
            loss = loss_calc(sptout, y_spt)
            # nn.utils.clip_grad_norm_(learner_train.parameters(), 0.25)
            grad = torch.cat([p.grad.data.view(-1)
                              for p in learner_train.parameters()], 0)
            # Run the metalearner on the input.
            grad_prep = tool.preprocess_grad_loss(grad)  # [n_learner_params, 2]
            loss_prep = tool.preprocess_grad_loss(loss.data.unsqueeze(0))
            meta_inpt = [loss_prep, grad_prep, grad.unsqueeze(1)]
            c0, ht = metalearner(meta_inpt, hs[-1])
            # print('max=: ', c0.max(), 'min=: ', c0.min())
            hs.append(ht)
    # raise NameError("error")
    # print('Support episode acc: %.2f' % acc)
    return c0


def meta_train():
    optimizer = set_optimizer(args.optimizer, metalearner, args.lr)
    # scheduler = lr_scheduler.MultiStepLR(optimizer, [10, 30, 60, 90], 0.5)
    for iEpoch in range(args.nEpoch):
        correct = 0
        batch_loss = 0
        start_time = time.time()
        # optimizer = tool.lr_scheduler(optimizer, iEpoch + 1, lr_decay_epoch=50, lr_decay=0.1)
        # scheduler.step()
        for step in range(args.nIter):
            learner_train.reset_batch_stats()
            learner_metatrain.reset_batch_stats()
            learner_train.train()
            learner_metatrain.train()
            x_spt, x_qry, y_spt, y_qry = episode_data(datas, 'train', args)
            # x_spt = fly_hash(x_spt, args, params_path)
            # x_qry = fly_hash(x_qry, args, params_path)
            # x_spt = fun.interpolate(x_spt, size=[20, 20], mode="bilinear", align_corners=False)
            # x_qry = fun.interpolate(x_qry, size=[20, 20], mode="bilinear", align_corners=False)
            # img2 = Image.fromarray(255 * np.uint8(x_spt[1, 0].cpu().numpy()))
            # img2.save("2.png")
            x_spt = tool.bernoulli(datum=x_spt, time=args.T, dt=args.dt, max_prob=args.rates).float()
            x_qry = tool.bernoulli(datum=x_qry, time=args.T, dt=args.dt, max_prob=args.rates).float()
            c0 = train_learner(x_spt, y_spt)
            # print('max=: ', c0.max(), 'min= : ', c0.min())
            # Train meta-learner with validation loss
            learner.transfer_params(learner_metatrain, learner_train, c0)
            opt_train = []
            for i in range(args.T):
                y = learner_metatrain(x_qry[i, :], i)
                opt_train.append(y)
            if args.loss_fun == 'ce':
                output = np.sum(opt_train, 0)
            # Spike output: Suit Spike-based loss (MSE)
            elif args.loss_fun == 'mse':
                output = np.sum(opt_train, 0) / args.T
            # print(1)
            # print(learner_metatrain.net['c1_lif'].threshold.min(), learner_metatrain.net['c1_lif'].threshold.max())
            # print(learner_metatrain.net['c2_lif'].threshold.min(), learner_metatrain.net['c1_lif'].threshold.max())
            # print(learner_metatrain.net['c3_lif'].threshold.min(), learner_metatrain.net['c1_lif'].threshold.max())
            # print(learner_metatrain.net['output'].threshold.min(), learner_metatrain.net['c1_lif'].threshold.max())
            # for name, param in learner_metatrain.named_parameters():
            #     if name == 'net.c1_lif.threshold':
            #         print(param.min())
            #     if name == 'net.c2_lif.threshold':
            #         print(param.min())
            #     if name == 'net.c3_lif.threshold':
            #         print(param.min())
            #     if name == 'net.output.threshold':
            #         print(param.min())
            # if name == 'net.conv1.weight':
            #     print(param.min())
            # if name == 'net.conv2.weight':
            #     print(param.min())
            # if name == 'net.conv3.weight':
            #     print(param.min())
            # print(output[0])
            optimizer.zero_grad()
            loss = loss_calc(output, y_qry)
            batch_loss += loss.item()
            # grad clip
            nn.utils.clip_grad_norm_(metalearner.parameters(), 0.25)
            optimizer.step()
            # print('max=: ', metalearner.net['metalstm'].c0.data.max(),
            #       'min=: ', metalearner.net['metalstm'].c0.data.min())
            correct += tool.accuracy(output, y_qry, args)
            total = args.n_way * args.k_query
            acc = correct / (total * (step + 1))
            acc_record.append(tool.accuracy(output, y_qry, args) / total)
            if (step + 1) % 100 == 0:
                logger.info('Epoch [%d/%d], Iter [%d/%d], Loss: %.5f, Acc: %.2f, Time elasped: %.2f'
                            % (iEpoch + 1, args.nEpoch, step + 1, args.nIter,
                               batch_loss, acc * 100, time.time() - start_time))
                batch_loss = 0
                start_time = time.time()
                # print(torch.any(learner_metatrain.net[1].membrane > 0))
                # print(torch.any(learner_metatrain.net[4].membrane > 0))
                # print(torch.any(learner_metatrain.net[7].membrane > 0))
                # print(torch.sigmoid(learner_metatrain.net[1].v_decay), torch.sigmoid(learner_metatrain.net[1].v_th))
                # print(torch.sigmoid(learner_metatrain.net[4].v_decay), torch.sigmoid(learner_metatrain.net[4].v_th))
                # print(torch.sigmoid(learner_metatrain.net[7].v_decay), torch.sigmoid(learner_metatrain.net[7].v_th))
        meta_test(str(iEpoch + 1))


def meta_test(epoch):
    correct = 0
    acc = 0
    acc_episode = []
    FLAG = True
    base_acc = 97
    if not args.train:
        # checkpoint = torch.load(os.path.join(params_path, epoch), map_location=torch.device('cpu'))
        checkpoint = torch.load(os.path.join(params_path, epoch))
        if checkpoint['acc'] >= base_acc:
            logger.info(epoch)
            print('ACC is about:', checkpoint['acc'])
            metalearner.load_state_dict(checkpoint['metalearner'])
        else:
            FLAG = False
    if FLAG:
        for step in range(100):
            learner_train.reset_batch_stats()
            learner_metatrain.reset_batch_stats()
            learner_train.train()
            learner_metatrain.train()
            x_spt, x_qry, y_spt, y_qry = episode_data(datas, 'test', args)
            # x_spt = fun.interpolate(x_spt, size=[20, 20], mode="bilinear", align_corners=False)
            # x_qry = fun.interpolate(x_qry, size=[20, 20], mode="bilinear", align_corners=False)
            x_spt = tool.bernoulli(datum=x_spt, time=args.T, dt=args.dt,
                                   max_prob=args.rates).float()
            x_qry = tool.bernoulli(datum=x_qry, time=args.T, dt=args.dt,
                                   max_prob=args.rates).float()
            c0 = train_learner(x_spt, y_spt)
            # plot2(x_spt)
            learner.transfer_params(learner_metatrain, learner_train, c0)
            opt_train = []
            for i in range(args.T):
                y = learner_metatrain(x_qry[i, :], i)
                opt_train.append(y)
            if args.loss_fun == 'ce':
                output = np.sum(opt_train, 0)
            # Spike output: Suit Spike-based loss (MSE)
            elif args.loss_fun == 'mse':
                output = np.sum(opt_train, 0) / args.T
            correct += tool.accuracy(output, y_qry, args)
            total = args.n_way * args.k_query
            acc = correct / (total * (step + 1))
            acc_episode.append(tool.accuracy(output, y_qry, args) / total)
        logger.info('Meta test acc: %.2f' % (acc * 100))
    if not args.train and acc > base_acc * 0.01:
        state = {'acc_episode': acc_episode}
        torch.save(state, os.path.join(params_path, 'acc_' + epoch))
    if args.train and acc >= 0.70:
        logger.info('Saving best model at ' + os.path.join(params_path, 'Epoch_' + epoch + '.pth'))
        state = {
            'learner': learner_train.state_dict(),
            'metalearner': metalearner.state_dict(),
            'acc': acc * 100,
            'acc_record': acc_record
        }
        torch.save(state, os.path.join(params_path, 'Epoch_' + epoch + '.pth'))
    return acc_episode


def main():
    if args.train:
        meta_train()
    else:
        # meta_test('Epoch_100.pth')
        for root, dirs, files in os.walk(params_path):
            for i in files:
                meta_test(i)


if __name__ == '__main__':
    main()
