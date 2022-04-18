# coding=utf-8
'''
   @Author       : Noah
   @Version      : v1.0.0
   @Date         : 2020-10-27 14:14:35
   @LastEditors  : Please set LastEditors
   @LastEditTime : 2020-11-28 14:54:54
   @CopyRight (c): 2019 NCRC, SCU. All rights reserved.
   @Description  : Plot Function
'''
import torch
import matplotlib.pyplot as plt
from pylab import *
import numpy as np
import matplotlib
# import utils.function as fc
matplotlib.rcParams['font.size'] = 5.0
# np.random.seed(19680801)

# plot for spiking dataset


def plot_spiking(data, target):
    '''
        data: torch.Float
    '''
    data = data.numpy()
    fig = plt.figure()
    idx1 = 0
    idx2 = 5
    for i in range(data.shape[1]):
        if target[i].numpy() == 0:
            idx1 = idx1 + 1
            axs = fig.add_subplot(2, 5, idx1)
        if target[i].numpy() == 1:
            idx2 = idx2 + 1
            axs = fig.add_subplot(2, 5, idx2)
        if target[i].numpy() == 0 or target[i].numpy() == 1:
            img = data[:, i, :].T
            # find firing neuron and its timestamp
            idx = np.where(img == 1)
            idx_n = np.unique(idx[0])   # fring neuron idx
            idx_t = []  # fring neuron idx
            for k in idx_n:
                idx_t.append(idx[1][idx[0] == k])
            colors2 = 'black'
            lineoffsets = idx_n
            linelengths = 1
            # create a horizontal plot (orientation='vertical')
            axs.eventplot(idx_t, colors=colors2, lineoffsets=lineoffsets, linelengths=linelengths)
            x_ticks = range(0, 21, 20)
            plt.xticks(x_ticks)
            y_ticks = range(0, 201, 200)
            plt.yticks(y_ticks)
            # plt.axis('scaled')  # 坐标轴原始比例
    plt.show()
    plt.savefig(str(target[i]) + '_' + str(i) + '.png', dpi=600)
    # print()
    print()


def plot_spiking2(data):
    '''
        data: template + samples
    '''
    # fig = plt.figure()
    row = 0
    for label in data:
        col = 0
        template = label[0]
        # (0,0)表示从第0行第0列开始作图，colspan表示列的跨度, rowspan表示行的跨度.
        axs1 = plt.subplot2grid((6, 4), (row + 1, col))
        col = col + 1
        plt.imshow(template.reshape((14, 14)))
        plt.colorbar()
        x_ticks = range(0, 14, 13)
        plt.xticks(x_ticks)
        y_ticks = range(0, 14, 13)
        plt.yticks(y_ticks)
        axs1.invert_yaxis()  # y轴坐标反向
        # axs1.set_title('Firing Template')  # 设置标题
        # axs1.tick_params(which='major', length=3)

        for i in range(1, len(label), 1):
            axs2 = plt.subplot2grid((6, 4), (row, col))
            col = col + 1
            if row < 5:
                axs2.tick_params(top=False, bottom=True, left=True, right=False)    # 坐标刻度显示
                axs2.tick_params(labeltop=False, labelbottom=False, labelleft=True, labelright=False)   # 刻度值显示
            plt.imshow(label[i].reshape((14, 14)))
            x_ticks = range(0, 14, 13)
            plt.xticks(x_ticks)
            y_ticks = range(0, 14, 13)
            plt.yticks(y_ticks)
            axs2.invert_yaxis()  # y轴坐标反向
            axs2.tick_params(which='major', length=3)
            # axs2.set_title('Sample')  # 设置标题

            axs3 = plt.subplot2grid((6, 4), (row, col), colspan=2)
            if row < 5:
                axs3.tick_params(top=False, bottom=True, left=True, right=False)
                axs3.tick_params(labeltop=False, labelbottom=False, labelleft=True, labelright=False)
            row = row + 1
            col = 1
            sptn = torch.from_numpy(label[i])
            sptn = fc.bernoulli(datum=sptn, time=50, dt=1, max_prob=1.0).float()
            sptn = sptn.numpy()
            idx_f = np.where(sptn == 1)
            idx_n = np.unique(idx_f[0])   # fring neuron idx
            idx_t = []  # fring neuron idx
            for k in idx_n:
                idx_t.append(idx_f[1][idx_f[0] == k])
            colors2 = 'black'
            lineoffsets = idx_n
            linelengths = 2
            # create a horizontal plot (orientation='vertical')
            axs3.eventplot(idx_t, colors=colors2, lineoffsets=lineoffsets, linelengths=linelengths)
            tick_params(which='major', length=3)
            x_ticks = range(0, 201, 200)
            plt.xticks(x_ticks)
            y_ticks = range(0, 51, 50)
            plt.yticks(y_ticks)
            # plt.xlabel('Afferent neurons')
            # plt.ylabel('Timesteps (ms)')
    # plt.show()
    plt.savefig('spiking.png', dpi=600)
    # print()


def plot_CM(cm, classes, title=None, cmap=plt.cm.Blues):
    plt.rc('font', family='sans-serif', size='15')   # 设置字体样式、大小
    # plt.rcParams['font.sans-serif'] = ['Tahoma', 'DejaVu Sans', 'SimHei', 'Lucida Grande', 'Verdana']  # 用来正常显示中文标签
    # plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    # plt.figure()
    plt.rcParams['figure.dpi'] = 200  # 分辨率

    # 按行进行归一化
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # 占比1%以下的单元格，设为0，防止在最后的颜色中体现出来
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if int(cm[i, j] * 100 + 0.5) == 0:
                cm[i, j] = 0

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)  # 侧边的颜色条带

    plt.title('Confusion matrix')
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=list(range(len(classes))), yticklabels=list(range(len(classes))),
           title=title,
           ylabel='Actual',
           xlabel='Predicted')

    # 通过绘制格网，模拟每个单元格的边框
    ax.set_xticks(np.arange(cm.shape[1] + 1) - .5, minor=True)
    ax.set_yticks(np.arange(cm.shape[0] + 1) - .5, minor=True)
    ax.grid(which="minor", color="gray", linestyle='-', linewidth=0.05)
    ax.tick_params(which="minor", bottom=False, left=False)

    # 将x轴上的lables旋转45度
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # 标注百分比信息
    fmt = 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if int(cm[i, j] * 100 + 0.5) > 0:
                ax.text(j, i, format(int(cm[i, j] * 100 + 0.5), fmt) + '%',
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.show()


def d(la, data):
    y = data[la - 1].squeeze()
    # x = range(0, 20, 19)
    # for i in range(y.shape[1]):   
    if y.shape[1] > 9000:
        plt.plot(y[:, 0:2000], color='red', alpha=0.2, linewidth=0.1)
    else:
        plt.plot(y, color='red', alpha=0.4, linewidth=0.1)
    plt.tick_params(labelsize=10)
    x_ticks = range(0, 20, 1)
    plt.xticks(x_ticks)
    # plt.ylim(0, 1)
    title = "Layer " + str(la)
    plt.title(title, fontsize=10)
    # plt.xlabel(u'Timesteps', fontsize=5)


def fi(data):
    fig = plt.figure(14)
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
                        wspace=None, hspace=1)
    plt.subplot(411)
    d(1, data)
    plt.subplot(412)
    d(2, data)
    plt.subplot(413)
    d(3, data)
    plt.subplot(414)
    d(4, data)
    # plt.show()
    plt.savefig("I.png", dpi=600)


if __name__ == "__main__":
    data = torch.load('F_I.pth')
    f_gate = data['F']
    i_gate = data['I']
    fi(i_gate)
    # fi(i_gate)