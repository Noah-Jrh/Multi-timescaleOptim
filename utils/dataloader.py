# -*- coding: utf-8 -*-
# @Time     : 2019/7/20 下午7:34
# @Author   : Noah
# @Email    : 15520816169@163.com
# @File     : dataloader.py
# @Software : PyCharm
# @Update   :
"""Instruction:
    Load Data
"""

import os
import os.path
import errno
import copy
import torch
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as data
import numpy as np
import scipy.io as io
from PIL import Image
import matplotlib.pyplot as plt
# import math
from utils.event import Events
from utils.plot import plot_spiking2


def loader(var, data_path, args):
    if var == 'mnist':
        dataset = load_mnist(data_path, args.batch_size)
    elif var == 'omniglot':
        dataset = load_omniglot(data_path, args)
    elif var == 'gesture_dvs':
        dataset = load_gesture_dvs(data_path, args)
    elif var == 'spiking':
        dataset = load_spikedata(data_path, args)
    else:
        raise (ValueError('Unsupported dataset'))
    return dataset


def load_mnist(data_path: str, batch_size: int):
    """ Load MNIST
    :param data_path:
    :param batch_size:
    :rtype: Tuple[torch.Tensor, torch.Tensor]
    """
    train_dataset = torchvision.datasets.MNIST(root=data_path, train=True, download=True,
                                               transform=transforms.ToTensor())
    train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    test_set = torchvision.datasets.MNIST(root=data_path, train=False, download=True,
                                          transform=transforms.ToTensor())
    test_loader = data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)
    # constant = {'n_image': 60000, 'n_label': 10}
    return train_loader, test_loader


def load_omniglot(data_path, args):
    dataset = EpisodeDataset(data_path, args)
    return dataset


def load_spikedata(data_path, args):
    dataset = EpisodeDataset(data_path, args)
    return dataset


def load_gesture_dvs(data_path, args):
    dataset = EpisodeDataset(data_path, args)
    return dataset


class Event2Frame(object):
    """ Convert DVS event streams to frames
    Args:
        snr （float）
        p (float):
    """

    def __init__(self, tr, img_size, T):
        self.tr = tr    # time resolution
        self.img_size = img_size
        self.T = T      # time resolution

    def __call__(self, sample):
        """
        Args:
            sample: mat
        Returns:
            frame: numpy array
        """
        events = io.loadmat(sample, squeeze_me=True, struct_as_record=False)
        frame = np.zeros([self.T, self.img_size, self.img_size], dtype=int)  # frame
        for j in range(0, self.T * self.tr, self.tr):    # tr ms 的帧
            idx = (events['TD'].ts >= j) & (events['TD'].ts < j + self.tr)  # 找到每个frame的索引范围
            frame[int(j / self.tr), events['TD'].y[idx] - 1, events['TD'].x[idx] - 1] = 1.0
        #     im = Image.fromarray((frame[int(j / self.tr), :] * 255).astype(np.uint8), "L")  # numpy 转 image类
        #     im.save(os.path.join('./', str(j / self.tr) + '.png'))
        # print()
        return np.reshape(frame, (self.T, self.img_size * self.img_size))


class Gesture_DVS(data.Dataset):
    '''
    '''

    def __init__(self, root, transform=None, target_transform=None, download=False):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform

        # 检查数据集路径
        if not os.path.exists(os.path.join(self.root)):
            if download:
                self.download()
            else:
                raise RuntimeError('Dataset not found.' + ' You can use download=True to download it')

        self.all_items = find_classes(os.path.join(self.root))
        self.idx_classes = index_classes(self.all_items)

    def __getitem__(self, index):
        filename = self.all_items[index][0]
        classname = self.all_items[index][1]
        filepath = self.all_items[index][2]

        img = os.path.join(str(filepath), str(filename))
        target = self.idx_classes[classname]
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def __len__(self):
        return len(self.all_items)


class Omniglot(data.Dataset):
    urls = [
        'https://github.com/brendenlake/omniglot/raw/master/python/images_background.zip',
        'https://github.com/brendenlake/omniglot/raw/master/python/images_evaluation.zip'
    ]
    raw_folder = 'raw'
    processed_folder = 'processed'
    training_file = 'training.pt'
    test_file = 'test.pt'

    '''
    The items are (filename,category). The index of all the categories can be found in self.idx_classes
    Args:
    - root: the directory where the dataset will be stored
    - transform: how to transform the input
    - target_transform: how to transform the target
    - download: need to download the dataset
    '''

    def __init__(self, root, transform=None, target_transform=None, download=False):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform

        if not self._check_exists():
            if download:
                self.download()
            else:
                raise RuntimeError('Dataset not found.' + ' You can use download=True to download it')

        self.all_items = find_classes(os.path.join(self.root, self.processed_folder))
        self.idx_classes = index_classes(self.all_items)

    def __getitem__(self, index):
        filename = self.all_items[index][0]
        classname = self.all_items[index][1]
        filepath = self.all_items[index][2]

        img = os.path.join(str(filepath), str(filename))
        target = self.idx_classes[classname]
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def __len__(self):
        return len(self.all_items)

    def _check_exists(self):
        return os.path.exists(os.path.join(self.root, self.processed_folder, "images_evaluation")) and \
            os.path.exists(os.path.join(self.root, self.processed_folder, "images_background"))

    def download(self):
        from six.moves import urllib
        import zipfile

        if self._check_exists():
            return

        # download files
        try:
            os.makedirs(os.path.join(self.root, self.raw_folder))
            os.makedirs(os.path.join(self.root, self.processed_folder))
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise

        for url in self.urls:
            print('== Downloading ' + url)
            img = urllib.request.urlopen(url)
            filename = url.rpartition('/')[2]
            file_path = os.path.join(self.root, self.raw_folder, filename)
            with open(file_path, 'wb') as f:
                f.write(img.read())
            file_processed = os.path.join(self.root, self.processed_folder)
            print("== Unzip from " + file_path + " to " + file_processed)
            zip_ref = zipfile.ZipFile(file_path, 'r')
            zip_ref.extractall(file_processed)
            zip_ref.close()
        print("Download finished.")


def find_classes(root_dir):
    # 统计样本数量
    retour = []
    for (root, dirs, files) in os.walk(root_dir):
        for f in files:
            if f.endswith("png") or f.endswith("mat"):
                r = root.split('/')
                lr = len(r)
                retour.append((f, r[lr - 2] + "/" + r[lr - 1], root))
    print("== Found %d items " % len(retour))
    return retour


def index_classes(items):
    # 统计类别数量
    idx = {}
    for i in items:
        if i[1] not in idx:
            idx[i[1]] = len(idx)
    print("== Found %d classes" % len(idx))
    return idx


class EpisodeDataset:
    def __init__(self, data_path, args):
        self.args = args
        self.root = data_path
        self.imgsz = args.img_size
        self.batchsz = args.batchSize          # 1
        self.n_way = args.n_way                # n way 5
        self.k_shot = args.k_shot              # k shot 1
        self.k_query = args.k_query            # k query 15
        assert (args.k_shot + args.k_query) <= 20

        if args.preprocess is None:
            self.filename = os.path.join(data_path, args.dataSet + '.npy')
            if not os.path.isfile(self.filename):
                self.x = self.proprocess(args.dataSet)
                np.save(self.filename, self.x)
                print('write into ' + self.filename)
            else:
                self.x = np.load(self.filename)
                print('load from ' + self.filename)
        elif args.preprocess == 'hmax':
            self.filename = os.path.join(data_path, args.dataSet + '.mat')
            if not os.path.isfile(self.filename):
                raise NameError("File {} is not exist".format(self.filename))
            else:
                x = io.loadmat(self.filename)
                self.x = x['features']
                print('load from ' + self.filename)
        else:
            raise NameError("Preprocess {} not recognized".format(args.preprocess))

        self.n_cls = self.x.shape[0]    # 1623
        self.inpt_size = self.x.shape[-1]
        # [1623, 20, 784]
        if args.dataSet == 'omniglot' or args.dataSet == 'spiking':
            self.x_train, self.x_test = self.x[:1200], self.x[1200:]
        elif args.dataSet == 'gesture_dvs':
            self.x_train, self.x_test = self.x[:6], self.x[6:]
        # self.normalization()

        # save pointer of current read batch in total cache
        self.indexes = {"train": 0, "test": 0}
        self.datasets = {"train": self.x_train, "test": self.x_test}  # original data cached
        print("DB: train", self.x_train.shape, "test", self.x_test.shape)

        self.datasets_cache = {"train": self.load_data_cache(self.datasets["train"]),  # current epoch data cached
                               "test": self.load_data_cache(self.datasets["test"])}

    def proprocess(self, dataset):
        if dataset == 'omniglot':
            x = Omniglot(self.root, download=True,
                         transform=transforms.Compose([lambda z: Image.open(z).convert('L'),
                                                       lambda z: z.resize((self.imgsz, self.imgsz)),
                                                       lambda z: np.reshape(z, (self.imgsz * self.imgsz)),
                                                       lambda z: 1.0 - z / 255.]))
            temp = dict()  # {label:img1, img2..., 20 imgs, label2: img1, img2,... in total, 1623 label}
            for (img, label) in x:  # 调用__getitem__
                if label in temp.keys():
                    temp[label].append(img)
                else:
                    temp[label] = [img]
            x = []
            for label, imgs in temp.items():  # labels info deserted , each label contains 20imgs
                x.append(np.array(imgs))
            # as different class may have different number of imgs
            x = np.array(x).astype(np.float)  # [[20 imgs],..., 1623 classes in total]

        elif dataset == 'gesture_dvs':
            x = Gesture_DVS(self.root, download=False, transform=Event2Frame(self.args.resolution, self.args.img_size, self.args.T))
            temp = dict()  # {label:img1, img2..., 20 imgs, label2: img1, img2,... in total, 1623 label}
            for (img, label) in x:
                if label in temp.keys():
                    temp[label].append(img)
                else:
                    temp[label] = [img]
            x = []
            for label, imgs in temp.items():  # labels info deserted , each label contains 20imgs
                x.append(np.array(imgs))
            # as different class may have different number of imgs
            x = np.array(x).astype(np.float)  # [[20 imgs],..., 1623 classes in total]

        elif dataset == 'spiking':
            label_num = 1623
            img_num = 20  # 3 for plot
            neuron_num = 196
            spike_num = 98
            x = []
            for i in range(label_num):
                y = []
                select_neuron = np.random.choice(neuron_num, spike_num, False)
                # for plot
                # template = np.zeros(neuron_num)
                # template[select_neuron] = 1
                # y.append(template)
                for j in range(img_num):
                    spikeimg = np.zeros(neuron_num)
                    # 0.3的概率删除
                    idx = np.random.choice(select_neuron, int(spike_num * 0.7), False)
                    # 发放神经元发放频率从均匀分布中采样
                    select_rate = np.random.uniform(0.01, 0.1, int(spike_num * 0.7))
                    spikeimg[idx] = select_rate
                    y.append(spikeimg)
                x.append(y)
                # if i == 1:
                #     plot_spiking2(x)
            x = np.array(x).astype(np.float)
            return x
        else:
            raise NameError("Dataset {} not recognized".format(self.filename))
        return x

    def load_data_cache(self, data_pack):
        """
        Collects several batches data for N-shot learning
        :param data_pack: [cls_num, 20, 1, 28, 28]
        :return: A list with [support_set_x, support_set_y, target_x, target_y] ready to be fed to our networks
        """
        #  take 5 way 1 shot as example: 5 * 1
        setsz = self.k_shot * self.n_way        # 1 * 5
        querysz = self.k_query * self.n_way     # 15 * 5
        data_cache = []

        # print('preload next 50 caches of batchsz of batch.')
        for sample in range(10):  # num of episodes
            x_spts, y_spts, x_qrys, y_qrys = [], [], [], []
            for i in range(self.batchsz):  # one batch means one set
                x_spt, y_spt, x_qry, y_qry = [], [], [], []
                selected_cls = np.random.choice(data_pack.shape[0], self.n_way, False)
                for j, cur_class in enumerate(selected_cls):
                    # select 1-shot and 15-query images from cur_class(20 images)
                    selected_img = np.random.choice(data_pack.shape[1], self.k_shot + self.k_query, False)
                    # meta-training and meta-test
                    x_spt.append(data_pack[cur_class][selected_img[:self.k_shot]])
                    x_qry.append(data_pack[cur_class][selected_img[self.k_shot:]])
                    y_spt.append([j for _ in range(self.k_shot)])
                    y_qry.append([j for _ in range(self.k_query)])

                # shuffle inside a batch
                perm = np.random.permutation(self.n_way * self.k_shot)
                x_spt = np.array(x_spt).reshape(self.n_way * self.k_shot, -1)[perm]
                y_spt = np.array(y_spt).reshape(self.n_way * self.k_shot)[perm]
                perm = np.random.permutation(self.n_way * self.k_query)
                x_qry = np.array(x_qry).reshape(self.n_way * self.k_query, -1)[perm]
                y_qry = np.array(y_qry).reshape(self.n_way * self.k_query)[perm]

                # append [sptsz, 1, 84, 84] => [b, setsz, 1, 84, 84]
                x_spts.append(x_spt)
                y_spts.append(y_spt)
                x_qrys.append(x_qry)
                y_qrys.append(y_qry)

            # [b, setsz, 1, 28, 28]
            x_spts = np.array(x_spts).astype(np.float32).reshape(self.batchsz, setsz, -1)
            y_spts = np.array(y_spts).astype(np.int).reshape(self.batchsz, setsz)
            # [b, qrysz, 1, 28, 28]
            x_qrys = np.array(x_qrys).astype(np.float32).reshape(self.batchsz, querysz, -1)
            y_qrys = np.array(y_qrys).astype(np.int).reshape(self.batchsz, querysz)

            data_cache.append([x_spts, y_spts, x_qrys, y_qrys])
        return data_cache

    def next(self, mode='train'):
        """
        Gets next batch from the dataset with name.
        :param mode: The name of the splitting (one of "train", "val", "test")
        :return:
        """
        # update cache if indexes is larger cached num
        if self.indexes[mode] >= len(self.datasets_cache[mode]):
            self.indexes[mode] = 0
            self.datasets_cache[mode] = self.load_data_cache(self.datasets[mode])

        next_batch = self.datasets_cache[mode][self.indexes[mode]]
        self.indexes[mode] += 1

        return next_batch


def episode_data(datas, mode, args):
    x_spt, y_spt, x_qry, y_qry = datas.next(mode)
    sptsz = args.n_way * args.k_shot
    qrysz = args.n_way * args.k_query
    # if args.structure == 'SCN_3':
    #     x_spt = torch.from_numpy(x_spt).view(sptsz, -1).to(args.device)
    #     x_qry = torch.from_numpy(x_qry).view(qrysz, -1).to(args.device)
    #     y_spt = torch.from_numpy(y_spt).view(sptsz).to(args.device)
    #     y_qry = torch.from_numpy(y_qry).view(qrysz).to(args.device)
    if args.structure[0:3] == 'SNN':
        x_spt = torch.from_numpy(x_spt).view(sptsz, -1).to(args.device)
        x_qry = torch.from_numpy(x_qry).view(qrysz, -1).to(args.device)
        y_spt = torch.from_numpy(y_spt).view(sptsz).to(args.device)
        y_qry = torch.from_numpy(y_qry).view(qrysz).to(args.device)
    if args.structure[0:3] == 'CNN' and args.dataSet == 'omniglot':
        x_spt = torch.from_numpy(x_spt).view(sptsz, 1, args.img_size, args.img_size).to(args.device)
        x_qry = torch.from_numpy(x_qry).view(qrysz, 1, args.img_size, args.img_size).to(args.device)
        y_spt = torch.from_numpy(y_spt).view(sptsz).to(args.device)
        y_qry = torch.from_numpy(y_qry).view(qrysz).to(args.device)
    if args.structure[0:3] == 'SCN' and args.dataSet == 'omniglot':
        x_spt = torch.from_numpy(x_spt).view(sptsz, 1, args.img_size, args.img_size).to(args.device)
        x_qry = torch.from_numpy(x_qry).view(qrysz, 1, args.img_size, args.img_size).to(args.device)
        y_spt = torch.from_numpy(y_spt).view(sptsz).to(args.device)
        y_qry = torch.from_numpy(y_qry).view(qrysz).to(args.device)
    if args.structure[0:3] == 'SCN' and args.dataSet == 'gesture_dvs':
        x_spt = torch.from_numpy(x_spt).view(sptsz, 1, args.T, args.img_size, args.img_size).permute(2, 0, 1, 3, 4).to(args.device)
        x_qry = torch.from_numpy(x_qry).view(qrysz, 1, args.T, args.img_size, args.img_size).permute(2, 0, 1, 3, 4).to(args.device)
        y_spt = torch.from_numpy(y_spt).view(sptsz).to(args.device)
        y_qry = torch.from_numpy(y_qry).view(qrysz).to(args.device)
    # for i in range(args.T):
    #     im = Image.fromarray((x_spt[i, 0, 0, :] * 255).numpy().astype(np.uint8), "L")  # numpy 转 image类
    #     im.save('./' + str(i) + '.png')
    return x_spt, x_qry, y_spt, y_qry
