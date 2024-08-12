# -*- coding: utf-8 -*-

import os
import sys
import argparse
import numpy as np
from datetime import datetime
from sklearn import metrics
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
sys.path.append('../../')
from utils.dataset import read_data
from utils.HARN import HARN

torch.manual_seed(22)


device = torch.device("cuda")

parse = argparse.ArgumentParser()
parse.add_argument('-height', type=int, default=100)
parse.add_argument('-width', type=int, default=100)
parse.add_argument('-traffic', type=str, default='internet')
parse.add_argument('-nb_flow', type=int, default=1)
parse.add_argument('-close_size', type=int, default=3)
parse.add_argument('-loss', type=str, default='l1', help='l1 | l2')
parse.add_argument('-lr', type=float, default=0.01)
parse.add_argument('-weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parse.add_argument('-batch_size', type=int, default=32, help='batch size')
parse.add_argument('-epoch_size', type=int, default=300, help='epochs')
parse.add_argument('-rows', nargs='+', type=int, default=[40, 60])
parse.add_argument('-cols', nargs='+', type=int, default=[40, 60])
parse.add_argument('-last_kernel', type=int, default=1)
parse.add_argument('-test_size', type=int, default=24 * 7)
parse.add_argument('-fusion', type=int, default=1)
parse.add_argument('-crop', dest='crop', action='store_true')
parse.add_argument('-no-crop', dest='crop', action='store_false')
parse.set_defaults(crop=True)
parse.add_argument('-train', dest='train', action='store_true')
parse.add_argument('-no-train', dest='train', action='store_false')
parse.set_defaults(train=True)
parse.add_argument('-l2', dest='l2', help='weight decay', type=float, default=1e-4)
parse.add_argument('-adam', dest='adam', help='use adam. Not recommended', action='store_true')
parse.add_argument('-save_dir', type=str, default='results')

opt = parse.parse_args()

opt.save_dir = '{}/{}'.format(opt.save_dir, opt.traffic)
print("save dir:", opt.save_dir)
path_name = 'results_data'
if not os.path.exists(path_name):
    os.makedirs(path_name)
else:
    print('path already exists.')


def get_optim(lr):
    if opt.adam:
        optimizer = optim.Adam(model.parameters(), weight_decay=opt.l2, lr=lr, eps=1e-3)
    else:
        optimizer = optim.SGD(model.parameters(), weight_decay=opt.l2, lr=lr, momentum=0.9)

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                     milestones=[0.5 * opt.epoch_size, 0.75 * opt.epoch_size],
                                                     gamma=0.1)
    return optimizer, scheduler


def log(fname, s):
    if not os.path.isdir(os.path.dirname(fname)):
        os.system("mkdir -p " + os.path.dirname(fname))
    f = open(fname, 'a')
    f.write(str(datetime.now()) + ': ' + s + '\n')
    f.close()


EPOCH_NUM = 0


def predict(test_type='train'):
    predictions = []
    ground_truth = []
    loss = []
    model.eval()

    model.load_state_dict(torch.load('internet/model=HARN.pt'))
    if test_type == 'train':
        data = train_loader
    elif test_type == 'test':
        data = test_loader
    elif test_type == 'valid':
        data = valid_loader

    with torch.no_grad():
        for idx, (c, X_meta, target) in enumerate(data):
            optimizer.zero_grad()
            model.zero_grad()
            x = c.float().to(device)
            y = target.float().to(device)
            X_meta = X_meta.float().to(device)
            pred = model(x, X_meta, H)
            predictions.append(pred.data.cpu())
            ground_truth.append(target.data)
            loss.append(criterion(pred, y).item())

    final_predict = np.concatenate(predictions)
    ground_truth = np.concatenate(ground_truth)
    print(
        "Shape of final prediction is {}, shape of ground truth is {}".format(final_predict.shape, ground_truth.shape))

    ground_truth = mmn.inverse_transform(ground_truth)
    final_predict = mmn.inverse_transform(final_predict)
    return final_predict, ground_truth


def train_valid_split(dataloader, test_size=0.2, shuffle=True, random_seed=0):
    length = len(dataloader)
    indices = list(range(0, length))

    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    if type(test_size) is float:
        split = int(np.floor(test_size * length))
    elif type(test_size) is int:
        split = test_size
    else:
        raise ValueError('%s should be an int or float'.format(str))
    return indices[split:], indices[:split]


if __name__ == '__main__':

    path = 'data/data_git_version.h5'
    feature_path = 'data/crawled_feature.csv'
    X, X_meta, y_meta, X_cross, y, mmn = read_data(path, feature_path, opt)

    samples, sequences, channels, height, width = X.shape
    dis_mat = np.mat(np.load('dtw-.npy'))
    m_prob = 1.0
    H = np.zeros((dis_mat.shape[0], dis_mat.shape[0]))
    is_probH = True
    k_neig = 3
    for center_idx in range(dis_mat.shape[0]):
        dis_mat[center_idx, center_idx] = 0
        dis_vec = dis_mat[center_idx]
        nearest_idx = np.array(np.argsort(dis_vec)).squeeze()
        avg_dis = np.average(dis_vec)
        if not np.any(nearest_idx[:k_neig] == center_idx):
            nearest_idx[k_neig - 1] = center_idx

        for node_idx in nearest_idx[:k_neig]:
            if is_probH:
                H[node_idx, center_idx] = np.exp(-dis_vec[0, node_idx] ** 2 / (m_prob * avg_dis) ** 2)
            else:
                H[node_idx, center_idx] = 1.0

    x_train, x_test = X[:-opt.test_size], X[-opt.test_size:]

    meta_train, meta_test = X_meta[:-opt.test_size], X_meta[-opt.test_size:]
    cross_train, cross_test = X_cross[:-opt.test_size], X_cross[-opt.test_size:]

    y_tr = y[:-opt.test_size]
    y_te = y[-opt.test_size:]

    prediction_ct = 0
    truth_ct = 0


    y_train = y_tr
    y_test = y_te

    train_data = list(zip(*[x_train, meta_train, y_train]))
    test_data = list(zip(*[x_test, meta_test, y_test]))

    train_idx, valid_idx = train_valid_split(train_data, 0.1)
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = DataLoader(train_data, batch_size=opt.batch_size, sampler=train_sampler,
                              num_workers=0, pin_memory=True)
    valid_loader = DataLoader(train_data, batch_size=opt.batch_size, sampler=valid_sampler,
                              num_workers=0, pin_memory=True)

    test_loader = DataLoader(test_data, batch_size=opt.batch_size, shuffle=False)

    input_shape = X.shape
    meta_shape = X_meta.shape
    cross_shape = X_cross.shape

    model = HARN(input_shape, meta_shape,
                   cross_shape, nb_flows=opt.nb_flow).to(device)

    optimizer = optim.Adam(model.parameters(), opt.lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                     milestones=[0.5 * opt.epoch_size,
                                                                 0.75 * opt.epoch_size, 0.9 * opt.epoch_size],
                                                     gamma=0.1)

    if not os.path.exists(opt.save_dir):
        os.makedirs(opt.save_dir)
    if not os.path.isdir(opt.save_dir):
        raise Exception('%s is not a dir' % opt.save_dir)

    if opt.loss == 'l1':
        criterion = nn.L1Loss().cuda()
    elif opt.loss == 'l2':
        criterion = nn.MSELoss().cuda()

    pred, truth = predict('test')
    prediction_ct += pred
    truth_ct += truth

    if opt.traffic != 'internet':
        prediction_ct[-24] = ((truth_ct[-25] + truth_ct[-26] + truth_ct[-27]) / 3.0) * 2.5

    print('Final RMSE:{:0.5f}'.format(
        metrics.mean_squared_error(prediction_ct.ravel(), truth_ct.ravel()) ** 0.5))
    print('Final MAE:{:0.5f}'.format(
        metrics.mean_absolute_error(prediction_ct.ravel(), truth_ct.ravel())))

    Y = truth_ct.ravel()
    Y_hat = prediction_ct.ravel()

    print('Final R^2 Score: {:.4f}'.format(metrics.r2_score(Y, Y_hat)))
    print('Final Variance Score: {:.4f}'.format(metrics.explained_variance_score(Y, Y_hat)))
