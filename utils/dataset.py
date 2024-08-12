# -*- coding: utf-8 -*-
import h5py
import numpy as np
from pandas import to_datetime
import pandas as pd
from sklearn import cluster


class MinMaxNorm01(object):
    def __init__(self):
        pass

    def fit(self, x):
        self.min = x.min()
        self.max = x.max()


    def transform(self, x):
        x = 1.0 * (x - self.min) / (self.max - self.min)
        return x

    def fit_transform(self, x):
        self.fit(x)
        return self.transform(x)

    def inverse_transform(self, x):
        x = x * (self.max - self.min) + self.min
        return x


def get_date_feature(idx):
    a = idx.weekday()
    v = [0 for _ in range(7)]
    v[a] = 1

    arr = np.zeros(24).tolist()
    b = idx.hour
    arr[b] = 1

    if a == 5:
        v.append(1)
        v.append(0)
    elif a==6:
        v.append(0)
        v.append(1)
    else:
        v.append(0)
        v.append(0)
    arr.extend(v)
    return arr


def traffic_loader(f, feature_path, opt):
    feature_names = ['social', 'BSs', 'poi_1', 'poi_2']
    feature_data = pd.read_csv(feature_path, header=0)
    feature_data.columns = feature_names

    feature = np.reshape(feature_data.values, (opt.height, opt.width, 4))
    if opt.nb_flow == 1:
        if opt.traffic == 'internet':
            data = f['data'][:, :, 2]
        else:
            raise IOError("Unknown traffic type")
        result = data.reshape((-1, 1, opt.height, opt.width))


        if opt.crop:
            result = result[:, :, opt.rows[0]:opt.rows[1], opt.cols[0]:opt.cols[1]]
            feature = feature[opt.rows[0]:opt.rows[1], opt.cols[0]:opt.cols[1], :]
        return result, feature

    elif opt.nb_flow == 2:
        if opt.traffic == 'internet':
            data = f['data'][:, :, 4]
            result = data.reshape((-1, 1, opt.height, opt.width))
        else:
            raise IOError("Unknown traffic type")
        
        if opt.crop:
            result = result[:, :, opt.rows[0]:opt.rows[1], opt.cols[0]:opt.cols[1]]
            feature = feature[opt.rows[0]:opt.rows[1], opt.cols[0]:opt.cols[1], :]
        return result, feature

    else:
        print("Wrong parameter with nb_flow")
        exit(0)


def get_label(data, feature, index, clusters):
    samples, channels, h, w = data.shape
    sum_data = np.sum(data, axis=1).reshape((samples, h*w))
    df_data = pd.DataFrame(sum_data, index=index)

    df_data = df_data.resample('1D').sum().transpose()

    feature = pd.DataFrame(np.reshape(feature, (h*w, -1)))
    df = pd.concat([df_data, feature], axis=1)
    df.fillna(0, inplace=True)
    clf = cluster.AgglomerativeClustering(n_clusters=clusters)
    clf.fit(df)
    return clf.labels_

def get_label_v2(data, feature, index, clusters):

    all_data = data.sum().sum() + np.sum(feature, axis=-1)
    mask = all_data.astype(bool)
    from sklearn.feature_extraction import image
    graph = image.img_to_graph(all_data, mask=mask)
    graph.data = np.exp(-graph.data / graph.data.std())
    labels = cluster.spectral_clustering(graph, n_clusters=clusters, eigen_solver='arpack')
    return labels


def read_data(path, feature_path, opt):
    f = h5py.File(path, 'r')
    data, feature_data = traffic_loader(f, feature_path, opt)

    index = f['idx'].value.astype(str)
    index = to_datetime(index, format='%Y-%m-%d %H:%M')

    mmn = MinMaxNorm01()
    data_scaled = mmn.fit_transform(data)

    X, y = [], []
    X_meta, y_meta = [], []

    h, w = data.shape[2], data.shape[3]

    for i in range(0, len(data)-opt.close_size):
        xc_ = [data_scaled[i + c][:,:,:] for c in range(0, opt.close_size)]

        x_meta = []
        for cc in range(0, opt.close_size):
            aar = get_date_feature(index[i + cc])
            aar = np.asarray(aar)
            x_meta.append(aar)
        if opt.close_size > 0:
            X.append(xc_)
            X_meta.append(x_meta)

        y.append(data_scaled[i+opt.close_size][:,:,:])
        aar = get_date_feature(index[i+opt.close_size])
        aar = np.asarray(aar)
        y_meta.append(aar)
    X = np.asarray(X)
    print("X.shape",X.shape)
    X_meta = np.asarray(X_meta)
    X_cross = np.asarray(feature_data)
    X_cross = np.reshape(X_cross, (h * w, -1))
    y = np.asarray(y)

    X_cross = np.moveaxis(X_cross, 0, -1)
    X_crossdata = np.repeat(X_cross, X.shape[0]).reshape((-1, 4, h, w))

    return X, X_meta,y_meta, X_crossdata, y, mmn