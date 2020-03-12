'''
测试集：
 钢筋的数据，（xmin,ymin,xmax,ymax）
'''

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from utils import cnames
import random


if __name__ == '__main__':
    '''
        加载数据
    '''

    label_path = '../../data/detection/rebar/train_labels.csv'
    det = []
    with open(label_path,'r') as f:
        for li in f.readlines():
            if 'jpg' not in li:
                continue

            # xmin,ymin,xmax,ymax
            temp_box = np.fromstring(li.split(',')[-1],sep=' ')
            # w,h
            w,h = temp_box[2]-temp_box[0],temp_box[3]-temp_box[1]
            det.append(np.array([w,h]))


    det = np.asarray(det)
    '''
        归一化：
            1、保证w,h的相对比例不变
            2、保证不同框的尺寸大小不变
    '''

    max_size = np.max(det)
    det_nor = det/max_size

    '''
        聚类
    '''
    n_clusters = 9
    estimator = KMeans(n_clusters=n_clusters)
    estimator.fit(det_nor)
    label_pred = estimator.labels_  # 获取每个点的聚类标签
    centroids = estimator.cluster_centers_  # 获取聚类中心
    inertia = estimator.inertia_  # 最后各个点到各个聚类中心距离之和

    '''
        绘图
    '''
    colors=random.sample(cnames.keys(),n_clusters)
    plt.xlabel('width_nor')
    plt.ylabel('height_nor')
    for i in range(n_clusters):
        temp_det = det_nor[np.argwhere(label_pred==i)]
        temp_det = np.squeeze(temp_det,axis=1)
        plt.scatter(x=temp_det[:, 0], y=temp_det[:, 1], c=colors[i],edgecolors='black')

    plt.show()

