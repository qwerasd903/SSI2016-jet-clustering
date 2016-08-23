# -*- coding: utf-8 -*-
"""
Created on Mon Aug 22 21:34:51 2016

@author: Jay
"""

print(__doc__)

import numpy as np

from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler
from read_data_jay import read_data

X=read_data()

db = DBSCAN(eps=0.4, min_samples=3).fit(X)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_
# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) #- (1 if -1 in labels else 0)
print('Estimated number of clusters: %d' % n_clusters_)

#print(labels)
import matplotlib.pyplot as plt
from itertools import cycle

plt.close('all')
plt.figure(1)
plt.clf()
for k in range(1,len(X)):
    if labels[k-1]==0:
        plt.plot(X[k-1][0],X[k-1][1],'ro')
    else:
        if labels[k-1]==1:
            plt.plot(X[k-1][0],X[k-1][1],'bo')
        else:
            plt.plot(X[k-1][0],X[k-1][1],'go')
plt.axis([-3,3,-math.pi,math.pi])
plt.show()