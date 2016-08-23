from sklearn.cluster import AffinityPropagation
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
import numpy as np
import csv
import numpy as np
import math

def read_data(filename='JetGenerator/testing.txt'):
    '''
    definition:
    -----------
        transform data from txt file to numpy ndarrays
    args:
    -----
        filename (optional): string with path to the data file
    returns:
    --------
        X: list of length n_events of numpy ndarrays of dimensions [n_points, 2],
           which represents the etas and phis of all points in all events
        e: list of length n_events of numpy ndarrays of dimension [n_points, 1],
           which represents the energies of all points in all events
        -- e.g.: to get the numpy array with etas and phis in the first event,
                 use X[0]
    '''
    X_flat = []
    #e_flat = []
    #n_flat=[]
    with open(filename, 'r') as csvfile:
        reader = csv.DictReader(csvfile, delimiter=' ')
        for row in reader:
            X_flat.append([int(row['event']), float(row['eta']), float(row['phi'])])
            #e_flat.append([int(row['event']), float(row['energy'])])
            #n_flat.append(int(row['event']))
    #X_flat = np.array(X_flat)
    #e_flat = np.array(e_flat)
    #n_flat = np.array(n_flat)

    X = []
    for ev in range(0,len(X_flat)-1):
        if X_flat[ev][0]==2:
            X.append([X_flat[ev][1],X_flat[ev][2]])
    return X#, n_flat
#x,n=read_data()
#print(n)
X=read_data()
import matplotlib.pyplot as plt
from itertools import cycle

#plt.close('all')
#plt.figure(1)
#plt.clf()
#for k in range(1,len(X)):
#    plt.plot(X[k][0],X[k][1],'ro')
#plt.axis([-3,3,-math.pi,math.pi])
#plt.show()