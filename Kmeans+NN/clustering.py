from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
import readData
import predict_util
from sklearn.cluster import KMeans


def Clustering(T, N_cluster, n_train, n_lag, load):
    print("clustering begins")
    #### k-means clustering ####
    n_train = 365 # days in training set
    
    X_train = np.zeros((n_train, T * n_lag))
    y_train = np.zeros((n_train, T ))
    
    #xaxis = range(T * n_lag)
    for i in range(n_train - n_lag):
        X_train[i, :] = load[i * T: (i + n_lag) * T]
        y_train[i, :] = load[(i + n_lag) * T:(i + n_lag + 1) * T]
    
    #N_cluster = 3
    kmeans = KMeans(N_cluster, random_state=0).fit(X_train)
    labels = kmeans.labels_ # labels of training data
    centers = kmeans.cluster_centers_ # kMeans centers
    #print(labels)
    #plot the results
    
    # init the groups of training sets
    X_train0 = np.zeros((0, T * n_lag))
    X_train1 = np.zeros((0, T * n_lag))
    X_train2 = np.zeros((0, T * n_lag))
    y_train0 = np.zeros((0, T ))
    y_train1 = np.zeros((0, T ))
    y_train2 = np.zeros((0, T ))
    
    # divide the training data to groups
    for i in range(n_train - n_lag):
        label = labels[i]
        if label == 0:
            #plt.step(xaxis, X_train[i,:], 'r')
            X_train0 = np.vstack([X_train0, X_train[i, :]])
            y_train0 = np.vstack([y_train0, y_train[i, :]])
        elif label == 1:
            #plt.step(xaxis, X_train[i,:], 'g')
            X_train1 = np.vstack([X_train1, X_train[i, :]])
            y_train1 = np.vstack([y_train1, y_train[i, :]])
        elif label == 2:
            #plt.step(xaxis, X_train[i,:], 'b')    
            X_train2 = np.vstack([X_train2, X_train[i, :]])
            y_train2 = np.vstack([y_train2, y_train[i, :]])
    
    
    ## test dataset
    n_days = int(load.size / T)
    
    X_test = np.zeros((n_days - n_train, T * n_lag))
    y_test = np.zeros((n_days - n_train, T )) 
    
    row = 0
    for i in range(n_train, n_days):
        X_test[row, :] = load[(i - n_lag) * T: i * T]
        y_test[row, :] = load[i * T: (i + 1) * T]
        row += 1
    
    test_labels = kmeans.predict(X_test)
    #print(test_labels.size)
    # init the groups of testing sets
    X_test0 = np.zeros((0, T * n_lag))
    X_test1 = np.zeros((0, T * n_lag))
    X_test2 = np.zeros((0, T * n_lag))
    y_test0 = np.zeros((0, T ))
    y_test1 = np.zeros((0, T ))
    y_test2 = np.zeros((0, T ))
    
    # divide the training data to groups
    for i in range(n_days - n_train):
        label = test_labels[i]
        if label == 0:
            #plt.step(xaxis, X_train[i,:], 'r')
            X_test0 = np.vstack([X_test0, X_test[i, :]])
            y_test0 = np.vstack([y_test0, y_test[i, :]])
        elif label == 1:
            #plt.step(xaxis, X_train[i,:], 'g')
            X_test1 = np.vstack([X_test1, X_test[i, :]])
            y_test1 = np.vstack([y_test1, y_test[i, :]])
        elif label == 2:
            #plt.step(xaxis, X_train[i,:], 'b')    
            X_test2 = np.vstack([X_test2, X_test[i, :]])
            y_test2 = np.vstack([y_test2, y_test[i, :]])
        
        
    print("clustering ends")   
    return (X_train0, y_train0, X_train1, y_train1, X_train2, y_train2, X_test0, X_test1, X_test2, y_test0, y_test1, y_test2)
          


  
if __name__ == "__main__":   
    # parameters
    T = 96
    n_train = 365
    n_lag = 2
    
    # import load data
    data = readData.loadResidentialData()
    n_customer = data.shape[1]
    # load sum, 2 years of data
    sumLoad = np.zeros((365 * 2 * T,))
    # sum up the load data
    for i in range(n_customer):
        customer_load = readData.getUserData(data, i)
        sumLoad += np.nan_to_num(customer_load)
    
    minLoad = np.min(sumLoad)
    maxLoad = np.max(sumLoad)
    sumLoad = (sumLoad - minLoad) / (maxLoad - minLoad)
           
    # call clustering function
    N_cluster = 3
    Clustering(T, N_cluster, n_train, n_lag, sumLoad)
        

            
    
    