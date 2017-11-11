from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
import readData
from sklearn.neighbors import KNeighborsRegressor
import predict_util

# generate historical data
# n_trains : number of days to construct kNN
# n_lag : number of lag days
# T: time intervals a day has
def kNN_historical(T, n_lag, n_train, load_data):
    X_train = np.zeros((n_train, T * n_lag))
    y_train = np.zeros((n_train, T ))
    
    for i in range(n_train - n_lag):
        X_train[i, :] = load_data[i * T: (i + n_lag) * T]
        y_train[i, :] = load_data[(i + n_lag) * T:(i + n_lag + 1) * T]
    
    return (X_train, y_train)

# kNN forecaster
# n_trains : number of days to construct kNN
# n_lag : number of lag days
# T: time intervals a day has
# minLoad: minimum load, comes from normalization
# maxLoad: maximum load, comes form normalization
def kNN_forecast(T, n_train, n_lag, load_data, minLoad, maxLoad):
    (X_train, y_train) = kNN_historical(T, n_lag, n_train, sumLoad)
    
    MAPE_sum = 0.0
    RMSPE_sum = 0.0
    
    #n_days = int(load_data.size / T)
    n_days = 365
    for d in range(n_train, n_days - 1):
        X_test = np.zeros((1, n_lag * T))
        X_test[0,:] = load_data[d * T - n_lag * T: d * T]
        y_test = load_data[d * T: d*T + T]
        
        # kNN
        neigh = KNeighborsRegressor(n_neighbors=10, weights = 'distance')
        neigh.fit(X_train, y_train)
        
        y_test_2d = np.zeros((1, T))
        y_test_2d[0, :] = y_test
        
        y_pred = neigh.predict(X_test)
        y_pred = y_pred * (maxLoad - minLoad) + minLoad
        y_test = y_test * (maxLoad - minLoad) + minLoad
        
        '''
        # plot daily forecast
        xaxis = range(T)
        plt.step(xaxis, y_pred.flatten(), 'r')
        plt.step(xaxis, y_test.flatten(), 'g')
        plt.show()
        '''
        # update the training set

        X_train = np.concatenate((X_train, X_test), axis = 0)
        y_train = np.concatenate((y_train, y_test_2d), axis = 0)
        
        mape = predict_util.calMAPE(y_test, y_pred)
        rmspe = predict_util.calRMSPE(y_test, y_pred)
        # update error metric results
        print('MAPE: %.2f, RMSPE: %.2f' % (mape, rmspe))
        MAPE_sum += mape
        RMSPE_sum += rmspe
    
    days_num = n_days - n_train
    
    return (MAPE_sum / days_num, RMSPE_sum / days_num)




if __name__ == "__main__":   
    # parameters
    T = 96
    n_train = 50
    n_lag = 1
    
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
    
    (MAPE_avg, RMSPE_avg) = kNN_forecast(T, n_train, n_lag, sumLoad, minLoad, maxLoad)
    print('forecast result MAPE: %.2f, RMSPE: %.2f' % (MAPE_avg, RMSPE_avg))
    