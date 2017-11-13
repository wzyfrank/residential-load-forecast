import pandas as pd
import numpy as np
import readData
import predict_util
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import clustering
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor

#### neural network forecast
def SVR_forecast(n_lag, T, X_train, y_train, X_test, y_test, maxLoad, minLoad):    
    # SVR regressor
    clf = SVR(C=10, epsilon=0.01)
    # stack SVRs together to forecast multiple outputs
    multiSVR = MultiOutputRegressor(clf)

    
    
    ################## prediction #########################################
    test_days = X_test.shape[0]
    MAPE_sum = 0
    RMSPE_sum = 0
    for d in range(test_days):                
        # train SVR model
        multiSVR.fit(X_train, y_train) 
        
        # prepare test data
        X_test_d = np.zeros((1, n_lag * T))
        X_test_d[0,:] = X_test[d,:]
        y_test_d = y_test[d,:]
        y_pred = multiSVR.predict(X_test_d)
        y_pred = y_pred * (maxLoad - minLoad) + minLoad
        y_test_n = y_test_d * (maxLoad - minLoad) + minLoad

        mape = predict_util.calMAPE(y_test_n, y_pred)
        rmspe = predict_util.calRMSPE(y_test_n, y_pred)
        
        # update error metric results
        print('MAPE: %.2f, RMSPE: %.2f' % (mape, rmspe))
        MAPE_sum += mape
        RMSPE_sum += rmspe
        
        # update training set
        X_train = np.concatenate((X_train, X_test_d), axis = 0)
        y_train = np.vstack([y_train, y_test_d])
        
    return (MAPE_sum / test_days, RMSPE_sum / test_days, test_days)

    
if __name__ == "__main__":
    # parameters
    T = 96
    n_train = 365
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
           
    # call clustering function
    N_cluster = 3
    (X_train0, y_train0, X_train1, y_train1, X_train2, y_train2, X_test0, X_test1, X_test2, y_test0, y_test1, y_test2) = clustering.Clustering(T, N_cluster, n_train, n_lag, sumLoad)
    
    
    # neural network forecast
    print("start NN forecast on group 0")
    (MAPE0, RMSPE0, days0) = SVR_forecast(n_lag, T, X_train0, y_train0, X_test0, y_test0, maxLoad, minLoad)
    print('forecast result group 0 : MAPE: %.2f, RMSPE: %.2f' % (MAPE0, RMSPE0))
    
    print("start NN forecast on group 1")
    (MAPE1, RMSPE1, days1) = SVR_forecast(n_lag, T, X_train1, y_train1, X_test1, y_test1, maxLoad, minLoad)
    print('forecast result group 1 : MAPE: %.2f, RMSPE: %.2f' % (MAPE1, RMSPE1))
    
    print("start NN forecast on group 2")
    (MAPE2, RMSPE2, days2) = SVR_forecast(n_lag, T, X_train2, y_train2, X_test2, y_test2, maxLoad, minLoad)
    print('forecast result group 2 : MAPE: %.2f, RMSPE: %.2f' % (MAPE2, RMSPE2))
    
    
    # overall average
    MAPE = (MAPE0 * days0 + MAPE1 * days1 + MAPE2 * days2) / (days0 + days1 + days2)
    RMSPE = (RMSPE0 * days0 + RMSPE1 * days1 + RMSPE2 * days2) / (days0 + days1 + days2)
    print(' ')
    print('overall forecast result MAPE: %.2f, RMSPE: %.2f' % (MAPE, RMSPE))
    
    