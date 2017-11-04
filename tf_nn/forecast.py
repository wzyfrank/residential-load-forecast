import numpy as np
import readData
import genTrainValidTest
import predict_util
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.ensemble import RandomForestRegressor

if __name__ == "__main__":
    userNo = 10
    
    data = readData.loadResidentialData()
    
    sumLoad = np.zeros((35040,))
    #userLoad = readData.getUserData(data, 0)
    for i in range(144):
        sumLoad += readData.getUserData(data, i)
    
    Ndays = 365
    T = 96
    n_train = 20
    n_valid = 0
    n_lag = 7
    
    MAPE_sum = 0.0
    RMSPE_sum = 0.0
    
    for d in range(n_train + n_lag, Ndays-1):
        (X_train, y_train, X_valid, y_valid, X_test, y_test) = genTrainValidTest.genData(sumLoad, n_train, n_valid, n_lag, T, d)
        max_load = np.max(X_train)
        min_load = np.min(X_train)  
        X_train = (X_train-min_load) / (max_load - min_load)
        y_train = (y_train-min_load) / (max_load - min_load)
        X_test = (X_test-min_load) / (max_load - min_load)
        
        rf = RandomForestRegressor(n_estimators = 100)
        rf.fit(X_train, y_train)
        
        y_pred = rf.predict(X_test)
        y_pred = y_pred * (max_load - min_load) + min_load
        
        mape = predict_util.calMAPE(y_test, y_pred)
        rmspe = predict_util.calRMSPE(y_test, y_pred)

        # update error metric results
        print('MAPE: %.2f, RMSPE: %.2f' % (mape, rmspe))
        MAPE_sum += mape
        RMSPE_sum += rmspe
    
        # plot (make this one module)
        xaxis = range(T)
        plt.step(xaxis, y_pred.flatten(), 'r')
        plt.step(xaxis, y_test.flatten(), 'g')
        red_patch = mpatches.Patch(color='red', label='prediction')
        green_patch = mpatches.Patch(color='green', label='actual')
        plt.legend(handles=[red_patch, green_patch])
        plt.show()
        
        
    days_sample = Ndays - n_train - n_lag
    MAPE_avg_rf = MAPE_sum / days_sample
    RMSPE_avg_rf = RMSPE_sum / days_sample
    print('forecast result MAPE: %.2f, RMSPE: %.2f' % (MAPE_avg_rf, RMSPE_avg_rf))