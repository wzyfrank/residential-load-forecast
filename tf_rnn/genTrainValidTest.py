import numpy as np

# user_load: load of one single user, 1d array
# n_train: number of days in trainig set
# n_valid: number of days in validation set
# n_lag: lag for load
# curr_day: current day (Test day)
def genData(user_load, n_train, n_valid, n_lag, T, curr_day):
    ################## generate data ##########################################
    y_train = np.zeros((n_train, T))
    X_train = np.zeros((n_train, T * n_lag))
    
    y_valid = np.zeros((n_valid, T))
    X_valid = np.zeros((n_valid, T * n_lag))
    
    row = 0
    for train_day in range(curr_day - n_train - n_valid, curr_day - n_valid):
        y_train[row,:] = user_load[train_day * T : train_day * T + T]
        X_train[row,0*T*n_lag:1*T*n_lag] = user_load[train_day * T - n_lag * T: train_day * T]
        row += 1
    
    row = 0
    for valid_day in range(curr_day - n_valid, curr_day):
        y_valid[row,:] = user_load[valid_day * T : valid_day * T + T]
        X_valid[row,0*T*n_lag:1*T*n_lag] = user_load[valid_day * T - n_lag * T: valid_day * T]
        row += 1    
        
    # building test data
    X_test = np.zeros((1, T * n_lag))
    X_test[0, 0*T*n_lag:1*T*n_lag] = user_load[curr_day*T - n_lag*T: curr_day*T]
    y_test = user_load[curr_day*T: curr_day *T + T]
    
    
    return(X_train, y_train, X_valid, y_valid, X_test, y_test)