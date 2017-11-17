import numpy as np
from minisom import MiniSom 

def callSOM(M, N, T, n_train, n_lag, load):
    # change load to 2d  
    n_days = int(load.size / T)     
        
    # init the SOM object
    # train a M*N 2d map
    som = MiniSom(M, N, T * n_lag, sigma=0.3, learning_rate=0.5)
    
    # train the SOM
    X_train = np.zeros((n_train, T * n_lag))
    y_train = np.zeros((n_train, T ))
    
    for i in range(n_train - n_lag):
        X_train[i, :] = load[i * T: (i + n_lag) * T]
        y_train[i, :] = load[(i + n_lag) * T:(i + n_lag + 1) * T]
        
    som.train_random(X_train, 1000)
        
    # classify training data
    X_train_dict = dict()
    y_train_dict = dict()
    for i in range(n_train - n_lag):
        node = som.winner(X_train[i, :])
        node = node[0] * N + node[1]
        #print(node)
        if node in X_train_dict:
            X_train_dict[node] = np.vstack((X_train_dict[node], X_train[i, :]))
            y_train_dict[node] = np.vstack((y_train_dict[node], y_train[i, :]))
        else:
            X_train_dict[node] = np.zeros((1, T * n_lag))
            X_train_dict[node][0, :] = X_train[i, :]
            y_train_dict[node] = np.zeros((1, T * n_lag))
            y_train_dict[node][0, :] = y_train[i, :]        


    # generate test data    
    X_test = np.zeros((n_train, T * n_lag))
    y_test = np.zeros((n_train, T ))
    row = 0
    for i in range(n_train - n_lag, n_days - n_lag):
        X_test[row, :] = load[i * T: (i + n_lag) * T]
        y_test[row, :] = load[(i + n_lag) * T:(i + n_lag + 1) * T]
        row += 1
        
    # classify test data
    X_test_dict = dict()
    y_test_dict = dict()
    for i in range(n_days - n_train):
        node = som.winner(X_test[i, :])
        node = node[0] * N + node[1]
        
        if node in X_test_dict:
            X_test_dict[node] = np.vstack((X_test_dict[node], X_test[i, :]))
            y_test_dict[node] = np.vstack((y_test_dict[node], y_train[i, :]))
        else:
            X_test_dict[node] = np.zeros((1, T * n_lag))
            X_test_dict[node][0, :] = X_test[i, :]
            y_test_dict[node] = np.zeros((1, T * n_lag))
            y_test_dict[node][0, :] = y_test[i, :]       
    
    
    return(X_train_dict, y_train_dict, X_test_dict, y_test_dict)
    