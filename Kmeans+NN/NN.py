import pandas as pd
import tensorflow as tf
import numpy as np
import readData
import predict_util
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import clustering


#### add one neural network layer ####
def add_layer(inputs, in_size, out_size, activation_function=None):
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return (outputs, Weights, biases)


#### neural network forecast
def NN_forecast(n_lag, T, X_train, y_train, X_test, y_test, maxLoad, minLoad):
    ############################ Iteration Parameter ##########################
    # maximum iteration
    Max_iter = 20000
    # stopping criteria
    epsilon = 1e-4
    last_l = 10000
        
    ############################ TensorFlow ###################################    
    # place holders
    xs = tf.placeholder(tf.float32, [None, T * n_lag])
    ys = tf.placeholder(tf.float32, [None, T])
    
    N_neuron = 50
    # hidden layers
    (l1, w1, b1) = add_layer(xs, T * n_lag, N_neuron, activation_function=tf.nn.relu)
    (l2, w2, b2) = add_layer(l1, N_neuron, N_neuron, activation_function=tf.nn.tanh)
    
    # output layer
    (prediction, wo, bo) = add_layer(l2, N_neuron, T, None)
    
    # loss function, RMSPE
    #loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), 1))  
    loss = T * tf.reduce_mean(tf.square(ys - prediction) )  
    
    loss += 1e-1 * ( tf.nn.l2_loss(w1) + tf.nn.l2_loss(b1) + tf.nn.l2_loss(wo) + tf.nn.l2_loss(bo) )
    loss += 1e-1 * ( tf.nn.l2_loss(w2) + tf.nn.l2_loss(b2) )
    
    # training step
    train_step = tf.train.AdamOptimizer().minimize(loss)
    
    init = tf.global_variables_initializer()
    # run
    sess = tf.Session()
    # init.
    
    training_size = 50
    MAPE_sum = 0
    RMSPE_sum = 0
    test_days = X_test.shape[0]
    
    for d in range(test_days):
        sess.run(init)  
        ################## training #########################################
        i = 0
        while (i < Max_iter):
            # training
            (t_step, l) = sess.run([train_step, loss], feed_dict={xs: X_train, ys: y_train})
            if(abs(last_l - l) < epsilon):
                break
            else:
                last_l = l
                i = i+1
                
        ################## prediction #########################################
        X_test_d = np.zeros((1, n_lag * T))
        X_test_d[0,:] = X_test[d,:]
        y_test_d = y_test[d,:]
        y_pred = prediction.eval(session = sess, feed_dict={xs: X_test_d})
        y_pred = y_pred * (maxLoad - minLoad) + minLoad
        y_test_n = y_test_d * (maxLoad - minLoad) + minLoad
        '''
        xaxis = range(T)
        plt.step(xaxis, y_pred.flatten(), 'r')
        plt.step(xaxis, y_test_d.flatten(), 'g')
        plt.show()
        '''

        mape = predict_util.calMAPE(y_test_n, y_pred)
        rmspe = predict_util.calRMSPE(y_test_n, y_pred)
        
        # update error metric results
        print('MAPE: %.2f, RMSPE: %.2f' % (mape, rmspe))
        MAPE_sum += mape
        RMSPE_sum += rmspe
        
        # update training set
        X_train = np.concatenate((X_train, X_test_d), axis = 0)
        X_train = X_train[-training_size:, :]
        y_train = np.vstack([y_train, y_test_d])
        y_train = X_train[-training_size:, :]        
        
    # close session
    tf.reset_default_graph() # reset the graph 
    sess.close() 
    
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
    (MAPE0, RMSPE0, days0) = NN_forecast(n_lag, T, X_train0, y_train0, X_test0, y_test0, maxLoad, minLoad)
    print('forecast result group 0 : MAPE: %.2f, RMSPE: %.2f' % (MAPE0, RMSPE0))
    
    print("start NN forecast on group 1")
    (MAPE1, RMSPE1, days1) = NN_forecast(n_lag, T, X_train1, y_train1, X_test1, y_test1, maxLoad, minLoad)
    print('forecast result group 1 : MAPE: %.2f, RMSPE: %.2f' % (MAPE1, RMSPE1))
    
    print("start NN forecast on group 2")
    (MAPE2, RMSPE2, days2) = NN_forecast(n_lag, T, X_train2, y_train2, X_test2, y_test2, maxLoad, minLoad)
    print('forecast result group 2 : MAPE: %.2f, RMSPE: %.2f' % (MAPE2, RMSPE2))
    
    
    # overall average
    MAPE = (MAPE0 * days0 + MAPE1 * days1 + MAPE2 * days2) / (days0 + days1 + days2)
    RMSPE = (RMSPE0 * days0 + RMSPE1 * days1 + RMSPE2 * days2) / (days0 + days1 + days2)
    print(' ')
    print('overall forecast result MAPE: %.2f, RMSPE: %.2f' % (MAPE, RMSPE))
    
    