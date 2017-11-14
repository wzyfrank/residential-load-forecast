import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import predict_util
import readData
import clustering


def RNN(x, weights, biases, num_hidden, timesteps):

    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, timesteps, n_input)
    # Required shape: 'timesteps' tensors list of shape (batch_size, n_input)

    # Unstack to get a list of 'timesteps' tensors of shape (batch_size, n_input)
    x = tf.unstack(x, timesteps, 1)
    
    # Define a lstm cell with tensorflow
    lstm_cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0, reuse=None)
    stacked_lstm = tf.contrib.rnn.MultiRNNCell( [lstm_cell for _ in range(1)] )
    
    # Get lstm cell output
    outputs, states = rnn.static_rnn(stacked_lstm, x, dtype=tf.float32)
    #outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)
    
    outputs = tf.reshape(outputs, [-1, timesteps*num_hidden])
    
    # Linear activation, using rnn inner loop last output
    return tf.matmul(outputs, weights) + biases


def RNN_LSTM(n_lag, T, X_train, y_train, X_test, y_test, maxLoad, minLoad):    
    num_hidden = 1 # hidden layer num of features
    num_input = 1 # number of inputs in each RNN cell
    n_train = X_train.shape[0] # number of training samples
    
    timesteps = T * n_lag # timesteps
    
    
    # tf Graph input
    X = tf.placeholder("float", [None, timesteps, num_input])
    Y = tf.placeholder("float", [None, T])
    
    # Define weights
    weights = tf.Variable(2* tf.random_normal([timesteps*num_hidden, T]))
    biases = tf.Variable(tf.random_normal([T]) )
     
    ###############################################################################
    # add the RNN network
    prediction = RNN(X, weights, biases, num_hidden, timesteps)

    # Define loss and optimizer
    loss = T * tf.reduce_mean(tf.square(Y - prediction) )  
    loss += 1e-3 * ( tf.nn.l2_loss(weights) + tf.nn.l2_loss(biases) )
    
    train_op = tf.train.AdamOptimizer(learning_rate = 0.1).minimize(loss)
    #train_op = tf.train.AdagradOptimizer(learning_rate = 0.1).minimize(loss)
    # Initialize the variables (i.e. assign their default value)
    init = tf.global_variables_initializer()
    # Start training
    sess = tf.Session()
    # Run the initializer
    sess.run(init)
        
    # Training Parameters
    training_steps = 10000 # maximum step of training 
    display_step = 100 # display loss function 
    last_loss = 10000.0 # init the loss 
    epsilon = 1e-4 # stopping criterion
    step = 0 # training step
    

    while(step < training_steps):
        X_train = X_train.reshape((n_train, timesteps, num_input))
        # Run optimization op (backprop)
        (t_step, l) = sess.run([train_op, loss], feed_dict={X: X_train, Y: y_train})
        
        if((step+1) % display_step == 0):
            print('iteration number %d, loss is %2f' % (step+1, l))
        if(abs(last_loss - l) < epsilon ):
            print('training stopped at: iteration number %d, loss is %2f' % (step+1, l))
            break
        else:
            last_loss = l
            step += 1
            
    # predict and compare with test output
    test_days = X_test.shape[0]
    MAPE_sum = 0
    RMSPE_sum = 0    
    for d in range(test_days):
        X_test_d = np.zeros((1, n_lag * T))
        X_test_d[0,:] = X_test[d,:]
        y_test_d = y_test[d,:]
        X_test_d = X_test_d.reshape((1, timesteps, num_input))
        y_pred = prediction.eval(session = sess, feed_dict={X: X_test_d})
        y_pred = y_pred * (maxLoad - minLoad) + minLoad
        y_test_n = y_test_d * (maxLoad - minLoad) + minLoad
        
        # error metrics
        mape = predict_util.calMAPE(y_test_n, y_pred)
        rmspe = predict_util.calRMSPE(y_test_n, y_pred)
        print('MAPE: %.2f, RMSPE: %.2f' % (mape, rmspe))
        MAPE_sum += mape
        RMSPE_sum += rmspe
        
        # plot forecast with result
        xaxis = range(T)
        plt.step(xaxis, y_pred.flatten(), 'r')
        plt.step(xaxis, y_test_n.flatten(), 'g')
        red_patch = mpatches.Patch(color='red', label='prediction')
        green_patch = mpatches.Patch(color='green', label='actual')
        plt.legend(handles=[red_patch, green_patch])
        plt.show()  
        
    
    tf.reset_default_graph()
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
    (MAPE0, RMSPE0, days0) = RNN_LSTM(n_lag, T, X_train0, y_train0, X_test0, y_test0, maxLoad, minLoad)
    print('forecast result group 0 : MAPE: %.2f, RMSPE: %.2f' % (MAPE0, RMSPE0))
    
    print("start NN forecast on group 1")
    (MAPE1, RMSPE1, days1) = RNN_LSTM(n_lag, T, X_train1, y_train1, X_test1, y_test1, maxLoad, minLoad)
    print('forecast result group 1 : MAPE: %.2f, RMSPE: %.2f' % (MAPE1, RMSPE1))
    
    print("start NN forecast on group 2")
    (MAPE2, RMSPE2, days2) = RNN_LSTM(n_lag, T, X_train2, y_train2, X_test2, y_test2, maxLoad, minLoad)
    print('forecast result group 2 : MAPE: %.2f, RMSPE: %.2f' % (MAPE2, RMSPE2))
    
    
    # overall average
    MAPE = (MAPE0 * days0 + MAPE1 * days1 + MAPE2 * days2) / (days0 + days1 + days2)
    RMSPE = (RMSPE0 * days0 + RMSPE1 * days1 + RMSPE2 * days2) / (days0 + days1 + days2)
    print(' ')
    print('overall forecast result MAPE: %.2f, RMSPE: %.2f' % (MAPE, RMSPE))