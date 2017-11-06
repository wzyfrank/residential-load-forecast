from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import predict_util
import changeInterval
import load_weather_corr
import cycles
import readData

def genData(load_weekday, n_train, n_valid, n_lag, T, curr_day):
    max_load = np.max(load_weekday)
    min_load = np.min(load_weekday)
    load_weekday = (load_weekday - min_load) / (max_load - min_load)
    
    ################## generate data ##########################################
    y_train = np.zeros((n_train, T))
    X_train = np.zeros((n_train, T * n_lag))
    
    y_valid = np.zeros((n_valid, T))
    X_valid = np.zeros((n_valid, T * n_lag))
    # training data
    row = 0
    for train_day in range(curr_day - n_train - n_valid, curr_day - n_valid):
        y_train[row,:] = load_weekday[train_day * T : train_day * T + T]
        X_train[row,0*T*n_lag:1*T*n_lag] = load_weekday[train_day * T - n_lag * T: train_day * T]
        row += 1
    
    # validation data
    row = 0
    for valid_day in range(curr_day - n_valid, curr_day):
        y_valid[row,:] = load_weekday[valid_day * T : valid_day * T + T]
        X_valid[row,0*T*n_lag:1*T*n_lag] = load_weekday[valid_day * T - n_lag * T: valid_day * T]
        row += 1    
        
    # test data
    X_test = np.zeros((1, T * n_lag))
    X_test[0, 0*T*n_lag:1*T*n_lag] = load_weekday[curr_day*T - n_lag*T: curr_day*T]
    y_test = load_weekday[curr_day*T: curr_day *T + T]
    
    
    return(X_train, y_train, X_valid, y_valid, X_test, y_test, min_load, max_load)
        
    
def RNN(x, weights, biases, num_hidden, timesteps):

    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, timesteps, n_input)
    # Required shape: 'timesteps' tensors list of shape (batch_size, n_input)

    # Unstack to get a list of 'timesteps' tensors of shape (batch_size, n_input)
    x = tf.unstack(x, timesteps, 1)
    
    # Define a lstm cell with tensorflow
    lstm_cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0, reuse=None)
    stacked_lstm = tf.contrib.rnn.MultiRNNCell( [lstm_cell for _ in range(3)] )
    
    # Get lstm cell output
    outputs, states = rnn.static_rnn(stacked_lstm, x, dtype=tf.float32)
    #outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)
    
    outputs = tf.reshape(outputs, [-1, timesteps*num_hidden])
    
    # Linear activation, using rnn inner loop last output
    return tf.matmul(outputs, weights) + biases




def RNN_LSTM(load, curr_day):    
    # Network Parameters
    num_input = 1 # MNIST data input (img shape: 28*28)
    T = 24
    num_hidden = 1 # hidden layer num of features
    n_train = 5
    n_valid = 1
    n_lag = 2
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
    loss += 1e-2 * ( tf.nn.l2_loss(weights) + tf.nn.l2_loss(biases) )
    
    train_op = tf.train.AdamOptimizer(learning_rate = 0.1).minimize(loss)
    #train_op = tf.train.AdagradOptimizer(learning_rate = 0.1).minimize(loss)
    # Initialize the variables (i.e. assign their default value)
    init = tf.global_variables_initializer()
    # Start training
    sess = tf.Session()
    # Run the initializer
    sess.run(init)
    
    # generate the training, validation and test data
    (X_train, y_train, X_valid, y_valid, X_test, y_test, min_load, max_load) = genData(load, n_train, n_valid, n_lag, T, curr_day)
    
    # Training Parameters
    training_steps = 10000 # maximum step of training 
    display_step = 100 # display loss function 
    last_loss = 10000.0 # init the loss 
    epsilon = 1e-5 # stopping criterion
    step = 0 # training step
        
    while(step < training_steps):

        X_train = X_train.reshape((n_train, timesteps, num_input))
        # Run optimization op (backprop)
        sess.run(train_op, feed_dict={X: X_train, Y: y_train})
        
        # Calculate loss on validation set
        X_valid = X_valid.reshape((n_valid, timesteps, num_input))
        l = sess.run(loss, feed_dict={X: X_valid, Y: y_valid})
        
        if((step+1) % display_step == 0):
            print('iteration number %d, loss is %2f' % (step+1, l))
        if(abs(last_loss - l) < epsilon ):
            print('training stopped at: iteration number %d, loss is %2f' % (step+1, l))
            break
        else:
            last_loss = l
            step += 1
            
    # predict and compare with test output
    X_test = X_test.reshape((1, timesteps, num_input))
    y_pred = prediction.eval(session = sess, feed_dict={X: X_test})
    y_pred = y_pred * (max_load - min_load) + min_load
    y_test = y_test * (max_load - min_load) + min_load
    
    # error metrics
    mape = predict_util.calMAPE(y_test, y_pred)
    rmspe = predict_util.calRMSPE(y_test, y_pred)
    
    # plot forecast with result
    xaxis = range(T)
    plt.step(xaxis, y_pred.flatten(), 'r')
    plt.step(xaxis, y_test.flatten(), 'g')
    red_patch = mpatches.Patch(color='red', label='prediction')
    green_patch = mpatches.Patch(color='green', label='actual')
    plt.legend(handles=[red_patch, green_patch])
    plt.show()  
    
    print('MAPE: %.2f, RMSPE: %.2f' % (mape, rmspe))
    
    tf.reset_default_graph()
    sess.close()  
    return (mape, rmspe)




if __name__ == "__main__":   
    # import load data
    data = readData.loadResidentialData()
    sumLoad = np.zeros((35040,))
    #userLoad = readData.getUserData(data, 0)
    for i in range(144):
        sumLoad += readData.getUserData(data, i)
    
    # import weather data
    (temperature, humidity, pressure) = load_weather_corr.getWeatherFeature()
    
    # import cycles data
    (dailycycle, weeklycycle) = cycles.getCycles()
    
    sumLoad = changeInterval.From15minTo1hour(sumLoad)
    
    MAPE_sum = 0.0
    RMSPR_sum = 0.0
    
    for curr_day in range(56, 364):
        print(curr_day)
        (mape, rmspe) = RNN_LSTM(sumLoad, curr_day)
        MAPE_sum += mape
        RMSPR_sum += rmspe
    
    days_sample = 365 - 1 - 56
    MAPE_sum = MAPE_sum / days_sample
    RMSPR_sum = RMSPR_sum / days_sample
    print('AVERAGE MAPE: %.2f, RMSPE: %.2f' % (MAPE_sum, RMSPR_sum))