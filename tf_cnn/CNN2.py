import pandas as pd
import tensorflow as tf
import numpy as np
import readData
import predict_util
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches



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
def NN_forecast(load_weekday, n_train, n_lag, T):
    ############################ Iteration Parameter ##########################
    # maximum iteration
    Max_iter = 20000
    # stopping criteria
    epsilon = 1e-4
    last_l = 100000
    display_step = 100
    
    ############################ TensorFlow ###################################    
    # place holders
    xs = tf.placeholder(tf.float32, [None, T * n_lag])
    ys = tf.placeholder(tf.float32, [None, T])
    
    input_layer = tf.reshape(xs, [-1, T, n_lag, 1])
        
    # Convolutional Layer #1
    # Computes 32 features using a 5x5 filter with ReLU activation.
    # Padding is added to preserve width and height.
    # Input Tensor Shape: [batch_size, T, n_lag, 1]
    # Output Tensor Shape: [batch_size, T, n_lag, 32]
    conv1 = tf.layers.conv2d(
          inputs=input_layer,
          filters=64,
          kernel_size=[4, 4],
          padding="same",
          activation=tf.nn.relu)
    
    # Pooling Layer #1
    # First max pooling layer with a 2x2 filter and stride of 2
    # Input Tensor Shape: [batch_size, T, n_lag, 32]
    # Output Tensor Shape: [batch_size, T/2, n_lag/2, 32]
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
    pool1_flat = tf.reshape(pool1, [-1, T * n_lag * 16])
  
    N_neuron = 100
    # hidden layers
    (l1, w1, b1) = add_layer(pool1_flat, T * n_lag * 16, N_neuron, activation_function=tf.nn.relu)
    #(l2, w2, b2) = add_layer(l1, N_neuron, N_neuron, activation_function=tf.nn.tanh)
    
    # output layer
    (prediction, wo, bo) = add_layer(l1, N_neuron, T, None)
    
    # loss function, RMSPE
    #loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), 1))  
    loss = T * tf.reduce_mean(tf.square(ys - prediction) )  
    loss += 1e-3 * ( tf.nn.l2_loss(w1) + tf.nn.l2_loss(b1) + tf.nn.l2_loss(wo) + tf.nn.l2_loss(bo) )
    #loss += 1e-3 * ( tf.nn.l2_loss(w2) + tf.nn.l2_loss(b2) )
    # training step
    train_step = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss)
    #train_step = tf.train.AdagradOptimizer(learning_rate=1).minimize(loss)
    
    init = tf.global_variables_initializer()
    # run
    sess = tf.Session()
    # init.
    sess.run(init)      
    
    
    n_days = int(load_weekday.size / T)
    ################## generate data ##########################################
    MAPE_sum = 0.0
    RMSPE_sum = 0.0
    
    for curr_day in range(n_train + n_lag, n_days-1):
        # init.
        #sess.run(init) 
    
        y_train = np.zeros((n_train, T))
        X_train = np.zeros((n_train, T * n_lag))
        row = 0
        for train_day in range(curr_day - n_train, curr_day):
            y_train[row,:] = load_weekday[train_day * T : train_day * T + T]
            X_train[row,0*T*n_lag:1*T*n_lag] = load_weekday[train_day * T - n_lag * T: train_day * T]
            row += 1
        max_load = np.max(X_train)
        min_load = np.min(X_train)    
		
        # building test data
        X_test = np.zeros((1, T * n_lag))
        X_test[0, 0*T*n_lag:1*T*n_lag] = load_weekday[curr_day*T - n_lag*T: curr_day*T]
        y_test = load_weekday[curr_day*T: curr_day *T + T]
        
        X_train = (X_train-min_load) / (max_load - min_load)
        y_train = (y_train-min_load) / (max_load - min_load)
        X_test = (X_test-min_load) / (max_load - min_load)
                

        # training 
        i = 0
        while (i < Max_iter):
            # training
            (t_step, l) = sess.run([train_step, loss], feed_dict={xs: X_train, ys: y_train})
            #if((i+1) % display_step == 0):
                #print('iteration number %d, loss is %2f' % (i+1, l))
            if(abs(last_l - l) < epsilon):
                break
            else:
                last_l = l
                i = i+1
            
        
        #y_ = prediction.eval(session = sess, feed_dict={xs: X_train})
        y_pred = prediction.eval(session = sess, feed_dict={xs: X_test})
        y_pred = y_pred * (max_load - min_load) + min_load
        # plot daily forecast
        '''
        xaxis = range(T)
        plt.step(xaxis, y_pred.flatten(), 'r')
        plt.step(xaxis, y_test.flatten(), 'g')
        plt.show()
        '''

        mape = predict_util.calMAPE(y_test, y_pred)
        rmspe = predict_util.calRMSPE(y_test, y_pred)

        # update error metric results
        print('MAPE: %.2f, RMSPE: %.2f' % (mape, rmspe))
        MAPE_sum += mape
        RMSPE_sum += rmspe
        

    # close session
    tf.reset_default_graph() # reset the graph 
    sess.close() 
    
    
    days_sample = n_days - 1 - n_train - n_lag

    return (MAPE_sum / days_sample, RMSPE_sum / days_sample)

    
if __name__ == "__main__":
    # number of days in training set    
    n_train = 40
    # number of lags
    n_lag = 8
    # time intervals per day
    T= 96
   
    # import load data
    data = readData.loadResidentialData()
    sumLoad = np.zeros((35040,))
    #userLoad = readData.getUserData(data, 0)
    for i in range(144):
        sumLoad += readData.getUserData(data, i)
    
    # call neural network forecast
    (MAPE_avg, RMSPE_avg) = NN_forecast(sumLoad, n_train, n_lag, T)
    print('forecast result MAPE: %.2f, RMSPE: %.2f' % (MAPE_avg, RMSPE_avg))