import tensorflow as tf
import numpy as np
import readData
import predict_util
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import changeInterval
import load_weather_corr

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
def NN_forecast_weather(load_weekday, n_train, n_lag, T, temperature, humidity, pressure):
    ############################ Iteration Parameter ##########################
    # maximum iteration
    Max_iter = 20000
    # stopping criteria
    epsilon = 1e-5
    last_l = 10000
    # set of features
    set_fea = 4
    # number of neurons in hidden layers
    N_neuron = 50
    
    ############################ TensorFlow ###################################    
    # place holders
    xs = tf.placeholder(tf.float32, [None, T * n_lag * set_fea])
    ys = tf.placeholder(tf.float32, [None, T])
    
    
    # hidden layers
    (l1, w1, b1) = add_layer(xs, T * n_lag * set_fea, N_neuron, activation_function=tf.nn.relu)
    (l2, w2, b2) = add_layer(l1, N_neuron, N_neuron, activation_function=tf.nn.tanh)
    
    # output layer
    (prediction, wo, bo) = add_layer(l2, N_neuron, T, None)
    
    # loss function, RMSPE
    #loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), 1))  
    loss = T * tf.reduce_mean(tf.square(ys - prediction) )  
    
    loss += 1e-2 * ( tf.nn.l2_loss(w1) + tf.nn.l2_loss(b1) + tf.nn.l2_loss(wo) + tf.nn.l2_loss(bo) )
    loss += 1e-2 * ( tf.nn.l2_loss(w2) + tf.nn.l2_loss(b2) )
    
    # training step
    train_step = tf.train.AdamOptimizer().minimize(loss)
    
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
  
    
        y_train = np.zeros((n_train, T))
        X_train = np.zeros((n_train, T * n_lag * set_fea))
        row = 0
        for train_day in range(curr_day - n_train, curr_day):
            y_train[row,:] = load_weekday[train_day * T : train_day * T + T]
            X_train[row,0*T*n_lag:1*T*n_lag] = load_weekday[train_day * T - n_lag * T: train_day * T]
            X_train[row,1*T*n_lag:2*T*n_lag] = temperature[train_day * T - n_lag * T: train_day * T]
            X_train[row,2*T*n_lag:3*T*n_lag] = humidity[train_day * T - n_lag * T: train_day * T]
            X_train[row,3*T*n_lag:4*T*n_lag] = pressure[train_day * T - n_lag * T: train_day * T]
            
            row += 1
        max_load = np.max(X_train[:, 0*T*n_lag:1*T*n_lag])
        min_load = np.min(X_train[:, 0*T*n_lag:1*T*n_lag])    
		
        # building test data
        X_test = np.zeros((1, T * n_lag * set_fea))
        X_test[0, 0*T*n_lag:1*T*n_lag] = load_weekday[curr_day*T - n_lag*T: curr_day*T]
        X_test[0, 1*T*n_lag:2*T*n_lag] = temperature[curr_day*T - n_lag*T: curr_day*T]
        X_test[0, 2*T*n_lag:3*T*n_lag] = humidity[curr_day*T - n_lag*T: curr_day*T]
        X_test[0, 3*T*n_lag:4*T*n_lag] = pressure[curr_day*T - n_lag*T: curr_day*T]
        
        y_test = load_weekday[curr_day*T: curr_day *T + T]
        
        X_train[:, 0*T*n_lag:1*T*n_lag] = (X_train[:, 0*T*n_lag:1*T*n_lag]-min_load) / (max_load - min_load)
        y_train = (y_train-min_load) / (max_load - min_load)
        X_test[:, 0*T*n_lag:1*T*n_lag] = (X_test[:, 0*T*n_lag:1*T*n_lag]-min_load) / (max_load - min_load)
                

        # training 
        i = 0
        while (i < Max_iter):
            # training
            (t_step, l) = sess.run([train_step, loss], feed_dict={xs: X_train, ys: y_train})
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
    n_train = 50
    # number of lags
    n_lag = 1
    # time intervals per day
    T= 24
   
    # import load data
    data = readData.loadResidentialData()
    sumLoad = np.zeros((35040,))
    #userLoad = readData.getUserData(data, 0)
    for i in range(144):
        sumLoad += readData.getUserData(data, i)
    
    # import weather data
    (temperature, humidity, pressure) = load_weather_corr.getWeatherFeature()
    
    
    sumLoad = changeInterval.From15minTo1hour(sumLoad)
    # call neural network forecast
    (MAPE_avg, RMSPE_avg) = NN_forecast_weather(sumLoad, n_train, n_lag, T, temperature, humidity, pressure)
    print('forecast result MAPE: %.2f, RMSPE: %.2f' % (MAPE_avg, RMSPE_avg))