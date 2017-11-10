import numpy as np
import pandas as pd
import readData
import changeInterval
import scipy.stats as sp

def normalize(input_data):
    return (input_data - np.min(input_data)) / (np.max(input_data) - np.min(input_data))
    

def getWeatherFeature():
    data = readData.loadResidentialData()
    
    sumLoad = np.zeros((35040,))
    #userLoad = readData.getUserData(data, 0)
    for i in range(144):
        sumLoad += readData.getUserData(data, i)
        
    load_1hr = changeInterval.From15minTo1hour(sumLoad)
    
    # load weather data
    #weather_data = pd.read_csv('F:/OneDrive/Load Forecast/residential/data/weather2013_Austin.csv')
    weather_data = pd.read_csv('F:/SkyDrive/Load Forecast/residential/data/weather2013_Austin.csv')
    # test correlations
    corr_temp = sp.pearsonr(load_1hr, np.array(weather_data['temperature']))[0]
    corr_humidity = sp.pearsonr(load_1hr, np.array(weather_data['humidity']))[0]
    corr_pressure = sp.pearsonr(load_1hr, np.array(weather_data['pressure']))[0]
    corr_precip = sp.pearsonr(load_1hr, np.array(weather_data['precip_intensity']))[0]
    
    print(corr_temp, corr_humidity, corr_pressure, corr_precip)
    
    # selected features: temperature, humidity, pressure
    temperature = np.array(weather_data['temperature'])
    temperature = normalize(temperature)
    
    humidity = np.array(weather_data['humidity'])
    humidity = normalize(humidity)
    
    pressure = np.array(weather_data['pressure'])
    pressure = normalize(pressure)
    
    return(temperature, humidity, pressure)
