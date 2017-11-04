import psycopg2
import pandas as pd
import datetime
import time
import pytz

def utc_to_local(utc_dt):
    local_dt = utc_dt.replace(tzinfo=pytz.utc).astimezone(local_tz)
    return local_tz.normalize(local_dt) # .normalize might be unnecessary

def getWeather():
    # connection to DB
    conn = psycopg2.connect(dbname = "postgres", user = "j8mKmsbNAGqx", password = "zq7Hwitibx4O", host = "dataport.cloud", port = 5434)
    query = "SELECT * FROM university.weather WHERE localhour BETWEEN '01-01-2013' AND '01-01-2014' limit 8760"
    
    df = pd.read_sql_query(query, conn)
    
    return df

if __name__ == "__main__":
    
    local_tz = pytz.timezone('US/Central')
    
    start_time = '2013-01-01 06:00:00'
    end_time = '2014-01-01 06:00:00'
    start_t = datetime.datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S')
    end_t = datetime.datetime.strptime(end_time, '%Y-%m-%d %H:%M:%S')
    
    start_t = utc_to_local(start_t)
    
    weatherData = getWeather()
    weatherData.to_csv('weather2013_Austin.csv', sep=',', index = True)
    