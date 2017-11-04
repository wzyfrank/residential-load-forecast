import psycopg2
import pandas as pd
import numpy as np
import datetime
import time

def getUserID():
    conn = psycopg2.connect(dbname = "postgres", user = "j8mKmsbNAGqx", password = "zq7Hwitibx4O", host = "dataport.cloud", port = 5434)
    
    query = "SELECT dataid FROM university.electricity_egauge_15min GROUP BY dataid"
    query_begin = "SELECT dataid FROM university.electricity_egauge_15min WHERE local_15min BETWEEN '01-01-2013' AND '01-02-2013' GROUP BY dataid"
    query_end = "SELECT dataid FROM university.electricity_egauge_15min WHERE local_15min BETWEEN '01-01-2017' AND '01-02-2017' GROUP BY dataid"

    df = pd.read_sql_query(query, conn)
    
    df_b = pd.read_sql_query(query_begin, conn)
    df_e = pd.read_sql_query(query_end, conn)
    
    userID = list(df['dataid'])
    
    user_b = list(df_b['dataid'])
    user_e = list(df_e['dataid'])
    userID = list(set(user_b).intersection(user_e))
    userID.sort()
    
    conn.close()
    
    return userID


def getUserLoadbyID(_ID):
    conn = psycopg2.connect(dbname = "postgres", user = "j8mKmsbNAGqx", password = "zq7Hwitibx4O", host = "dataport.cloud", port = 5434)
    query = "SELECT local_15min, use FROM university.electricity_egauge_15min WHERE dataid=" + _ID + " AND local_15min BETWEEN '01-01-2013' AND '01-01-2017'"
    df = pd.read_sql_query(query, conn)
    
    load_map = dict()
    
    N_lines = len(df.index)
    for i in range(N_lines):
        timestamp = df.iloc[i, 0]
        t = int(time.mktime(timestamp.timetuple()))
        load_map[t] = df.iloc[i, 1]
    
    
    conn.close()

    return load_map


def GenkWLoad(load_map):
    # build the timestamp -> kWh map
    start_time = '2013-01-01 00:00:00'
    end_time = '2017-01-01 00:00:00'
    
    start_t = datetime.datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S')
    start_t = int(time.mktime(start_t.timetuple()))
    end_t = datetime.datetime.strptime(end_time, '%Y-%m-%d %H:%M:%S')
    end_t = int(time.mktime(end_t.timetuple()))
    
    # the timestamp of kW readings
    new_time = np.arange(start_t,end_t,900)
    newload_map = dict()
    
    
    for t in new_time:
        # t in kWh map
        if t in load_map.keys():
            newload_map[t] = load_map[t]          
        elif t - 900 * 96 in newload_map.keys():
            newload_map[t] = newload_map[t-900*96]
        elif t - 900 * 192 in newload_map.keys():
            newload_map[t] = newload_map[t-900*192]
        else:
            newload_map[t] = newload_map[t-900]
        
    loadFilled = np.array(list(newload_map.values()))
    
    return loadFilled
    
    
if __name__ == "__main__":
    
    userID = getUserID()
    
    
    df = pd.DataFrame()
    for ID in userID:
        curID = str(ID)
        print(curID)
        
        load_map = getUserLoadbyID(curID)
        loadFilled = GenkWLoad(load_map)
        
        df[curID] = loadFilled
    
    
    df.to_csv('residential_load.csv', sep=',', index = True)
    
