import psycopg2
import pandas as pd

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


if __name__ == "__main__":
    userID = getUserID()