import pandas as pd

def loadResidentialData():
    #data = pd.read_csv('F:/OneDrive/Load Forecast/residential/data/residential_load_20132014.csv')
    data = pd.read_csv('F:/SkyDrive/Load Forecast/residential/data/residential_load_20132014.csv')
    data = data.as_matrix()
    return data


def getUserData(data, column):
    user_load = data[:, column]
    return user_load


if __name__ == "__main__":
    data = loadResidentialData()
    
    userNo = 10
    user_load = getUserData(data, userNo)