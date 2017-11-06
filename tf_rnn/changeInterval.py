import numpy as np
import readData

def From15minTo1hour(load_input):
    length = int(load_input.size / 4)
    load_out = load_input.reshape(length, 4)
    load_out = np.average(load_out, axis = 1)
    return load_out

'''
def From1hrTo15min(load_input):
    load_out = np.matlib.repmat(load_input, 4)
''' 

if __name__ == "__main__":
    data = readData.loadResidentialData()
    
    sumLoad = np.zeros((35040,))
    #userLoad = readData.getUserData(data, 0)
    for i in range(144):
        sumLoad += readData.getUserData(data, i)
        
    load_1hr = From15minTo1hour(sumLoad)