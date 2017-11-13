import numpy as np
#### time interval 1h ####
def genDailyCycle():
    d = np.array(range(24))
    d = np.tile(d, 365)
    ratio = 24 / (2 * np.pi)
    d = d / ratio
    d = np.cos(d)
    return d

def genWeeklyCycle():
    d = np.array(range(24))
    w = np.array(range(168))
    w = np.tile(w, 52)
    w = np.concatenate((w, d), axis = 0)
    ratio = 168 / (2 * np.pi)
    w = w / ratio
    w = np.cos(w)
    return w

def getCycles():
    d = genDailyCycle()
    w = genWeeklyCycle()
    return (d, w)

#### time interval 15min ####
def genDailyCycle_15min():
    d = np.array(range(96))
    d = np.tile(d, 365)
    ratio = 96 / (2 * np.pi)
    d = d / ratio
    d = np.cos(d)
    return d

def genWeeklyCycle_15min():
    d = np.array(range(96))
    w = np.array(range(672))
    w = np.tile(w, 52)
    w = np.concatenate((w, d), axis = 0)
    ratio = 672 / (2 * np.pi)
    w = w / ratio
    w = np.cos(w)
    return w

def getCycles_15min():
    d = genDailyCycle_15min()
    w = genWeeklyCycle_15min()
    return (d, w)

if __name__ == "__main__":
    d = genDailyCycle()
    w = genWeeklyCycle()