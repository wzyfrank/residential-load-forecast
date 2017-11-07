import numpy as np

def genDailyCycle():
    d = np.array(range(24))
    d = np.tile(d, 365)
    d = d / (2 * np.pi)
    d = np.cos(d)
    return d

def genWeeklyCycle():
    d = np.array(range(24))
    w = np.array(range(168))
    w = np.tile(w, 52)
    w = np.concatenate((w, d), axis = 0)
    w = w / (2 * np.pi)
    w = np.cos(w)
    return w

def getCycles():
    d = genDailyCycle()
    w = genWeeklyCycle()
    return (d, w)


if __name__ == "__main__":
    d = genDailyCycle()
    w = genWeeklyCycle()