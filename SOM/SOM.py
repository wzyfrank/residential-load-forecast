from minisom import MiniSom 

# test SOM package
data = [[ 0.80,  0.55,  0.22,  0.03],
        [ 0.82,  0.50,  0.23,  0.03],
        [ 0.80,  0.54,  0.22,  0.03],
        [ 0.80,  0.53,  0.26,  0.03],
        [ 0.79,  0.56,  0.22,  0.03],
        [ 0.75,  0.60,  0.25,  0.03],
        [ 0.77,  0.59,  0.22,  0.03]]  


som = MiniSom(6, 6, 4, sigma=0.3, learning_rate=0.5) # initialization of 6x6 SOM
print("Training...")
som.train_random(data, 100) # trains the SOM with 100 iterations
print("...ready!")

x = [0.4, 0.8, 0.0, 1.0]
print(som.winner(x))