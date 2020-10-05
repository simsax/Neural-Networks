# cd Desktop\RETI NEURALI E ALG GENETICI\Cazzeggio\1) xor problem
import sys
sys.path.insert(1, '/Users/Sax/Desktop/RETI NEURALI E ALG GENETICI/Cazzeggio/toy_nn')
from nn import *
import random

if __name__ == '__main__':

    training_data = [{
            "inputs":[0,1], "targets":[1]
        },
         {
            "inputs":[1,0], "targets":[1]
        },
         {
            "inputs":[1,1], "targets":[0]
        },
         {
            "inputs":[0,0], "targets":[0]
        }]
    
    nn = NeuralNetwork(2, 2, 1)
    for i in range(0,100000):
        data = random.choice(training_data)
        nn.train(data["inputs"], data["targets"])

    guess1 = nn.feedforward([1,0])
    guess2 = nn.feedforward([0,1])
    guess3 = nn.feedforward([0,0])
    guess4 = nn.feedforward([1,1])
    print(f"inputs [1,0] -> guessed {guess1}")
    print(f"inputs [0,1] -> guessed {guess2}")
    print(f"inputs [0,0] -> guessed {guess3}")
    print(f"inputs [1,1] -> guessed {guess4}")