# cd Desktop\RETI NEURALI E ALG GENETICI\Cazzeggio\2) doodle classifier
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(1, '/Users/Sax/Desktop/RETI NEURALI E ALG GENETICI/Cazzeggio/toy_nn')
from matrix import *
from nn import *

TOTAL_DATA = 1000 
TRAINING_DATA = 800 # numero di immagini (array) per il training
TESTING_DATA = TOTAL_DATA-TRAINING_DATA # numero di immagini per il test

CAT = 0
RAINBOW = 1
TRAIN = 2

#oggetto Matrix in input
def invertGrey(m):
    for i in range(0, m.rows):
        for j in range(0, m.cols):
            m.data[i][j] = 255 - m.data[i][j]

def prepareData(category, data, label):
    for i in range(0,TOTAL_DATA):
        if i < TRAINING_DATA:
            category["training"][i]["arr"] = np.empty(784)
            category["training"][i]["arr"] = data[i]
            category["training"][i]["label"] = label
        else:
            category["testing"][i-TRAINING_DATA]["arr"] = np.empty(784)
            category["testing"][i-TRAINING_DATA]["arr"] = data[i]
            category["testing"][i-TRAINING_DATA]["label"] = label
    return category

def trainEpoch(training):
    np.random.shuffle(training)
    for i in range(0, len(training)):
        inputs = np.empty(784)
        data = training[i]["arr"]
        label = training[i]["label"]
        for j in range(0, 784):
            inputs[j] = data[j] / 255.0 # normalizzo gli input per avere valori floating point tra 0 e 1
        targets = [0, 0, 0]
        targets[label] = 1
        nn.train(inputs, targets)

def testAll(testing):
    correct = 0
    for i in range(0, len(testing)):
        inputs = np.empty(784)
        data = testing[i]["arr"]
        label = testing[i]["label"]
        for j in range(0, 784):
            inputs[j] = data[j] / 255.0 # normalizzo gli input per avere valori floating point tra 0 e 1
        guess = nn.feedforward(inputs)
        indexGuess = np.argmax(guess)
        print(guess)
        # print(indexGuess)
        # print(label)

        if indexGuess == label:
            correct+=1
    percent = correct/len(testing)
    return percent

if __name__ == '__main__':
    rainbow_data = np.load("rainbow.npy") #ogni elemento di rainbow_data corrisponde a un disegno
    cat_data = np.load("cat.npy")
    train_data = np.load("train.npy")

    rainbows = {"training":[{"arr":[],"label":None} for i in range(TRAINING_DATA)], "testing":[{"arr":[],"label":None} for i in range(TESTING_DATA)]}
    cats = {"training":[{"arr":[],"label":None} for i in range(TRAINING_DATA)], "testing":[{"arr":[],"label":None} for i in range(TESTING_DATA)]}
    trains = {"training":[{"arr":[],"label":None} for i in range(TRAINING_DATA)], "testing":[{"arr":[],"label":None} for i in range(TESTING_DATA)]}

    rainbows = prepareData(rainbows, rainbow_data, RAINBOW) #dizionario con array training e testing
    cats = prepareData(cats, cat_data, CAT)
    trains = prepareData(trains, train_data, TRAIN)

    nn = NeuralNetwork(784, 64, 3)

    #metto tutti gli array in un unico array training e li mischio
    training = np.concatenate((rainbows["training"], cats["training"], trains["training"]), axis=None) #2400 elementi
    testing = np.concatenate((rainbows["testing"], cats["testing"], trains["testing"]), axis=None) #600 elementi
    np.random.shuffle(testing)
    
    for i in range(1,11):
        trainEpoch(training)
        percentCorrect = testAll(testing)
        print(f"Epoch: {i}\nCorrect guesses: {percentCorrect*100:.2f}%")
    
    #cercare se ci sono modi di preservare l'allenamento della rete una volta spento il programma. richiederebbe probabilmente di salvare tutti i pesi su disco

    #stampo i primi 100 disegni di gatto in una griglia 10x10 
    # for i in range(0,100):
    #     plt.subplot(10,10,i+1)
    #     cat_temp = Matrix.createMatrixFromArray(cats["training"][i]["arr"],28,28)
    #     invertGrey(cat_temp)
    #     plt.imshow(cat_temp.data, cmap="gray")
    #     plt.axis('off')
    # plt.show()
    
    
 