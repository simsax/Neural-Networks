import numpy as np
import matplotlib.pyplot as plt
import sys
import tkinter as tk
from PIL import Image, ImageDraw, ImageGrab
sys.path.insert(1, '/Users/Sax/Desktop/RETI NEURALI E ALG GENETICI/Cazzeggio/toy_nn')
from matrix import *
from nn import *

TOTAL_DATA = 1000 
TRAINING_DATA = 800 # numero di immagini (array) per il training
TESTING_DATA = TOTAL_DATA-TRAINING_DATA # numero di immagini per il test

CAT = 0
RAINBOW = 1
TRAIN = 2

epochCounter = 0

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

def trainButton(training):
    global epochCounter
    trainEpoch(training)
    epochCounter += 1
    print(f"Epoch: {epochCounter}")

def testButton(testing):
    percentCorrect = testAll(testing)
    print(f"Correct guesses: {percentCorrect*100:.2f}%")

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
        if indexGuess == label:
            correct+=1
    percent = correct/len(testing)
    return percent

def draw(event):
    x1, y1 = (event.x-1),(event.y-1)
    x2, y2 = (event.x+1),(event.y+1)
    c.create_oval(x1,y1,x2,y2,width=12, fill="black")

def guessButton(c, root):
    x2=200
    y2=274
    x1=x2+348
    y1=y2+348
    ImageGrab.grab().crop((x2,y2,x1,y1)).save("./test.jpg")
    img = Image.open("test.jpg")
    img = img.resize((28,28))
    img = img.convert("L")
    pixels = list(img.getdata())
    for i in range(0,784):
        pixels[i] = (255 - pixels[i])/255.0
    guess = nn.feedforward(pixels)
    indexGuess = np.argmax(guess)
    if indexGuess == CAT:
        print("Cat")
    elif indexGuess == TRAIN:
        print("Train")
    else:
        print("Rainbow")

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
    
    # for i in range(1,11):
    #     trainEpoch(training)
    #     percentCorrect = testAll(testing)
    #     print(f"Epoch: {i}\nCorrect guesses: {percentCorrect*100:.2f}%")

    root = tk.Tk()
    root.geometry('280x316+150+150')
    root.title("Doodle classifier")
    c = tk.Canvas(root, height=280, width=280, bg="white")
    c.place(x=0,y=36)
    c.bind("<B1-Motion>", draw)
    trainB = tk.Button(root, text="Train", command=lambda: trainButton(training))
    trainB.place(height=35, width=70, x=0, y=0)
    testB = tk.Button(root, text="Test", command=lambda: testButton(testing))
    testB.place(height=35, width=70, x=71, y=0)
    clearB = tk.Button(root, text="Clear", command=lambda: c.delete("all"))
    clearB.place(height=35, width=70, x=141, y=0)
    clearB = tk.Button(root, text="Guess", command=lambda: guessButton(c,root)) 
    clearB.place(height=35, width=70, x=211, y=0)
    root.mainloop()

    #stampo i primi 100 disegni di gatto in una griglia 10x10 
    # for i in range(0,100):
    #     plt.subplot(10,10,i+1)
    #     cat_temp = Matrix.createMatrixFromArray(cats["training"][i]["arr"],28,28)
    #     invertGrey(cat_temp)
    #     plt.imshow(cat_temp.data, cmap="gray")
    #     plt.axis('off')
    # plt.show()
    
    
 
