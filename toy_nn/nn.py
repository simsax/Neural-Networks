from matrix import *
import math

def sigmoid(x):
    return 1/(1+math.exp(-x))

def dsigmoid(y):
    #return sigmoid(x)*(1-sigmoid(x))
    return y*(1-y)

class NeuralNetwork:

    def __init__(self, numI, numH, numO, saved):
        self.input_nodes = numI
        self.hidden_nodes = numH
        self.output_nodes = numO
        self.weights_ih = Matrix(self.hidden_nodes, self.input_nodes)
        self.weights_ho = Matrix(self.output_nodes, self.hidden_nodes)
        self.bias_h = Matrix(self.hidden_nodes, 1)
        self.bias_o = Matrix(self.output_nodes, 1)
        if saved == 1:
            self.load()
        else:
            self.weights_ih.randomize()
            self.weights_ho.randomize()
            self.bias_h.randomize()
            self.bias_o.randomize()
        self.learning_rate =  0.1

    def feedforward(self, input_array):
        inputs = Matrix.fromArray(input_array)
        hidden = Matrix.Mul(self.weights_ih,inputs)
        hidden.add(self.bias_h)
        hidden.map(sigmoid) #activation function
        outputs = Matrix.Mul(self.weights_ho,hidden)
        outputs.add(self.bias_o)
        #outputs.map(sigmoid)
        outputs.softmax()
        return outputs.toArray()

    def train(self, input_array, target_array):
        inputs = Matrix.fromArray(input_array)
        hidden = Matrix.Mul(self.weights_ih,inputs)
        hidden.add(self.bias_h)
        hidden.map(sigmoid) #activation function
        outputs = Matrix.Mul(self.weights_ho,hidden)
        outputs.add(self.bias_o)
        #outputs.map(sigmoid)
        outputs.softmax()

        targets = Matrix.fromArray(target_array)
        #calculate the error -> error = targets - outputs
        output_errors = Matrix.subtract(targets, outputs)
        #calculate gradient (lr*E*(O*(1-O)))
        gradients = Matrix.Map(outputs,dsigmoid) #O*(1-O)
        gradients.mul(output_errors) 
        gradients.mul(self.learning_rate)
        #calculate deltas
        hidden_t = Matrix.Transpose(hidden)
        weights_ho_deltas = Matrix.Mul(gradients, hidden_t) 
        self.weights_ho.add(weights_ho_deltas) #adjust the weights by deltas
        self.bias_o.add(gradients) #adjust the bias by its deltas (which is just the gradient)
        #calculate hidden layer errors
        who_t = Matrix.Transpose(self.weights_ho)
        hidden_errors = Matrix.Mul(who_t, output_errors)
        hidden_gradient = Matrix.Map(hidden,dsigmoid)
        hidden_gradient.mul(hidden_errors)
        hidden_gradient.mul(self.learning_rate)
        inputs_t = Matrix.Transpose(inputs)
        weights_ih_deltas = Matrix.Mul(hidden_gradient, inputs_t)
        self.weights_ih.add(weights_ih_deltas)
        self.bias_h.add(hidden_gradient)

    def save(self):
        self.weights_ih.save('weights_ih')
        self.weights_ho.save('weights_ho')
        self.bias_h.save('bias_h')
        self.bias_o.save('bias_o')

    def load(self):
        self.weights_ih.load('weights_ih')
        self.weights_ho.load('weights_ho')
        self.bias_h.load('bias_h')
        self.bias_o.load('bias_o')
