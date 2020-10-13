import numpy as np
import math

def frand(x):
    return x*2-1

class Matrix: 

    def __init__(self,rows,cols):
        self.rows = rows
        self.cols = cols
        self.data = np.zeros([rows,cols]) #the matrix

    def randomize(self):
        self.data = np.random.rand(self.rows,self.cols)
        self.map(frand) #random values between -1 and 1

    def randomizeInt(self):
        self.data = np.random.randint(10, size=(self.rows,self.cols))

    def add(self,n):
        if isinstance(n, Matrix):
            self.data = np.add(self.data,n.data)
        else:
            self.data += n

    #Matrix product
    @staticmethod
    def Mul(a,b):
        try:
            result = Matrix(a.rows,b.cols)
            result.data = np.matmul(a.data,b.data)
            return result
        except:
            print("ERROR: cols of A must match rows of B when performing A*B.")

    @staticmethod
    def MulEl(a,b):
        result = Matrix(a.rows,a.cols)
        result.data = np.multiply(a.data,b.data)
        return result

    #scalar product
    def mul(self,n):
        if isinstance(n, Matrix):
            self.data = np.multiply(self.data,n.data)
        else:
            self.data *= n

    @staticmethod
    def Transpose(a):
        result = Matrix(a.cols,a.rows)
        result.data = np.transpose(a.data) 
        return result

    def transpose(self):
        self.data = np.transpose(self.data)

    @staticmethod
    def fromArray(arr):
        m = Matrix(len(arr), 1)
        for i in range(0,len(arr)):
            m.data[i][0] = arr[i]
        return m

    @staticmethod
    def createMatrixFromArray(arr, rows, cols):
        m = Matrix(rows, cols)
        for i in range(0, len(arr)):
            m.data[math.floor(i/rows)][i%cols] = arr[i]
        return m

    @staticmethod
    def subtract(a,b):
        result = Matrix(a.rows, a.cols)
        result.data = np.subtract(a.data,b.data)
        return result

    def toArray(self):
        arr = np.zeros(self.rows*self.cols)
        k=0
        for i in range (0, self.rows):
            for j in range (0, self.cols):
                arr[k] = self.data[i][j]
                k+=1
        return arr

    #apply a function to every element of the matrix
    def map(self, func):
        func_vec = np.vectorize(func)
        self.data = func_vec(self.data)

    @staticmethod
    def Map(func1,func2):
        func2_vec = np.vectorize(func2)
        func1.data = func2_vec(func1.data)
        return func1

    def print(self):
        print(self.data)

    def softmax(self):
        e_x = np.exp(self.data - np.max(self.data)) # si sottrae per il massimo per avere stabilità numerica
        self.data = e_x / e_x.sum()

    def save(self, filename):
        np.save(filename + '.npy', self.data)

    def load(self, filename):
        self.data = np.load(filename + '.npy')
