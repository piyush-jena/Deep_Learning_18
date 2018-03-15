import numpy as np
np.random.seed(100)

def relu(x,deriv=False):
    if (deriv==True):
        return 1 * (x > 0)
    return x * (x > 0)

def softmax(x):
    X = np.exp(x)
    sumX = np.sum(X, axis=1, keepdims=True)
    return X/sumX

class NN(object):
    def __init__(self,input,hidden,output):
        self.input = input
        self.hidden = hidden
        self.output = output
        self.W1 = 0.01 * np.random.randn(self.input,self.hidden)
        self.W2 = 0.01 * np.random.randn(self.hidden,self.output)
        self.B1 = np.zeros((1,self.hidden))
        self.B2 = np.zeros((1,self.output))
        self.lr = 0.001
        self.rgparam = 0.01
        self.batch_size = 60000
        self.loss = 100
    def forward(self,inputs,outputs):
        A1 = np.dot(inputs, self.W1) + self.B1
        H1 = relu(A1)
        A2 = np.dot(H1, self.W2) + self.B2
        O1 = softmax(A2)
        cross_entropy = -1*np.log(O1[range(self.batch_size),outputs])
        cross_entropy_loss = np.sum(cross_entropy)/self.batch_size
        reg_loss = 0.5*self.rgparam*np.sum(np.square(self.W1)) + 0.5*self.rgparam*np.sum(np.square(self.W2))
        self.loss = cross_entropy_loss + reg_loss
        delta = O1
        delta[range(self.batch_size),outputs] -= 1
        delta /= self.batch_size
        return H1,delta,A2
    def backward(self, inputs, hidden_layer, delta):
        dW2 = np.dot(hidden_layer.T, delta)
        db2 = np.sum(delta, axis=0, keepdims=True)
        dhidden = np.dot(delta, (self.W2).T)
        dhidden[hidden_layer <= 0] = 0
        dW1 = np.dot(inputs.T, dhidden)
        db1 = np.sum(dhidden, axis=0, keepdims=True)
        dW2 += self.rgparam * self.W2
        dW1 += self.rgparam * self.W1
        self.W1 += -1*self.lr * dW1
        self.B1 += -1*self.lr * db1
        self.W2 += -1*self.lr * dW2
        self.B2 += -1*self.lr * db2
