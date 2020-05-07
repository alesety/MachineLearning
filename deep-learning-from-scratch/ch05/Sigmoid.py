import numpy as np

class Sigmoid:
    def __init__(self):
        self.y = None

    def forward(self, x):
        y = 1 / (1 + np.exp(-x))
        self.y = y
        return y

    def backward(self, dout):
        dx = dout * (1.0 - self.y) * self.y
        return dx

if __name__ == "__main__":
    sigmoid = Sigmoid()
    x = np.array([[1.0, -0.5], [-2.0, 3.0]])
    print(x)
    print(sigmoid.forward(x))
    print(sigmoid.backward(x))
