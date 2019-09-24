import numpy as np
import renom as rm
from renom.optimizer import Sgd
import matplotlib.pyplot as plt

X = np.array([[1,1],
            [1,0],
            [0,1],
            [0,0]])

y = np.array([[1],
             [0],
             [0],
             [1]])

class Mnist(rm.Model):
    def __init__(self):
        self.layer1 = rm.Dense(output_size = 5)
        self.layer2 = rm.Dense(1)

    def forward(self, x):
        t1 = self.layer1(x)
        t2 = rm.relu(t1)
        t3 = self.layer2(t2)
        return t3

epoch = 50
batch = 1
N = len(X)
optimizer = Sgd(lr=0.1, momentum=0.4)

network = Mnist()
learning_curve = []

for i in range(epoch):
    perm = np.random.permutation(N)
    loss = 0
    for j in range(0, N // batch):
        train_batch = X[perm[j*batch : (j+1)*batch]]
        response_batch = y[perm[j*batch : (j+1)*batch]]
        with network.train():
            result = network(train_batch)
            l = rm.sigmoid_cross_entropy(result, response_batch)
        grad = l.grad()
        grad.update(optimizer)
        loss += l
    train_loss = loss / (N // batch)
    learning_curve.append(train_loss)

plt.plot(learning_curve, linewidth=3, label="train")
plt.show()
