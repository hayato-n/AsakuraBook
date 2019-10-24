import numpy as np
from .activation import Sigmoid, ReLU
from .layers import AffineLayer, LinearLayer
from .loss import MeanSquareError


class MinibatchGenerator:
    def __init__(self, y, X, batchsize=32):
        self.y = y
        self.X = X
        self.N = len(self.y)
        self.batchsize = int(batchsize)

    def __call__(self):
        mask = np.random.choice(range(self.N), size=self.batchsize)
        return self.y[mask], self.X[mask]


class MLP:
    def __init__(self, input_dim, n_hidden=1, n_units=10,
                 activation=Sigmoid, lossfunc=MeanSquareError, init="Xavier"):
        self.input_dim = int(input_dim)
        self.n_hidden = int(n_hidden)
        self.n_units = int(n_units)
        self.activation = activation
        self.lossfunc = lossfunc()
        self.init = str(init)

        # define network
        self.layers = []
        # input layer
        self.layers.append(
        LinearLayer(AffineLayer(self.input_dim, self.n_units, init=self.init),
                        self.activation())
        )
        # hidden layers
        for n in range(n_hidden):
            self.layers.append(
                LinearLayer(AffineLayer(self.n_units, self.n_units, init=self.init),
                            self.activation())
            )
        # output layer
        self.layers.append(
            AffineLayer(self.n_units, 1, init=self.init)
        )

    def __call__(self, x):
        return self.forward(x)

    def predict(self, x):
        return self.forward(x)

    def forward(self, x):
        self.input = x

        proped = self.input
        for layer in self.layers:
            proped = layer.forward(proped)
        self.output = proped

        return self.output

    def get_loss(self, pred, obs):
        return self.lossfunc(pred, obs.reshape((-1, 1)))

    def backward(self):
        backproped = self.lossfunc.backward()
        for layer in reversed(self.layers):
            backproped = layer.backward(backproped)

    def update(self, optimizer):
        # update LinearLayers
        for i, layer in enumerate(self.layers[:-1]):
            layer.affine.w = optimizer(layer.affine.w,
                                       layer.affine.grad["w"],
                                       "layer-{}:w".format(i))
            layer.affine.b = optimizer(layer.affine.b,
                                       layer.affine.grad["b"],
                                       "layer-{}:b".format(i))

        # update last AffineLayer
        layer = self.layers[-1]
        layer.w = optimizer(layer.w,
                            layer.grad["w"],
                            "lastlayer:w")
        layer.b = optimizer(layer.b,
                            layer.grad["b"],
                            "lastlayer:b")
