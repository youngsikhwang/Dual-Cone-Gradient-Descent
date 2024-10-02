import numpy as np
import torch

def gen_layer(input_size, layer_n, Neuron_n, output_size=1):
    layer = [input_size]
    neuron = [Neuron_n for i in range(layer_n)]
    layer = layer + neuron + [output_size]
    return layer

class Sampler:
    # Initialize the class
    def __init__(self, dim, coords, func, name=None):
        self.dim = dim
        self.coords = coords
        self.func = func
        self.name = name

    def sample(self, N):
        x = self.coords[0:1, :] + (self.coords[1:2, :] - self.coords[0:1, :]) * np.random.rand(N, self.dim)
        y = self.func(x)
        return x, y
