import torch
from collections import OrderedDict

class DNN(torch.nn.Module):
    def __init__(self, layers):
        super(DNN, self).__init__()
        
        # parameters
        self.depth = len(layers) - 1
        
        # set up layer order dict
        self.activation = torch.nn.Tanh
        
        layer_list = list()
        for i in range(self.depth - 1): 
            layer_list.append(
                ('layer_%d' % i, torch.nn.Linear(layers[i], layers[i+1]))
            )
            layer_list.append(('activation_%d' % i, self.activation()))

        layer_list.append(
            ('layer_%d' % (self.depth - 1), torch.nn.Linear(layers[-2], layers[-1]))
        )
        layerDict = OrderedDict(layer_list)
        
        # deploy layers
        self.layers = torch.nn.Sequential(layerDict)
        
        def init_Xavier_normal(submodule):
            if isinstance(submodule, torch.nn.Linear):
                torch.nn.init.xavier_normal_(submodule.weight)
                submodule.bias.data.zero_()

        self.layers.apply(init_Xavier_normal)
        
    def forward(self, x):
        out = self.layers(x)
        return out