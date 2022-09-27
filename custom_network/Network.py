import torch.nn as nn
import torch.nn.functional as F

class Network(nn.Module):
    def __init__(self, model_layers):
        super().__init__()
        self.layers = model_layers
        #self.trace = []

    def forward(self, x):
        for index, layer in enumerate(self.layers):
            # if index == 55:
            #     print('hai')
            x = layer(x)
            #self.trace.append(x)
        return x

