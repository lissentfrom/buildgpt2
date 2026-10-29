'''layer normalizarion

normalize on each embedding vecotr, to mean = 0, std deviation = 1

for training stability

also introduce trainable parameter to scale, and shift to incorprate embedding vector amplitude 
shared scale and shift for the whole layer, Rich token-to-token interactions and contextural variations are 
handled by attention layers and Feed-Forward Networks already

implemented as torch Layer
'''

import torch

class LayerNorm(torch.nn.Module):
    def __init__(self, embeddingDim):
        super().__init__()
        self.eps = 1e-5 #small number, add to std deviation to avoid divided by zero
        #shared scale and shift, trainable
        self.scale = torch.nn.Parameter(torch.ones(embeddingDim))
        self.shift = torch.nn.Parameter(torch.zeros(embeddingDim))
        
    def forward(self, x):
        #only work on last dimendsion, feature dimension
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        normX = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * normX + self.shift
    
    
        