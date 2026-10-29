
'''skip connection network
add input to output of layer, so next leayer input has both
help avoid vanish gradient

implement in forward(), remember to check dimension compatability

Layer Norm
Multi Head Attenction
Dropout

skip connection
Layer Norm
Feed Fordward Network
Dropout

slip connection


'''
import torch

from attention import MultiHeadSelfAttention
from feedforward import FeedForward
from layernorm import LayerNorm

class Trasnformer(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        dIn = cfg["embeddingDim"] 
        numTokens = cfg["contextLength"]
        numHeads = cfg["attentionHeads"]
        dropout = cfg["dropoutRate"]
        qkvBias = cfg["qkvBias"]
        dOut = dIn
        self.layerNorm1 = LayerNorm(dIn)
        self.multiHeadAttn = MultiHeadSelfAttention(dIn, dOut, numHeads=numHeads, contextLength=numTokens, dropout=dropout, qkvBias=qkvBias)
        self.dropoutLayer = torch.nn.Dropout(dropout)
        self.layerNorm2 = LayerNorm(dIn)
        self.ffn = FeedForward(cfg)

        
    def forward(self, X):
        #save inpout for shortcut skip connection
        shortcut = X
        #X dimension (batch, numTokens, dIn)
        X = self.layerNorm1(X)
        X = self.multiHeadAttn(X)
        X = self.dropoutLayer(X)
        
        #shortcut connection
        X = X + shortcut
        
        shortcut = X
        X = self.layerNorm2(X)
        X = self.ffn(X)
        X = self.dropoutLayer(X)
        X = X + shortcut
        return X