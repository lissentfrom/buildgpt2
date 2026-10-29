
import torch

class GELU(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        '''element wise if inout is a multdimension tensor'''
        return 0.5 * x * (1 + torch.tanh(torch.sqrt(torch.tensor(2.0/torch.pi)) * (x + 0.044715 * torch.pow(x, 3))))
    
class FeedForward(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        '''linear, GELU, linear
        FFN is menaing of words, according to research, attentions capture contextual
        expand to higher dimension allow richer exploration of representation space
        intution is it allow intangle of diffent patterns
        '''
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(cfg["embeddingDim"], 4 * cfg["embeddingDim"]),
            GELU(),
            torch.nn.Linear(4 * cfg["embeddingDim"], cfg["embeddingDim"]),
        )
        
    def forward(self, x):
        return self.layers(x)
    
