'''Self Attention
causal Attention
multi-head self attention
'''

import torch


class SingleHeadCausalAttention(torch.nn.Module):
    def __init__(self, dIn, dOut, contextLength, dropout, qkvBias=False):
        super().__init__()
        '''dimension of weights, (din, dOut)
        '''
        self.dOut = dOut
        self.wKey = torch.nn.Linear(dIn, dOut, bias=qkvBias)
        self.wQuery = torch.nn.Linear(dIn, dOut, bias=qkvBias)
        self.wValue = torch.nn.Linear(dIn, dOut, bias=qkvBias)
        #causal masking, only pass tokens can impact future,  not the other way around, diagonal value set to 0
        self.register_buffer(
            'mask',
            torch.triu(torch.ones(contextLength, contextLength), diagonal=1)
        )
        #buffer so that mask auotmatically moved to appropriate dvice (CPU, or GPU)
        #dropout
        self.droptout = torch.nn.Dropout(dropout)
        
    def forward(self, X):
        #X dimension (batch, context length, embedding dimension=dIn)
        b, numTokens, dIn = X.shape
        keys = self.wKey(X)
        queries = self.wQuery(X)
        values = self.wValue(X)
        # keys, queries, values dimension (batch, context length, dOut)
        
        attentionScores = queries @ keys.transpose(1, 2)  #matricx mult on last two dimensions
        #attenston scores dimension (batch, context length, context length)
        
        #causual masking
        attentionScores.masked_fill_(
            self.mask.bool()[:numTokens, :numTokens], -torch.inf
        )
        
        #softmax along last dimension (2D tension context length x context length)
        attentionWeights = torch.softmax(
            attentionScores / keys.shape[-1] ** 0.5, dim=-1
        )
        
        #dropout
        attentionWeights = self.droptout(attentionWeights)
        
        #context vectors
        contextVectors = attentionWeights @ values #2d tensor broadcast to 3-d, outout (batch, context length, dOut) 
        return contextVectors
    
class MultiHeadSelfAttention(torch.nn.Module):
    def __init__(self, dIn, dOut, numHeads, contextLength, dropout, qkvBias=False):
        super().__init__()
        self.dOut = dOut
        self.numHeads = numHeads
        assert(dOut % numHeads == 0), 'dOut must be multiples of number of heads'
        self.headDim = dOut // numHeads
        #same number of weight parameters for multi head attentions vs single head
        #pact them into a single matric for math manupilations
        self.wKey = torch.nn.Linear(dIn, dOut, bias=qkvBias)
        self.wQuery = torch.nn.Linear(dIn, dOut, bias=qkvBias)
        self.wValue = torch.nn.Linear(dIn, dOut, bias=qkvBias)
        #dropout
        self.dropout = torch.nn.Dropout(dropout)
        #causal
        self.register_buffer(
            'mask',
            torch.triu(torch.ones(contextLength, contextLength), diagonal=1)
        )
        #additional linear output layer
        self.outProj = torch.nn.Linear(dOut, dOut)
        
    def forward(self, X):
        #batch, context length, embeddind dimension
        b, numTokens, dIn = X.shape
        keys = self.wQuery(X)
        queries = self.wQuery(X)
        values = self.wValue(X)
        # dimension for keys, queries, values, (b, numTokens, dOut)
        
        #split keys, queries, values to multi heads,  last dimension to t dimensions
        keys = keys.view(b, numTokens, self.numHeads, self.headDim)
        queries = queries.view(b, numTokens, self.numHeads, self.headDim)
        values = values.view(b, numTokens, self.numHeads, self.headDim)
        
        #swap numTokens and numHead dimensions, so math will work on multi heads with headDim
        #dimension (b, numHeads, numTokens, headDim)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)
        
        # attention scores dimension (b, numHeads, numTokens, numTokens)
        attentionScores = queries @ keys.transpose(2 ,3)
        
        #causal, mask last 2 dimensions
        maskBool = self.mask.bool()[:numTokens, :numTokens]
        attentionScores = attentionScores.masked_fill_(
            maskBool, -torch.inf
        )
        
        #attention weights, dimension (d=b, numHeads, numTokens, numTokens)
        attentionWeights = torch.softmax(attentionScores / keys.shape[-1]**0.5, dim=-1)
        
        #dropout
        attentionWeights = self.dropout(attentionWeights)
        
        #dimension (b, numHeads, numTokens, headDim)
        contextVectors = attentionWeights @ values
        #swap dimension numHeads and numTokens, (b, numTokens, numHeads, headDim)
        contextVectors = contextVectors.transpose(1, 2)
        #concat headDims to dOut, note contiguous is needed 
        contextVectors = contextVectors.contiguous().view(b, numTokens, self.dOut)
        
        #final projection
        contextVectors = self.outProj(contextVectors)
        
        return contextVectors
