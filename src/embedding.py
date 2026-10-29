'''embedding for gpt training
'''

import torch

vocSize = 6
contextLength = 2
embeddingDim = 3

#init
torch.manual_seed(123)

embeddingLayer = torch.nn.Embedding(vocSize, embedding_dim=embeddingDim)

print(embeddingLayer(torch.tensor([3, 4])))

positionEmbeddingLayer = torch.nn.Embedding(contextLength, embedding_dim=embeddingDim)
positionEmbeddings = positionEmbeddingLayer(torch.arange(contextLength))

print(positionEmbeddings)