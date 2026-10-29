from gpt2 import GPT2Model, GPT_CONFIG_124M
import torch

model = GPT2Model(GPT_CONFIG_124M)

totalParams = sum(p.numel() for p in model.parameters())
print(f"Total number of parmaters in this gpt2 model: {totalParams:,}")

totalSizeByte = totalParams * 4 
#float 32, 4 bytes
totalSizeMb = totalSizeByte / (1024 * 1024)
print(f"Total RAM for parameters: {totalSizeMb:.2f} MB")


def generateText(gpt, seedBatchTokens, maxNewTokensGenerate, contextLength):
    batchTokens = seedBatchTokens
    #dimension (batch, tokens count)
    for _ in range(maxNewTokensGenerate):
        #context legnth, crop if longer
        batchTokensContext = batchTokens[:, -contextLength:]
        with torch.no_grad(): 
            logits = gpt(batchTokensContext)
        #get last token generated, gptout dimension (batch, context length, vocabSize)
        logits = logits[:, -1, :]
        #dimension (batch, vocaSizee)
        probas = torch.softmax(logits, dim=-1)
        tokensGen = torch.argmax(probas, dim=-1, keepdim=True)
        #dimenstion (batch, 1)
        batchTokens = torch.cat((batchTokens, tokensGen), dim =1)
    return batchTokens

import tiktoken
tokenizer = tiktoken.get_encoding("gpt2")
seedText = "Blue sky good day"
encoded = tokenizer.encode(seedText)
encodedTensor = torch.tensor(encoded).unsqueeze(0)

model.eval()
#no dropout etc...
out = generateText(model, encodedTensor, 6, GPT_CONFIG_124M["contextLength"])
print(out)
decodedText = tokenizer.decode(out.squeeze(0).tolist())
print(decodedText)