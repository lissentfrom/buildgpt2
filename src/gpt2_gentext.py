from gpt2 import GPT2Model, GPT_CONFIG_124M
import torch
import tiktoken

def generateTextTopKWithTemperature(model, batchSeedTokens, maxNewTokensGenerate, contextLength, topK=None, temperature=1.0, eosToken=None):
    '''use top K tokens iwhtn hgihest probabilities in outout logits, when generate next token
    in each output logit, mask other prob number  other than top k
    scale probabilities with temprature
    use multinominal function s=to smaple 1 token as token generated
    '''
    #batchSeedTokens dimension (batch, input tokens count)
    #output dimension (batch, inout tokens count + maxNewTokensGenerate)
    for _ in range(maxNewTokensGenerate):
        #crop to context lemght if longer
        batchSeedTokensContext = batchSeedTokens[:, -contextLength:]
        with torch.no_grad():
            logits = model(batchSeedTokensContext)
            #dimension (batch, contextLength, vocabSize)
            lastTokenLogits = logits[:, -1, :]
            #dimension (batch, vocabSize)
            if topK:
                #top k
                topKLogits, topKLogitsIndices = torch.topk(lastTokenLogits, topK)
                #dimension (batch, topK)
                #print(topKLogits)
                minVal = topKLogits[:, -1]
                #print(minVal)
                #dimension (batch)
                lastTokenLogits = torch.where(
                    condition=lastTokenLogits < minVal,
                    input=torch.tensor(float('-inf')).to(lastTokenLogits.device),
                    other=lastTokenLogits
                )
            
            if temperature > 0.0:
                #scale
                lastTokenLogits = lastTokenLogits / temperature
                #softmax
                probs = torch.softmax(lastTokenLogits, dim=-1)
                #sample by multinomial function
                nextTokens = torch.multinomial(probs, num_samples=1)
            else:
                nextTokens = torch.argmax(lastTokenLogits, dim=-1, keepdim=True)
            #dimension (batch)  
            #print(nextTokens) 
            
            if nextTokens == eosToken:
                break
            
            batchSeedTokens = torch.cat((batchSeedTokens, nextTokens), dim=1)
                
    return batchSeedTokens

def generateText(gpt, seedBatchTokens, maxNewTokensGenerate, contextLength):
    '''alway use higest prob token from output logit
    no randomness'''
    batchTokens = seedBatchTokens
    #dimension (batch, tokens count)
    for _ in range(maxNewTokensGenerate):
        #context legnth, crop if longer
        batchTokensContext = batchTokens[:, -contextLength:]
        with torch.no_grad(): 
            logits = gpt(batchTokensContext)
        #get last token generated, gptout dimension (batch, context length, vocabSize)
        logits = logits[:, -1, :]
        #dimension (batch, vocaSize)
        probas = torch.softmax(logits, dim=-1)
        tokensGen = torch.argmax(probas, dim=-1, keepdim=True)
        #dimenstion (batch, 1)
        batchTokens = torch.cat((batchTokens, tokensGen), dim =1)
    return batchTokens

def createSmapleGpt2Model():
    model = GPT2Model(GPT_CONFIG_124M)

    totalParams = sum(p.numel() for p in model.parameters())
    print(f"Total number of parmaters in this gpt2 model: {totalParams:,}")

    totalSizeByte = totalParams * 4 
    #float 32, 4 bytes
    totalSizeMb = totalSizeByte / (1024 * 1024)
    print(f"Total RAM for parameters: {totalSizeMb:.2f} MB")
    return model

def generateTextSmaple(randomness=False):
    tokenizer = tiktoken.get_encoding("gpt2")
    seedText = "Blue sky good day"
    encoded = tokenizer.encode(seedText)
    encodedTensor = torch.tensor(encoded).unsqueeze(0)

    model = createSmapleGpt2Model()
    model.eval()
    #no dropout etc...
    if randomness:
        out = generateTextTopKWithTemperature(model, encodedTensor, 6, GPT_CONFIG_124M["contextLength"], topK=3, temperature=2)
    else:
        out = generateText(model, encodedTensor, 6, GPT_CONFIG_124M["contextLength"])
    print(out)
    decodedText = tokenizer.decode(out.squeeze(0).tolist())
    print(decodedText)
    
#generateTextSmaple()
#generateTextSmaple(True)
