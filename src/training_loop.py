import torch
from tokenizer import GPT2_TOKENIZER, textToTokenIdsTensor, tokenIdsTensorToText
from gpt2_gentext import generateText

'''typical pytorch training loop 
epochs loop
    batch loop
        reset gradients
        forward, calcualte batch loss
        calculate gradients
        step backward to adjust parameters
        typically compute traing loss, validation loss periodically to show progress

    optional genreation show model capablility in each epoch
        '''

def trainingLoop(model, optimizer, epochs, trainingDataLoader, validationDataLoader, device, evalEveryNumberOfSteps, evalNumberOfBatchs, samplePrompt, tokenizer):
    #track traing loss, validation loss, token seeneach time eval
    trainingLosses, validationLosses, tokensSeen = [], [], []
    totalTokensSeen = 0
    globalTraingCounter = -1
    
    for epoch in range(epochs):
        model.train()
        for X, Y in trainingDataLoader:
            optimizer.zero_grad()
            
            loss = calculateBatchLoss(model, X, Y, device)
            
            loss.backward()
            optimizer.step()
            
            globalTraingCounter = globalTraingCounter + 1
            totalTokensSeen = totalTokensSeen + X.numel()
            print(globalTraingCounter, totalTokensSeen)
            
            if globalTraingCounter % evalEveryNumberOfSteps == 0:
                tokensSeen.append(totalTokensSeen)
                #time to eval
                trainLoss, valLoss = evaluateModelLoss(model, trainingDataLoader, validationDataLoader, device, evalNumberOfBatchs)
                trainingLosses.append(trainLoss)
                validationLosses.append(valLoss)
                print(f"Epoch {epoch +1}, Step {globalTraingCounter}",
                      f"traing loss: {trainLoss:.3f}",
                      f"validation loss: {valLoss:.3f}")
        
        #generate sampel showing progession
        generateAndShowSample(model, tokenizer, device, samplePrompt)
        
    return trainingLosses, validationLosses, tokensSeen

def calculateBatchLoss(model, X, Y, device):
    '''compute model output, use cross entropy loss function
    cross entropy capture likehood of two probability vectors'''
    inputBatch = X.to(device)
    targetBatch = Y.to(device)
    #dimension (batch, contextLength)
    logits = model(inputBatch)
    #dimension (batch, contextLength, vocabSize)
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), targetBatch.flatten())
    # dimension (batch * contextLength, vocabSize),  (batch * contextLength)
    '''torch Applies Softmax along the vocabulary dimension ($V$) for each of the $N$ tokens to convert logits into probabilities.
    Extracts the predicted probability assigned to the true class given by target_batch.
    Computes log(ext{probability})$ for each token.
    Takes the mean across all $N$ tokens in the flattened batch to return a single floating-point scalar value. '''
    
    return loss

def calculateDataLoadLoss(model, dataLoader, device, evalNumberOfBatchs=None):
    '''calculate loss for the model on all data, 
    on all data in data loaders, optionally limited by evalNumberOfBatchs'''
    loss = 0.0
    if len(dataLoader) == 0:
        return float('nan')
    elif evalNumberOfBatchs:
        numBatches = min(evalNumberOfBatchs, len(dataLoader))
    else:
        numBatches = len(dataLoader)
        
    for b, (X, Y) in zip(range(numBatches), dataLoader):
        loss = loss + calculateBatchLoss(model, X, Y, device)

    return loss / numBatches

def evaluateModelLoss(model, trainingDataLoader, validationDataLoader, device, evalNumberOfBatchs):
    '''calculate training loss and validation loss for the model, 
    on all data in data loaders, optionally limited by evalNumberOfBatchs'''
    model.eval()
    with torch.no_grad():   #no tracking on gradient, less computation
        trainLoss = calculateDataLoadLoss(model, trainingDataLoader, device=device, evalNumberOfBatchs=evalNumberOfBatchs)
        valLoss = calculateDataLoadLoss(model, validationDataLoader, device=device, evalNumberOfBatchs=evalNumberOfBatchs)
    
    model.train()
    return trainLoss, valLoss

def generateAndShowSample(model, tokenizer, device, samplePrompt):
    '''generate text from sample prompt'''
    model.eval()
    with torch.no_grad():
        contextLength = model.positionEmbedding.weight.shape[0]
        encoded = textToTokenIdsTensor(samplePrompt, tokenizer).to(device=device)
        textTokens = generateText(model, encoded, 50, contextLength=contextLength)
        text = tokenIdsTensorToText(textTokens, tokenizer)
        print(text)
    model.train()

from gpt2 import GPT2Model, GPT_CONFIG_124M
from dataloader import createDataLoader

def train():
    batchSize = 2
    contextLength = 256
    
    cfg = GPT_CONFIG_124M
    cfg["contextLength"] = contextLength
    
    with open("src/the-veridct.txt", "r", encoding="utf-8") as f:
        rawText = f.read()
    print("Total number of character:", len(rawText))
    print(rawText[:99])

    splitIdenx = int(0.85 * len(rawText))
    trainText = rawText[:splitIdenx]
    valText = rawText[splitIdenx:]
    print(len(trainText), len(valText))
    trainLoader = createDataLoader(trainText, batchSize=batchSize, contextLength=contextLength, stride=contextLength)
    valLoader = createDataLoader(valText, batchSize=batchSize, contextLength=contextLength, stride=contextLength)
    
    torch.manual_seed(123)
    model = GPT2Model(GPT_CONFIG_124M)
    
    # Select MPS if available, otherwise fall back to CPU
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")    
    print(f"Using device: {device}")

    model.to(device)
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr = 0.0004,
        weight_decay=0.1
    )
    
    numEpochs = 5
    trainLosses, valLosses, tokenSeens = trainingLoop(model, 
                                                      optimizer=optimizer,
                                                      epochs=numEpochs,
                                                      trainingDataLoader=trainLoader,
                                                      validationDataLoader=valLoader,
                                                      device=device,
                                                      evalEveryNumberOfSteps=5,
                                                      evalNumberOfBatchs=None,
                                                      samplePrompt="Great people reads novels",
                                                      tokenizer=GPT2_TOKENIZER
                                                      )
    
    
train()
