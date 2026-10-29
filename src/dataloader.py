'''torch based dataloader for gpt training
'''

import torch
from torch.utils.data import Dataset, DataLoader
import tiktoken

class GPTDataset(Dataset):
    def __init__(self, txt, tokenizer, contextLength, stride):
        self.X = []
        self.Y = []
        
        tokenIds = tokenizer.encode(txt)
        
        for i in range(0, len(tokenIds) - contextLength, stride):
            x = tokenIds[i : i + contextLength]
            y = tokenIds[i + 1 : i + contextLength + 1]
            self.X.append(torch.tensor(x))
            self.Y.append(torch.tensor(y))
            
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]
    
    def __len__(self):
        return len(self.X)
        
def createDataLoader(txt, batchSize, contextLength, stride=1, shuffle=True, dropLast=True, numWorkers=0):
    '''droplast drop last batch if last batch is shorted of batch size
    numWorkers: how many processors for data loading
    '''
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDataset(txt, tokenizer=tokenizer, contextLength=contextLength, stride=stride)
    dataloader = DataLoader(dataset=dataset, batch_size=batchSize, shuffle=shuffle, num_workers=numWorkers, drop_last=dropLast)
    
    return dataloader

txt = '''
# Hi, I'm Yao Li 

Financial Technology Architect | Credit & CLO | Software Engineer | AI Engineering

Software architect and engineer focused on financial technology, with experience at Goldman Sachs and buy-side firms across hedge funds and CLO/credit management. I enjoy designing scalable systems, solving complex financial-data problems, and building practical solutions with modern cloud, distributed systems, and AI technologies.

## What I work with

### Enterprise Java

- Java
- J2EE / Jakarta EE
- JSP
- Servlets
- XML
- JDBC
- REST Services
- SOAP Services
- Enterprise Integration

### Python

- Python
- Fast API
- Jupyter

### Architecture

- Object-Oriented Design
- Design Patterns
- Domain Driven Design (DDD)
- Service-Oriented Architecture (SOA)
- Layered Architecture
- Enterprise Application Design
- API Design
- Performance Optimization

### Data

- Oracle
- SQL Server
- Sybase
- DB2
- MySQL
- PostgreSQL

One of the projects I'm particularly proud of involved building a **Java abstraction layer** that provided a unified query interface across multiple databases and heterogeneous data sources, simplifying enterprise data access while improving maintainability.

### Frontend

- Vue.js
- AG-Grid
- JavaScript
- HTML5
- CSS3
- JSP
- Responsive Web UI
- Bootstrap

Although most of my career has been backend-focused, I've also built and maintained user interfaces using **Vue.js, JSP, HTML, CSS, and JavaScript**, delivering complete end-to-end enterprise applications.

## Financial Technology

I've spent much of my career developing software for:

- Trading platforms
- Hedge funds
- Financial services
- Portfolio management
- Market data integration
- Enterprise back-office systems

These environments demanded high availability, performance, accuracy, and clean architecture.

## Currently Learning

Technology keeps evolving, and so do I.

I'm currently focused on modern AI development, including:

- Large Language Models (LLMs)
- Retrieval-Augmented Generation (RAG)
- Fine-tuning LLM Model
- AI-assisted software development
- Prompt Engineering
- AI Agents
- MCP (Model Context Protocol)
- Vector Databases
- Embeddings
- Local AI
- Cursor / AI coding workflows

I'm especially interested in combining decades of software engineering experience with modern AI tools to build practical, production-ready applications.

## My Philosophy

Good software isn't just about writing code.

It's about:

- Building systems that last
- Designing for change
- Writing maintainable code
- Solving real business problems
- Continuously learning new technologies

Twenty-plus years of experience has taught me that technologies change, but solid engineering principles remain valuable.

## Looking Ahead

As I continue expanding my AI engineering skills, I'm interested in opportunities involving:

- Enterprise Java modernization
- AI integration into existing systems
- Backend engineering
- Architecture and technical leadership
- Financial technology
- Intelligent business applications

---

> "Experience gives perspective. Curiosity keeps it relevant."

'''

dataloader = createDataLoader(txt, 4, 16)
dataIter = iter(dataloader)
firstBatch = next(dataIter)
print(firstBatch)
