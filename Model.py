from torch import nn

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.h1 = nn.Linear(12, 6)
        self.h2 = nn.Linear(6, 3)
        self.output = nn.Linear(3, 1)
        
        self.relu = nn.ReLU()
        
        self.dropout = nn.Dropout(p=0.167)
        self.batchnorm1 = nn.BatchNorm1d(6)
        self.batchnorm2 = nn.BatchNorm1d(3)
        
    def forward(self, x):
        x = self.relu(self.h1(x))
        x = self.batchnorm1(x)
        x = self.dropout(x)
        x = self.relu(self.h2(x))
        x = self.batchnorm2(x)
        x = self.output(x)
        
        return x