import torch.nn as nn

class NEWMODEL(nn.Module):
    def __init__(self, args):
        super(NEWMODEL, self).__init__()
        self.args = args
        self.linear = nn.Linear(1, 1)
    
    def forward(self, x):
        return x