import torch
import torch.nn as nn

class BNN(nn.Module):
    def __init__(self, input_dim, p=0.1):
        super(BNN, self).__init__()
        
        self.fc1 = nn.Linear(input_dim, 400)
        self.bn1 = nn.BatchNorm1d(400)
        self.fc2 = nn.Linear(400, 400)
        self.bn2 = nn.BatchNorm1d(400)
        self.fc3 = nn.Linear(400, 400)
        self.bn3 = nn.BatchNorm1d(400)
        self.fc4 = nn.Linear(400, 400)
        self.bn4 = nn.BatchNorm1d(400)
        self.fc5 = nn.Linear(400, 400)
        self.bn5 = nn.BatchNorm1d(400)
        self.fc_out = nn.Linear(400, 1)
        self.dropout = nn.Dropout(p=p)
        
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        x = torch.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = torch.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = torch.relu(self.bn3(self.fc3(x)))
        x = self.dropout(x)
        x = torch.relu(self.bn4(self.fc4(x)))
        x = self.dropout(x)
        x = torch.relu(self.bn5(self.fc5(x)))
        x = self.dropout(x)
        x = self.fc_out(x)
        return x