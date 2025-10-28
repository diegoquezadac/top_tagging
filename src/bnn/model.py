import torch
import torch.nn as nn

class BNN(nn.Module):
    def __init__(self, input_dim, hidden_layers=[400, 400, 400, 400], p=0.1):

        super(BNN, self).__init__()

        layers = []
        in_dim = input_dim

        for h in hidden_layers:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p))
            in_dim = h

        layers.append(nn.Linear(in_dim, 1))  # output layer

        self.net = nn.Sequential(*layers)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)