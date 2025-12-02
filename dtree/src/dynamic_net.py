import torch.nn as nn

class DynamicNet(nn.Module):
    #Build a feedforward neural network with varying neural network layers
    def __init__(self, input_dim, hidden_size, output_dim, total_layers):
        super().__init__()

        assert total_layers >= 2, "Need at least 2 layers (1 hidden + 1 output)"

        layers = []
        layers.append(nn.Linear(input_dim, hidden_size))  # first hidden
        layers.append(nn.ReLU())

        # middle hidden layers
        for _ in range(total_layers - 2):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())

        # output layer
        layers.append(nn.Linear(hidden_size, output_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
