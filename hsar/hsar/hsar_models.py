import torch

class PostureMLP(torch.nn.Module):
    def __init__(self, input_dim=99, num_classes=3):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 128),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(128),
            torch.nn.Dropout(0.3),

            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(64),
            torch.nn.Dropout(0.3),

            torch.nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.net(x)