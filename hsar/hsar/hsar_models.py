import torch

# ============================================================
# MODELS
# ============================================================

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
    
    

class ArmStateMLP(torch.nn.Module):
    def __init__(self, input_dim=24, num_classes=3):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 64),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(64),
            torch.nn.Dropout(0.3),

            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(32),
            torch.nn.Dropout(0.3),

            torch.nn.Linear(32, num_classes),
        )

    def forward(self, x):
        return self.net(x)



class PointingMLP(torch.nn.Module):
    def __init__(self, input_dim=24, num_classes=2):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 64),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(64),
            torch.nn.Dropout(0.3),

            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(32),
            torch.nn.Dropout(0.3),

            torch.nn.Linear(32, num_classes)
        )

    def forward(self, x):
        return self.net(x)



class ArmRaisedDirectionMLP(torch.nn.Module):
    def __init__(self, input_dim=24, num_classes=2):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 64),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(64),
            torch.nn.Dropout(0.3),

            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(32),
            torch.nn.Dropout(0.3),

            torch.nn.Linear(32, num_classes),
        )

    def forward(self, x):
        return self.net(x)
    