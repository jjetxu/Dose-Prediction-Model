import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple


class PreoperativeANN(nn.Module):
    def __init__(self, input_dim: int = 5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.SELU(),
            nn.Linear(16, 8),
            nn.SELU(),
            nn.Linear(8, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_model(model: nn.Module, X: torch.Tensor, y: torch.Tensor, epochs: int = 1000, lr: float = 0.01) -> nn.Module:
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=3e-3)
    loss_fn = nn.L1Loss()
    for epoch in range(epochs):
        model.train()
        yhat = model(X)
        loss = loss_fn(yhat, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0 or epoch == epochs - 1:
            print(f"Epoch: {epoch} | Loss: {loss.item()}")
    return model


def evaluate_model(model: nn.Module, X: torch.Tensor, y: torch.Tensor) -> Tuple[float, float, float]:
    loss_fn = nn.L1Loss()
    model.eval()
    with torch.no_grad():
        yhat = model(X)
        mae = torch.mean(torch.abs(yhat - y)).item()
        eps = 1e-8
        mape = (torch.mean(torch.abs((yhat - y) / (y + eps))) * 100.0).item()
    return mae, mape
