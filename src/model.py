import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple


class LinearRegression(nn.Module):
  def __init__(self, input_dim, output_dim):
    super(LinearRegression, self).__init__()
    # in the constructor, define the layers we'll use.
    self.linear_layer = nn.Linear(input_dim, output_dim)

  def forward(self, X):
    # in the forward pass, connect the layers
    return self.linear_layer(X)


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


def get_model(model_name: str, input_dim: int) -> nn.Module:
    name = model_name.lower()
    if name in ("ann", "mlp", "preoperativeann"):
        return PreoperativeANN(input_dim=input_dim)
    elif name in ("linear", "linreg", "linearregression"):
        return LinearRegression(input_dim=input_dim, output_dim=1)
    else:
        raise ValueError(f"Unknown model_name: {model_name}. Use 'ann' or 'linear'.")


def train_model(
    model: nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    epochs: int = 1000,
    lr: float = 1e-3,
    weight_decay: float = 0
) -> nn.Module:
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.L1Loss()

    for epoch in range(epochs):
        model.train()
        yhat = model(X)
        loss = loss_fn(yhat, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0 or epoch == epochs - 1:
            print(f"Epoch: {epoch} | Train MAE: {loss.item():.6f}")

    return model


def evaluate_model(model: nn.Module, X: torch.Tensor, y: torch.Tensor) -> Tuple[float, float]:
    model.eval()
    with torch.no_grad():
        yhat = model(X)
        mae = torch.mean(torch.abs(yhat - y)).item()

        eps = 1e-8
        mape = (torch.mean(torch.abs((yhat - y) / (y + eps))) * 100.0).item()

    return mae, mape
