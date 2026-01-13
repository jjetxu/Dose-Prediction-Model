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

        if epoch % 1000 == 0 or epoch == epochs - 1:
            print(f"Epoch: {epoch} | Train MAE: {loss.item():.6f}")

    return model


def evaluate_model(model: nn.Module, X: torch.Tensor, y_norm: torch.Tensor, y_stats) -> Tuple[float, float]:
    y_mean, y_std = y_stats
    model.eval()
    with torch.no_grad():
        yhat_norm = model(X)

        # denormalize back to mg
        yhat = yhat_norm * y_std + y_mean
        ytrue = y_norm * y_std + y_mean

        mae = torch.mean(torch.abs(yhat - ytrue)).item()
        eps = 1e-8
        mape = (torch.mean(torch.abs((yhat - ytrue) / (ytrue + eps))) * 100.0).item()

    return mae, mape


def get_test_predictions(
    model: torch.nn.Module,
    X_test: torch.Tensor,
    y_test_norm: torch.Tensor,
    y_stats,
):
    """
    Returns true and predicted doses in mg for the test set.
    """
    y_mean, y_std = y_stats

    model.eval()
    with torch.no_grad():
        y_pred_norm = model(X_test)

        # denormalize
        y_pred = y_pred_norm * y_std + y_mean
        y_true = y_test_norm * y_std + y_mean

    return y_true.squeeze(), y_pred.squeeze()


def bucketed_metrics_by_dose(y_true: torch.Tensor, y_pred: torch.Tensor):
    """
    Computes MAE and MAPE per true-dose quartile.
    """
    # convert to 1D CPU tensors
    y_true = y_true.detach().cpu()
    y_pred = y_pred.detach().cpu()

    # compute quartile cutoffs
    q25, q50, q75 = torch.quantile(y_true, torch.tensor([0.1, 0.5, 0.75]))

    buckets = {
        "10th percentile (lowest dose)": y_true <= q25,
        "Q2": (y_true > q25) & (y_true <= q50),
        "Q3": (y_true > q50) & (y_true <= q75),
        "Q4 (highest dose)": y_true > q75,
    }

    print("\nBucketed performance by TRUE dose:")
    for name, mask in buckets.items():
        yt = y_true[mask]
        yp = y_pred[mask]

        mae = torch.mean(torch.abs(yp - yt))
        mape = torch.mean(torch.abs((yp - yt) / (yt + 1e-8))) * 100.0

        print(
            f"{name:18s} | "
            f"N={yt.numel():3d} | "
            f"MAE={mae.item():6.2f} mg | "
            f"MAPE={mape.item():6.2f}%"
        )