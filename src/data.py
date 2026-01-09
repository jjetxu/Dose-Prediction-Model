import pandas as pd
import torch
from typing import Dict, Tuple
from config import FEATURES, FEATURE_INDEX, DEVICE


def process_data(df: pd.DataFrame, device: torch.device = DEVICE) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[int, Tuple[torch.Tensor, torch.Tensor]]]:
    required_cols = FEATURES + ["preoperative_dose"]
    df = df.dropna(subset=required_cols)
    df = df[df["preoperative_dose"] > 0]
    df = df.sample(frac=1).reset_index(drop=True)

    split_idx = int(len(df) * 0.8)
    df_train = df.iloc[:split_idx].reset_index(drop=True)
    df_test = df.iloc[split_idx:].reset_index(drop=True)

    X_train = torch.tensor(df_train[FEATURES].values, dtype=torch.float32, device=device)
    y_train = torch.tensor(df_train["preoperative_dose"].values, dtype=torch.float32, device=device).unsqueeze(1)

    X_test = torch.tensor(df_test[FEATURES].values, dtype=torch.float32, device=device)
    y_test = torch.tensor(df_test["preoperative_dose"].values, dtype=torch.float32, device=device).unsqueeze(1)

    # normalization stats for age, weight, sbp, dbp (same columns used previously)
    feature_index = FEATURE_INDEX
    inputs_to_normalize = ["age", "weight", "sbp", "dbp"]
    stats = {}
    for fname in inputs_to_normalize:
        idx = feature_index[fname]
        mean = X_train[:, idx].mean()
        std = X_train[:, idx].std()
        stats[idx] = (mean, std)
        # apply normalization in-place
        if std == 0:
            std = 1.0
        X_train[:, idx] = (X_train[:, idx] - mean) / std
        X_test[:, idx] = (X_test[:, idx] - mean) / std

    return X_train, y_train, X_test, y_test, stats
