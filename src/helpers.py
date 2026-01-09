from typing import Dict, List, Tuple, Union
import torch


def _validate_inputs(inputs: Dict[str, Union[int, float, None]]) -> List[str]:
    """Validate that inputs are not None and can be converted to float."""
    warnings: List[str] = []
    for k, v in inputs.items():
        if v is None:
            warnings.append(f"{k} is None (missing).")
        else:
            try:
                float(v)
            except Exception:
                warnings.append(f"{k} is not a number: {v}")
    return warnings


def _check_plausibility(
    sex_f: float, age_f: float, weight_f: float, sbp_f: float, dbp_f: float
) -> List[str]:
    """Check if input values are within plausible ranges."""
    warnings: List[str] = []
    plausibility_checks = [
        (sex_f not in (0.0, 1.0), "sex is expected to be 0 or 1."),
        (not (0 <= age_f <= 120), f"age={age_f} is outside plausible range (0–120)."),
        (not (20 <= weight_f <= 250), f"weight={weight_f} is outside plausible range (20–250 kg)."),
        (not (60 <= sbp_f <= 250), f"sbp={sbp_f} is outside plausible range (60–250 mmHg)."),
        (not (30 <= dbp_f <= 150), f"dbp={dbp_f} is outside plausible range (30–150 mmHg)."),
        (dbp_f >= sbp_f, "dbp is >= sbp, which is unusual—please double-check values."),
    ]
    for condition, message in plausibility_checks:
        if condition:
            warnings.append(message)
    return warnings


def _build_feature_vector(
    sex_f: float, age_f: float, weight_f: float, sbp_f: float, dbp_f: float,
    feature_index: Dict[str, int], device: torch.device
) -> torch.Tensor:
    """Build feature tensor in the same order as training."""
    x = torch.zeros((1, len(feature_index)), dtype=torch.float32, device=device)
    x[0, feature_index["sex"]] = sex_f
    x[0, feature_index["age"]] = age_f
    x[0, feature_index["weight"]] = weight_f
    x[0, feature_index["sbp"]] = sbp_f
    x[0, feature_index["dbp"]] = dbp_f
    return x


def _normalize_features(
    x: torch.Tensor, stats: Dict[int, Tuple[torch.Tensor, torch.Tensor]], device: torch.device
) -> Tuple[torch.Tensor, List[str]]:
    """Normalize features using training stats."""
    warnings: List[str] = []
    x_norm = x.clone()
    for col_idx, (mean, std) in stats.items():
        mean_t = mean.to(device=device, dtype=torch.float32)
        std_t = std.to(device=device, dtype=torch.float32)
        if float(std_t.item()) == 0.0:
            warnings.append(f"std for column {col_idx} is 0; skipping normalization for that column.")
            continue
        x_norm[:, col_idx] = (x_norm[:, col_idx] - mean_t) / std_t
    return x_norm, warnings


def _run_inference(
    model: torch.nn.Module, x_norm: torch.Tensor, device: torch.device
) -> float:
    """Run model inference on normalized features."""
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        y_hat = model(x_norm)
    return float(y_hat.squeeze().item())
