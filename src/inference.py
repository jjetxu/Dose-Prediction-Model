import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Union
from helpers import (
    _validate_inputs,
    _check_plausibility,
    _build_feature_vector,
    _normalize_features,
    _run_inference,
)


def predict_preop_dose(
    sex: Union[int, float],
    age: Union[int, float],
    weight: Union[int, float],
    sbp: Union[int, float],
    dbp: Union[int, float],
    model: nn.Module,
    stats: Dict[int, Tuple[torch.Tensor, torch.Tensor]],   # X normalization stats
    y_stats: Tuple[torch.Tensor, torch.Tensor],            # (y_mean, y_std)
    feature_index: Dict[str, int],
    device: torch.device = torch.device("cpu"),
) -> Tuple[float, List[str]]:
    """
    Single-patient inference for preoperative dose (mg).

    Model outputs normalized dose; this function denormalizes back to mg.
    """

    # 1) Validate inputs
    inputs = {"sex": sex, "age": age, "weight": weight, "sbp": sbp, "dbp": dbp}
    warnings = _validate_inputs(inputs)

    if any("missing" in w or "not a number" in w for w in warnings):
        return float("nan"), warnings

    # 2) Convert to floats
    sex_f, age_f, weight_f, sbp_f, dbp_f = (
        float(sex), float(age), float(weight), float(sbp), float(dbp)
    )

    # 3) Plausibility checks
    warnings.extend(_check_plausibility(sex_f, age_f, weight_f, sbp_f, dbp_f))

    # 4) Build feature vector (raw)
    x = _build_feature_vector(
        sex_f, age_f, weight_f, sbp_f, dbp_f, feature_index, device
    )

    # 5) Normalize features using training stats
    x_norm, norm_warnings = _normalize_features(x, stats, device)
    warnings.extend(norm_warnings)

    # 6) Run inference (normalized output)
    pred_norm = _run_inference(model, x_norm, device)

    # 7) Denormalize output back to mg
    y_mean, y_std = y_stats
    pred_mg = pred_norm * float(y_std.item()) + float(y_mean.item())

    # 8) Output sanity check
    if pred_mg < 0:
        warnings.append(
            "Predicted dose is negative; consider clamping to 0 and review input validity."
        )

    return float(pred_mg), warnings
