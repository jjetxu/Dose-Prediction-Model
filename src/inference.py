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
    stats: Dict[int, Tuple[torch.Tensor, torch.Tensor]],
    feature_index: Dict[str, int],
    device: torch.device = torch.device("cpu"),
) -> Tuple[float, List[str]]:
    """
    Single-patient inference for preoperative dose.

    Inputs must match the model's expected features:
      ["sex", "age", "weight", "sbp", "dbp"]  (same order used in training)

    Normalization:
      Applies training mean/std stored in `stats` to the normalized columns.
      `stats` is expected to map col_idx -> (mean, std) for columns you normalized.

    Returns:
      (predicted_dose, warnings)
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

    # 4) Build feature vector
    x = _build_feature_vector(sex_f, age_f, weight_f, sbp_f, dbp_f, feature_index, device)

    # 5) Normalize features
    x_norm, norm_warnings = _normalize_features(x, stats, device)
    warnings.extend(norm_warnings)

    # 6) Run inference
    pred = _run_inference(model, x_norm, device)

    # 7) Check output sanity
    if pred < 0:
        warnings.append("Predicted dose is negative; consider clamping to 0 and review input validity.")

    return pred, warnings
