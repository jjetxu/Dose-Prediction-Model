import torch

# runtime device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PATH = "io/anaesthetic_data.csv"
FEATURES = ["sex", "age", "weight", "sbp", "dbp"]
FEATURE_INDEX = {k: i for i, k in enumerate(FEATURES)}
