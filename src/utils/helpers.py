import numpy as np

def convert_ndarray(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: convert_ndarray(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_ndarray(v) for v in obj]
    return obj