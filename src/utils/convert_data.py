import numpy as np
import torch


def to_numpy(arr):
    if isinstance(arr, np.ndarray):
        return arr
    elif isinstance(arr, torch.Tensor):
        return arr.cpu().detach().numpy()
    elif isinstance(arr, list):
        return np.asarray(arr)
    else:
        raise NotImplementedError(f"Conversion from {type(arr)} to numpy.ndarray not implemented.")


def to_torch(arr, device=None):
    if isinstance(arr, torch.Tensor):
        ret = arr
        device = arr.device if device is None else device
    elif isinstance(arr, np.ndarray):
        ret = torch.from_numpy(arr)
    elif isinstance(arr, list):
        ret = torch.tensor(arr)
    else:
        raise NotImplementedError(f"Conversion from {type(arr)} to torch.Tensor not implemented.")

    return ret.to("cpu" if device is None else device)
