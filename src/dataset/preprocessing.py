"""Code adapted from https://github.com/amazon-science/earth-forecasting-transformer/blob/main/src/earthformer/datasets/sevir/sevir_dataloader.py"""

import numpy as np
import torch
from torch.nn.functional import avg_pool2d, interpolate

PREPROCESS_SCALE_SEVIR = {
    "vis": 1,  # Not utilized in original paper
    "ir069": 1 / 1174.68,
    "ir107": 1 / 2562.43,
    "vil": 1 / 47.54,
    "lght": 1 / 0.60517,
}
PREPROCESS_OFFSET_SEVIR = {
    "vis": 0,  # Not utilized in original paper
    "ir069": 3683.58,
    "ir107": 1552.80,
    "vil": -33.44,
    "lght": -0.02990,
}
PREPROCESS_SCALE_01 = {
    "vis": 1,
    "ir069": 1,
    "ir107": 1,
    "vil": 1 / 255,  # currently the only one implemented
    "lght": 1,
}
PREPROCESS_OFFSET_01 = {"vis": 0, "ir069": 0, "ir107": 0, "vil": 0, "lght": 0}  # currently the only one implemented


def data_dict_to_tensor(data_dict, data_types=None):
    """Convert each element in data_dict to torch.Tensor (copy without grad)."""
    ret_dict = {}
    if data_types is None:
        data_types = data_dict.keys()
    for key, data in data_dict.items():
        if key in data_types:
            if isinstance(data, torch.Tensor):
                ret_dict[key] = data.detach().clone()
            elif isinstance(data, np.ndarray):
                ret_dict[key] = torch.from_numpy(data)
            else:
                raise ValueError(f"Invalid data type: {type(data)}. Should be torch.Tensor or np.ndarray")
        else:  # key == "mask"
            ret_dict[key] = data

        # add channel dimension if missing
        if ret_dict[key].dim() == 2:
            ret_dict[key] = ret_dict[key].unsqueeze(0)
    return ret_dict


def normalize_data_dict(data_dict, img_types: set[str] = None, method: str = None):
    """Normalize the data into [0, 1] range ("01") or by subtracting the mean and dividing by standard deviation
    as done in the original paper ("sevir").

    Parameters
    ----------
    data_dict:  Dict[str, Union[np.ndarray, torch.Tensor]]
    img_types
        The data types that we want to rescale. This mainly excludes "mask" from preprocessing.
    method
        'sevir': use the offsets and scale factors in original implementation.
        '01': scale all values to range 0 to 1, currently only supports 'vil'
    Returns
    -------
    data_dict:  Dict[str, Union[np.ndarray, torch.Tensor]] normalized data
    """
    if method == "sevir":
        scale_dict = PREPROCESS_SCALE_SEVIR
        offset_dict = PREPROCESS_OFFSET_SEVIR
    elif method == "01":
        scale_dict = PREPROCESS_SCALE_01
        offset_dict = PREPROCESS_OFFSET_01
    else:
        raise ValueError(f"Invalid rescale option: {method}.")
    if img_types is None:
        img_types = data_dict.keys()
    for key, data in data_dict.items():
        if key in img_types:
            if isinstance(data, np.ndarray):
                data = scale_dict[key] * (data.astype(np.float32) + offset_dict[key])
            elif isinstance(data, torch.Tensor):
                data = scale_dict[key] * (data.float() + offset_dict[key])
            data_dict[key] = data
    return data_dict


def denormalize_data_dict(data_dict, img_types: set[str] = None, method: str = None):
    """Reverse normalization done by normalize_data_dict()"""
    if method == "sevir":
        scale_dict = PREPROCESS_SCALE_SEVIR
        offset_dict = PREPROCESS_OFFSET_SEVIR
    elif method == "01":
        scale_dict = PREPROCESS_SCALE_01
        offset_dict = PREPROCESS_OFFSET_01
    else:
        raise ValueError(f"Invalid rescale option: {method}.")
    if img_types is None:
        img_types = data_dict.keys()
    for key, data in data_dict.items():
        if key in img_types:
            data_dict[key] = data.float() / scale_dict[key] - offset_dict[key]
    return data_dict


def resize_data_dict(data_dict, downsample_dict=None, upsample_dict=None):
    """Downsample/upsample the data with factors specified in downsample_dict or upsample_dict"""
    if downsample_dict is None:
        downsample_dict = {}
    if upsample_dict is None:
        upsample_dict = {}

    for key, data in data_dict.items():
        if key in downsample_dict:
            data_dict[key] = avg_pool2d(input=data_dict[key], kernel_size=downsample_dict[key])
        elif key in upsample_dict:
            data_dict[key] = interpolate(input=data_dict[key].unsqueeze(0), scale_factor=upsample_dict[key]).squeeze(0)

    return data_dict
