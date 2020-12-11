import os
from typing import Optional, Union

import torch


def set_device_id(device_id: Optional[Union[int, str]] = None) -> str:
    """
    Converts device id to cuda/cpu str for moving tensors
    :param device_id:
    :return: cuda or cpu
    """
    if device_id is not None and device_id != "cpu" and torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
        return "cuda"
    else:
        return "cpu"
