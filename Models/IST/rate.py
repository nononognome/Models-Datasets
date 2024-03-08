from typing import Union
from torch.utils.data import TensorDataset
from torch import Tensor
import torch
import numpy as np

def rate(dataset: Union[TensorDataset, Tensor], num_steps=False):

    if type(dataset) == TensorDataset:
        data, labels = dataset.tensors
    elif type(dataset) != Tensor:
        raise TypeError("Dataset should have been type Dataset (in form '(data, labels)') or Tensor (in form [n x t x f])")
    else:
        data = dataset


    # If Time Varying
    if not num_steps:
        # Ensure data is between 0 and 1 and pass through bernoulli for a poisson spike train
        clipped_data = torch.clamp(data, min=0, max=1)
        output_data = torch.bernoulli(clipped_data)


    # If flattened input
    else:
        time_data = (
            data.repeat(
                tuple(
                    [num_steps]
                    + torch.ones(len(data.size()), dtype=int).tolist()
                )
            )
        )

        # Ensure data is between 0 and 1 and pass through bernoulli for a poisson spike train
        clipped_data = torch.clamp(time_data, min=0, max=1)
        output_data = torch.bernoulli(clipped_data)
    

    if type(dataset) == TensorDataset: return TensorDataset(output_data, labels)
    else: return output_data