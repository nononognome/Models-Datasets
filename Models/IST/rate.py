from typing import Union
from torch.utils.data import TensorDataset
from torch import Tensor
import torch
import numpy as np

def rate(dataset: Union[TensorDataset, Tensor], num_steps=False, extend=False):
    if type(dataset) == TensorDataset:
        data, labels = dataset.tensors
    elif type(dataset) != Tensor:
        raise TypeError("Dataset should have been type Dataset (in form '(data, labels)') or Tensor (in form [n x t x f])")
    else:
        data = dataset


    # If Time Varying with same Dims
    if not num_steps:
        # Ensure data is between 0 and 1 and pass through bernoulli for a poisson spike train
        clipped_data = torch.clamp(data, min=0, max=1)
        output_data = torch.bernoulli(clipped_data)

    # If Time Varying with extended Dims
    elif extend:
        time_data = (data.repeat(tuple(
                    [num_steps] + torch.ones(len(data.size()), dtype=int).tolist()
                    )))

        # Ensure data is between 0 and 1 and pass through bernoulli for a poisson spike train
        clipped_data = torch.clamp(time_data, min=0, max=1)
        output_data = torch.bernoulli(clipped_data)

    # If flattened input
    else:
        data = data.flatten(start_dim=1)
        time_data = (data.repeat(tuple(
                    [num_steps] + torch.ones(len(data.size()), dtype=int).tolist()
                    )))

        # Ensure data is between 0 and 1 and pass through bernoulli for a poisson spike train
        clipped_data = torch.clamp(time_data, min=0, max=1)
        output_data = torch.bernoulli(clipped_data)
    

    if type(dataset) == TensorDataset: return TensorDataset(output_data, labels)
    else: return output_data



def count(dataset: Union[TensorDataset, Tensor], max_spikes=10, time_varying=False):
    if type(dataset) == TensorDataset:
        data, labels = dataset.tensors
    elif type(dataset) != Tensor:
        raise TypeError("Dataset should have been type Dataset (in form '(data, labels)') or Tensor (in form [n x t x f])")
    else:
        data = dataset

    # Ensure data is between 0 and 1 to note the proportion of spikes
    clipped_data = torch.clamp(data, min=0, max=1)

    # Make a mapping of the different spike patterns to the number of spikes
    # e.g. if x is 5, then the mapping is {... 5: (1, 1, 1, 1, 1, 0, 0, 0, 0, 0) ...}
    count_mapping = {x: ((1,)*x + (0,) * (max_spikes-x)) for x in range(max_spikes + 1)}

    # Apply the mapping
    mapping = torch.tensor(list(count_mapping.values())).float()

    if time_varying:
        # Calculate the number of spikes for each x
        num_spikes = ((clipped_data * max_spikes).round()).long()

        # Put the final dimension last so it can be added to
        num_spikes = torch.transpose(num_spikes, 2, 1)

        spikes = mapping[num_spikes]

        # Flatten the final dimension, put array/tensor back in (n x t x f)
        # where t now equals (t x n)
        spikes = torch.flatten(spikes, start_dim=2)
        spikes = torch.transpose(spikes, 2, 1)

    else:
        # Assuming the data is in the form of (n x t x f)
        clipped_data = clipped_data.flatten(start_dim=1)

        # Calculate the number of spikes for each x
        num_spikes = ((clipped_data * max_spikes).round()).long()

        spikes = mapping[num_spikes]
        spikes = torch.transpose(spikes, 2, 1)


    if type(dataset) == TensorDataset:
        return TensorDataset(spikes, labels)
    else:
        return Tensor(spikes)
    