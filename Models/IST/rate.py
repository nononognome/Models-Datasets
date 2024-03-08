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

    if time_varying:
        # Calculate the number of spikes for each x
        num_spikes = ((clipped_data * max_spikes).round()).long()

        # Make a mapping of the different spike patterns to the number of spikes
        # e.g. if x is 5, then the mapping is {... 5: (1, 1, 1, 1, 1, 0, 0, 0, 0, 0) ...}
        count_mapping = {x: ((1,)*x + (0,) * (max_spikes-x)) for x in range(max_spikes + 1)}

        # Put the final dimension last so it can be added to
        num_spikes = torch.transpose(num_spikes, (0, 2, 1))

        # Apply the mapping
        mapping = torch.tensor(list(count_mapping.values()))
        spikes = mapping[num_spikes]

        # Flatten the final dimension, put array/tensor back in (n x t x f)
        # where t now equals (t x n)
        spikes = torch.stack(spikes)
        spikes = torch.transpose(spikes, (1, 2, 3, 0))
        spikes = spikes.reshape(num_spikes.shape + (-1,))
        spikes_shape = spikes.shape
        spikes = torch.reshape(spikes, (spikes_shape[0], spikes_shape[1], -1))
        spikes = torch.transpose(spikes, (0, 2, 1))

    else:
        # Assuming the data is in the form of (n x t x f)
        clipped_data = clipped_data.flatten(start_dim=1)

        # Calculate the number of spikes for each x
        num_spikes = ((clipped_data * max_spikes).round()).long()

        # Make a mapping of the different spike patterns to the number of spikes
        # e.g. if x is 5, then the mapping is {... 5: (1, 1, 1, 1, 1, 0, 0, 0, 0, 0) ...}
        count_mapping = {x: ((1,)*x + (0,) * (max_spikes-x)) for x in range(max_spikes + 1)}

        # Apply the mapping
        mapping = torch.tensor(list(count_mapping.values()))
        spikes = mapping[num_spikes].float()


    if type(dataset) == TensorDataset:
        return TensorDataset(spikes, labels)
    else:
        return Tensor(spikes)
    