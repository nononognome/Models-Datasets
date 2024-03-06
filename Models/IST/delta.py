from typing import Union
from torch.utils.data import TensorDataset
from torch import Tensor
import torch
import numpy as np

def delta(dataset: Union[TensorDataset, Tensor], threshold = 0.1, off_spike = False, padding = True, spiking=True, cumulative=False, return_intermediate_steps=False, threshold_as_percentage=False):
    # Check dataset input and set variables accordingly; whether it is a TensorDataset or Tensor
    if type(dataset) == TensorDataset:
        data, labels = dataset.tensors
    elif type(dataset) != Tensor:
        raise TypeError("Dataset should have been type Dataset (in form '(data, labels)') or Tensor (in form [n x t x f])")
    else:
        data = dataset
    
    # Define functions for event that cumulative flag is set
    def cumulative_spikes(delta, threshold, off_spike):
        """
        Private function which calculates cumulative sum and the spikes along the cumulative sum for some array
        Delta should be the differences between consecutive time sequences over all features, of shape (..., t, n)
        """

        def rowwise_cumulative_spikes(f, threshold, off_spike):
            '''
            Private function which calculates cumulative sum and the spikes along the cumulative sum for each time sequence
            f should be the values for all time sequences across some feature of shape (t,)

            SILLY NAIVE IMPLEMENTATION -- Much better if it ONLY used tensor/array operations
            '''

            cumulative_sum = np.cumsum(f, axis=-1)
            result = np.zeros_like(cumulative_sum, dtype=int)

            for i, t in enumerate(cumulative_sum):
                if t >= threshold:
                    cumulative_sum[i+1:] -= cumulative_sum[i]
                    result[i] = 1
                elif (off_spike and t <= -threshold):
                    cumulative_sum[i+1:] -= cumulative_sum[i]
                    result[i] = -1
            return result, cumulative_sum

        resultandSum = np.apply_along_axis(rowwise_cumulative_spikes, axis=-1, arr=delta, off_spike=off_spike, threshold=threshold)
        resultandSum = np.moveaxis(resultandSum, -2, -1)
        result = Tensor(resultandSum[...,0])
        cumulative_sum = Tensor(resultandSum[...,1])
        return result, cumulative_sum
        

    # Pad data_offset when flagged
    if padding:
        # data_offset = first time step of data + all time steps of data bar the last time step
        data_offset = torch.cat((data[:,[0]], data[:,:-1]), dim=-2)
    else:
        # data_offset = zero starting time step + all time steps of data bar the last time step
        data_offset = torch.cat((torch.zeros_like(data[:,[0]]), data[:,:-1]), dim=-2)
    
    # Calculate the difference between sequential datapoints
    delta: Tensor = (data - data_offset)

    # Calculate spikes when spiking flag is set, and a threshold set
    if spiking and threshold:
        if not cumulative:
            if not off_spike:
                output_data = torch.ones_like(data) * (delta >= threshold)
            else:
                on_spk = torch.ones_like(data) * (delta >= threshold)
                off_spk = -torch.ones_like(data) * (delta <= -threshold)
                output_data = on_spk + off_spk
        else:
            output_data, cumulative_delta = cumulative_spikes(delta, off_spike=off_spike, threshold=threshold)
    elif spiking and threshold and threshold_as_percentage:
        if not cumulative:
            threshold = np.percentile(delta.abs(), 100 - (threshold*100))
            if not off_spike:
                output_data = torch.ones_like(data) * (delta >= threshold)
            else:
                on_spk = torch.ones_like(data) * (delta >= threshold)
                off_spk = -torch.ones_like(data) * (delta <= -threshold)
                output_data = on_spk + off_spk
        else:
            threshold = np.percentile(delta.abs(), 100 - (threshold*10))
            output_data, cumulative_delta = cumulative_spikes(delta, off_spike=off_spike, threshold=threshold)
    else:
        if not cumulative:
            output_data: Tensor = delta
        else:
            _, cumulative_delta = cumulative_spikes(off_spike=off_spike, threshold=threshold)
            output_data = cumulative_delta

    
    if (return_intermediate_steps and spiking and threshold):
        if cumulative:
            if type(dataset) == TensorDataset: return TensorDataset(output_data, labels), cumulative_delta, delta
            else: return output_data, cumulative_delta, delta
        else:
            if type(dataset) == TensorDataset: return TensorDataset(output_data, labels), delta
            else: return output_data, delta
    else:
        if type(dataset) == TensorDataset: return TensorDataset(output_data, labels)
        else: return output_data