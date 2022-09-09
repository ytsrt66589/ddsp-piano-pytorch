import os
import torch

import ddsp_piano.utils.io_utils as io_utils

def get_training_dataset(dataset_dir, cache_data_path, device='cpu', split='train', max_polyphony=16):
    return get_dataset(dataset_dir, cache_data_path, device, split, max_polyphony)

def get_validation_dataset(dataset_dir, cache_data_path, device='cpu', split='validation', max_polyphony=16):
    return get_dataset(dataset_dir, cache_data_path, device, split, max_polyphony)

def get_test_dataset(dataset_dir, cache_data_path, device='cpu', split='test', max_polyphony=None):
    return get_dataset(dataset_dir, cache_data_path, device, split, max_polyphony)

def get_dataset(dataset_dir,
                cache_data_path,
                device='cpu',
                split='train',
                max_polyphony=16):
    """Pytorch dataset pipeline for feeding the training with conditioning
    MIDI inputs and audio target outputs. Automatically splits full tracks into
    segments.
    Args:
        - dataset_dir (path): path to the maestro-v3.0.0/ folder.
        - cache_data_path: path to the pre-processing data 
        - device: device for training, cpu or cuda
        - split (str): which dataset subset to use (among 'train', 'validation'
        and 'test').
        - max_polyphony (int): filter out segments with more simultaneous notes
        than the model polyphonic capacity. Does not filter anything if set to
        `None`.
    Returns:
        - dataset (Dataset): segments dataset
    """

    # Init Pytorch Dataset from .csv file
    dataset = io_utils.dataset_from_csv(
        os.path.join(dataset_dir, "maestro-v3.0.0.csv"),
        split=split,
        device=device,
        data_path=dataset_dir,
        cache_data_path=cache_data_path,
        max_polyphony=max_polyphony
    )

    return dataset
    