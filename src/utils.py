import os
import cv2
import numpy as np 
from pathlib import Path
from skimage import measure, exposure
import skimage.io
import colorsys
import pickle
import h5py

DEFAULT_CHANNELS = (1, 2)

RGB_MAP = {
    1: {"rgb": np.array([255, 0, 0]), "range": [0, 50]},
    2: {"rgb": np.array([0, 0, 255]), "range": [0, 60]},
}


def convert_to_rgb(t, channels=DEFAULT_CHANNELS, vmax=255, rgb_map=RGB_MAP):
    """
    Converts and returns the image data as RGB image
    Parameters
    ----------
    t : np.ndarray
        original image data
    channels : list of int
        channels to include
    vmax : int
        the max value used for scaling
    rgb_map : dict
        the color mapping for each channel
        See rxrx.io.RGB_MAP to see what the defaults are.
    Returns
    -------
    np.ndarray the image data of the site as RGB channels
    """
    dim1, dim2, _ = t.shape
    colored_channels = []
    for i, channel in enumerate(channels):
        x = (t[:, :, channel - 1] / vmax) / (
            (rgb_map[channel]["range"][1] - rgb_map[channel]["range"][0]) / 255
        ) + rgb_map[channel]["range"][0] / 255
        x = np.where(x > 1.0, 1.0, x)
        x_rgb = np.array(
            np.outer(x, rgb_map[channel]["rgb"]).reshape(dim1, dim2, 3), dtype=int
        )
        colored_channels.append(x_rgb)
    im = np.array(np.array(colored_channels).sum(axis=0), dtype=int)
    im = np.where(im > 255, 255, im)
    im = im.astype(np.uint8)
    return im

def one_channel(t, channel, vmax=255, rgb_map=RGB_MAP):
    """
    Converts and returns the image data as RGB image
    Parameters
    ----------
    t : np.ndarray
        original image data
    channels : list of int
        channels to include
    vmax : int
        the max value used for scaling
    rgb_map : dict
        the color mapping for each channel
        See rxrx.io.RGB_MAP to see what the defaults are.
    Returns
    -------
    np.ndarray the image data of the site as RGB channels
    """
    dim1, dim2, _ = t.shape
    colored_channels = []
    x = (t[:, :, 0] / vmax) / (
        (rgb_map[channel]["range"][1] - rgb_map[channel]["range"][0]) / 255
    ) + rgb_map[channel]["range"][0] / 255
    x = np.where(x > 1.0, 1.0, x)
    x_rgb = np.array(
        np.outer(x, rgb_map[channel]["rgb"]).reshape(dim1, dim2, 3), dtype=int
    )
    colored_channels.append(x_rgb)
    im = np.array(np.array(colored_channels).sum(axis=0), dtype=int)
    im = np.where(im > 255, 255, im)
    im = im.astype(np.uint8)
    return im

def save_pkl(filename, save_object):
    writer = open(filename,'wb')
    pickle.dump(save_object, writer)
    writer.close()

def load_pkl(filename):
    loader = open(filename,'rb')
    file = pickle.load(loader)
    loader.close()
    return file

def save_hdf5(path:str, name:str, data: np.ndarray, attr_dict= None, mode:str='a') -> None:
    # Read h5 file
    hf = h5py.File(path, mode)
    # Create z_stack_dataset
    if hf.get(name) is None:
        data_shape = data.shape
        data_type = data.dtype
        chunk_shape = (1, ) + data_shape[1:]
        max_shape = (data_shape[0], ) + data_shape[1:]
        dset = hf.create_dataset(name, shape=data_shape, maxshape=max_shape, chunks=chunk_shape, dtype=data_type, compression="gzip")
        dset[:] = data
        if attr_dict is not None:
            for attr_key, attr_val in attr_dict.items():
                dset.attrs[attr_key] = attr_val
    else:
        print(f'Dataset {name} exists')
        
    hf.close()