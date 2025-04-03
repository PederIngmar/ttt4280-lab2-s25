import os
import numpy as np
from scipy.signal import detrend

def raspi_import(inputdir, filename, channels=5):
    path = f"{inputdir}/{filename}.bin"
    with open(path, 'r') as fid:
        sample_period = np.fromfile(fid, count=1, dtype=float)[0]
        data = np.fromfile(fid, dtype='uint16').astype('float64')
        # The "dangling" `.astype('float64')` casts data to double precision
        # Stops noisy autocorrelation due to overflow
        data = data.reshape((-1, channels))

    # sample period is given in microseconds-changes units to seconds
    sample_period *= 1e-6
    return sample_period, data

def detrend_all(data):
    """
    Detrend all channels in the data array.
    """
    detrended_data = np.zeros_like(data)
    for i in range(data.shape[1]):
        detrended_data[:, i] = detrend(data[:, i])
    return detrended_data


def find_latest_file(local_dir):
    files = [f for f in os.listdir(local_dir) if f.endswith('.bin')]
    if not files:
        raise FileNotFoundError("No .bin files found in the folder.")
    latest_file = max(files, key=lambda f: os.path.getmtime(os.path.join(local_dir, f)))
    return os.path.splitext(latest_file)[0]

def import_latest_file(local_dir):
    latest_file = find_latest_file(local_dir)
    sample_period, data = raspi_import(local_dir, latest_file)
    data = detrend_all(data)
    data *= 0.8e-3
    return latest_file, data

def import_all_files(local_dir):
    files = [f for f in os.listdir(local_dir) if f.endswith('.bin')]
    file_names = []
    all_data = []
    for file in files:
        file_name = os.path.splitext(file)[0]
        file_names.append(file_name)
        sample_period, data = raspi_import(local_dir, file_name)
        data = detrend_all(data)
        data *= 0.8e-3
        all_data.append(data)
    return file_names, all_data