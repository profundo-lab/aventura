"""
This module is a collection of frequently used routines
and helper functions.


    - https://github.com/profundo-lab/aventura/

    actualizado en el veintidos de octubre de 2020
"""

import datetime as dt
import os
import sys
import gzip
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import shutil
from psutil import virtual_memory
import torch
import torchvision
import time
import requests
import pytz
from pathlib import Path


def determine_working_root(project_path: str) -> str:
    try:
        from google.colab import drive, files
        drive.mount('/content/drive')
        home_dir = '/content/drive/My Drive/'
        in_colab = True
    except ModuleNotFoundError:
        in_colab = False
        if sys.platform == 'linux':
            home_dir = '/mnt/hgfs/'
        else:
            home_dir = Path.home()

    return os.path.join(
        home_dir,
        '' if in_colab else 'Google Drive',
        project_path
    )

def display_working_env() -> None:
    print(f'Your runtime is running on {sys.platform}')
    ram_gb = virtual_memory().total / 1e9
    print('.............has {:.1f} gigabytes of available RAM\n'.format(ram_gb))
    print(f'Python version = {sys.version}')
    print(f'scikit-learn version = {sklearn.__version__}')
    try:
        print(f'PyTorch version = {torch.__version__}')
        print(f'TorchVision version = {torchvision.__version__}')

        device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        print('Torch Device set to -->', device)

        # Additional Info when using cuda
        if device.type == 'cuda':
            print(torch.cuda.get_device_name(0))
            print('Memory Usage:')
            print('Allocated:', round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1), 'GB')
            print('Cached:   ', round(torch.cuda.memory_reserved(0) / 1024 ** 3, 1), 'GB')
    except ModuleNotFoundError:
        pass


CST = pytz.timezone('Asia/Taipei')


def print_now() -> None:
    print(f'Local Time = {dt.datetime.now(CST)}')


def dropbox_link(did, fname) -> str:
    return 'https://dl.dropboxusercontent.com/s/%s/%s' % \
           (did, fname)


def ungzip(original, uncompressed):
    with gzip.open(original, 'rb') as f_in:
        with open(uncompressed, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)


def fetch_file_via_requests(url, save_in_dir) -> str:
    local_filename = url.split('/')[-1]
    # NOTE the stream=True parameter below
    output_path = save_in_dir + local_filename
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(output_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:  # filter out keep-alive new chunks
                    f.write(chunk)
                    # f.flush()
    return output_path


def start_plot(figsize=(10, 8), style='whitegrid', dpi=100):
    fig = plt.figure(figsize=figsize, dpi=dpi)
    gs = fig.add_gridspec(1, 1)
    plt.tight_layout()
    with sns.axes_style(style):
        ax = fig.add_subplot(gs[0, 0])
    return ax


def dough_shape(t) -> None:
    print(t)
    print('tensor address = ', t.storage().data_ptr())
    print(t.shape)
    print(f'require_grads =', t.requires_grad)
    print('-------')


def timeit(func):
    """
    Decorator for measuring function's running time.
    """
    def measure_time(*args, **kw):
        start_time = time.time()
        result = func(*args, **kw)
        print("Processing time of %s(): %.6f seconds."
              % (func.__qualname__, time.time() - start_time))
        return result

    return measure_time
