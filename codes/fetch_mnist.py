"""
This program fetches the MNIST data from MNIST DATABASE hosted
and maintained by Yann LeCun. Downloaded data will be decoded
and serialized to a pickle file for future use. The data will 
be aranged in the order described below:

    training_data       60000, 784
    training_label      60000
    test_data           10000, 784
    test_label          10000

This program can be executed in command line, use the command
    python3 fetch_mnist.py put_data_in_the_directory
    e. g.
    $ python3 fetch_mnist.py ~/Dropbox/my_project/mnist

If you're working in Windows environment, edit the source to change
directory separator (form '/' to '\').

    altualizado en el veintidos de octubre de 2020
"""
from common import fetch_file_via_requests, ungzip, print_now
from common import determine_working_root
import os
import sys
import numpy as np
from pathlib import Path
import struct
import pickle
import datetime as dt

mnist_host = 'http://yann.lecun.com/exdb/mnist/'

train_data_file = 'train-images-idx3-ubyte'
train_label_file = 'train-labels-idx1-ubyte'
test_data_file = 't10k-images-idx3-ubyte'
test_label_file = 't10k-labels-idx1-ubyte'
mnist_files = [
    train_data_file,
    train_label_file,
    test_data_file,
    test_label_file
]


def read_idx(filename):
    with open(filename, 'rb') as f:
        zero, data_type, dims = struct.unpack('>HBB', f.read(4))
        if dims == 3:
            n = struct.unpack('>I', f.read(4))[0]
            r = struct.unpack('>I', f.read(4))[0]
            c = struct.unpack('>I', f.read(4))[0]
            # shape = (n, r, c)
        else:
            n = struct.unpack('>I', f.read(4))[0]
            # shape = (n)
        # print('idx shape =', shape)
        return np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))


def fetch_remote_mnist_data(save_to: str) -> None:

    for the_file in mnist_files:
        remote_file = mnist_host + the_file + '.gz'
        fetch_file_via_requests(remote_file, save_to)
        downloaded = os.path.join(save_to, the_file + '.gz')
        result_file = os.path.join(save_to, the_file)
        ungzip(downloaded, result_file)

    train_data = read_idx(save_to + train_data_file)    # train_x
    train_x = np.reshape(train_data, (60000, 28*28)).astype(np.float)
    train_y = read_idx(save_to + train_label_file)      # train_y
    test_data = read_idx(save_to + test_data_file)      # test_x
    test_x = np.reshape(test_data, (10000, 28*28)).astype(np.float)
    test_y = read_idx(save_to + test_label_file)

    with open(save_to + "float_mnist.pkl", "bw") as f:
        data = (train_x, train_y, test_x, test_y)
        pickle.dump(data, f)


if __name__ == "__main__":

    if len(sys.argv) == 1:
        mnist_dir = os.path.join(
            determine_working_root('profundo'),
            'mnist source'
        )
    else: 
        mnist_sir = argv[1]
        
    if mnist_dir[-1] != '/':
        mnist_dir += '/'

    start_time = dt.datetime.now()
    print(f'mnist data will be stored in {mnist_dir}')
    fetch_remote_mnist_data(mnist_dir)
    print('fetching mnist db and converting idx-coded data')
    print('elapsed time =', dt.datetime.now() - start_time)
    print_now()
