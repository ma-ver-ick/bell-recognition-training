__author__ = 'msei'

from scipy.io.wavfile import read, write
import numpy
import random
import math
import pickle
import h5py

RING_01_TEST_DATA = '../data/01_ring'
RING_02_TEST_DATA = '../data/02_ring'


def generate_spectogram_iterator(data, window=256, multiplier=1.0, filter=None):
    """
        real + imaginary, no scaling at default, no filter at default and full window
    """

    for i in range(0, len(data)):
        fft_data = numpy.fft.fft(data[i:i+window])
        fft_data = numpy.array([numpy.real(fft_data), numpy.imag(fft_data)]).flatten()
        fft_data = numpy.real(fft_data)
        if filter:
            fft_data = numpy.clip(fft_data, filter, 65536.0)
        fft_data *= multiplier
        yield i, fft_data


def test_data_iterator(file_name, window=256, divider=2):
    """

    :param file_name:
    :param window:
    :param divider:
    :return:
    """
    wav_file_name = file_name + ".wav"
    # data_file_name = file_name + ".pickle"

    rate, data = read(wav_file_name)
    true_areas = []
    # numpy.array(pickle.load(open(data_file_name, 'r'))) // divider

    new_data = list()
    for i in range(0, len(data), divider):
        new_data.append(data[i])

    for position, fft in generate_spectogram_iterator(new_data, window=window):
        c = 0
        for r in true_areas:
            if r[0] < position + window < r[1]:
                c = 1
                break

        yield position, fft, c

# generate fft for easy loading afterwards

file_name = RING_02_TEST_DATA
divider = 2

with h5py.File(file_name + ".hdf5", "w") as f:
    rate, data = read(file_name + ".wav")

    complete = list()
    for position, fft, c in test_data_iterator(file_name):
        temp = fft
        temp = numpy.asarray(temp, dtype=numpy.int32)
        if len(temp) == 512:
            complete.append(temp)

        if position % 10000 == 9999:
            print position

    complete = numpy.asarray(complete)
    f.create_dataset("dataset", data=complete, compression="lzf", dtype=numpy.int32, shuffle=True)






