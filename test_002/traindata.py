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


def divider_iterator(input_iterator, divider=1):
    i = 0
    for element in input_iterator:
        if i % divider == 0:
            yield element

        i += 1


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


def generate_hdf5_file(input_file_name, output_file_name, divider=2, scaling=4096, window=256):
    with h5py.File(output_file_name + ".hdf5", "w") as f:
        rate, data = read(input_file_name + ".wav")

        new_data = list()
        for i in range(0, len(data), divider):
            new_data.append(data[i])

        complete = list()
        for position, fft in generate_spectogram_iterator(new_data, window):
            temp = fft * scaling
            temp = numpy.asarray(temp, dtype=numpy.int32)
            if len(temp) == window*2:  # real + imaginary
                complete.append(temp)

            if position % 10000 == 9999:
                print position
                break

        complete = numpy.asarray(complete)
        f.create_dataset("dataset", data=complete, compression="lzf", dtype=numpy.int32, shuffle=True)



# generate fft for easy loading afterwards

generate_hdf5_file(RING_01_TEST_DATA, "01_ring", divider=2)
generate_hdf5_file(RING_02_TEST_DATA, "02_ring", divider=2)




