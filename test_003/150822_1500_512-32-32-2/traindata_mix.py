from scipy.io.wavfile import read, write
import numpy
import random
import math
import pickle

RING_01_TEST_DATA = '../../data/01_ring'
RING_02_TEST_DATA = '../../data/02_ring'


def shuffle(data, window, deck=None):
    i_window = int(window)
    if not deck:
        deck = range(0, int(math.ceil(len(data)/(i_window*1.0))))
        random.shuffle(deck)

    ret = list()

    for i in deck:
        start = i*i_window
        end = min(start+i_window, len(data))
        for c in range(start, end):
            ret.append(data[c])

    return ret, deck


def __prepare_test_data_old(rate, data, r, window, test_value):
    c = []

    for i in range(0, len(data)):
        c.append(0)

    for i in range(0, len(r)):
        for ii in range(r[i][0], r[i][1]):
            c[ii] = test_value

    result, deck = shuffle(data, window)  # nop= range(0, 74)
    c, deck_c = shuffle(c, window, deck)

    #      audio,  classifier
    return result, c


def __prepare_tone_1_test_data_old(window, test_value=1000000):
    rate, data = read(RING_01_TEST_DATA)

    r = [[1050210, 1088990],
         [1455410, 1493870],
         [1777920, 1816660],
         [2068090, 2106850]]

    return __prepare_test_data_old(rate, data, r, window, test_value)


def __generate_spectogram_old(data, window=256, scale=None, filter=None, divider=2):
    """
    Generates (with very crude code) a crude spectogram of the 01_ring.wav file. If scale is set, the values are scaled
    to a maximum of _scale_. If filter is set, it sets every value to zero that is smaller than the given one
    (before scaling).

    :return:
        spec: The spectogram as a list of lists.
        [ 1. window: [abs(<Real>), abs(<Imaginary>)], 2. window: [abs(<Real>), abs(<Imaginary>)], 3. window: None ]
        None = Complete array is zero after scaling and filtering.
    """

    spec = list()
    m = -1
    print "1"
    for i in range(0, len(data), divider):
        f = numpy.fft.fft(data[i:i+window])
        entry = list()
        for e in f:
            temp = abs(float(numpy.real(e)))
            m = max(m, temp)
            entry.append(temp)
        for e in f:
            temp = abs(float(numpy.imag(e)))
            m = max(m, temp)
            entry.append(temp)

        if i % 10000 == 0:
            print i, "/", len(data)

        spec.append(entry)

    print "2"
    if scale is not None or filter is not None:
        if scale is not None:
            scale = float(scale) / m
        temp = list()
        for step in spec:
            inner = list()
            for e in step:
                e_t = e
                if filter is not None and e_t < filter:
                    e_t = 0
                elif scale is not None:
                    e_t = float(e) * scale

                if e_t > 0:
                    inner.append(e_t)

            if len(inner) == 0:
                inner = None
            # else:
            #     inner = numpy.array(inner)
            temp.append(inner)
        spec = temp

    return spec


def generate_spectogram_iterator(data, window=256, multiplier=1.0/65536.0, filter=0):
    """
    """

    for i in range(0, len(data) - window):
        fft_data = numpy.fft.fft(data[i:i+window])
        # fft_data = fft_data[0:window/2.0]
        fft_data = numpy.array([numpy.real(fft_data), numpy.imag(fft_data)]).flatten()
        # fft_data = numpy.real(fft_data)
        # fft_data = numpy.clip(fft_data, filter, 65536.0) * multiplier
        yield i, fft_data


def test_data_iterator(file_name, window=256, divider=2):
    """

    :param file_name:
    :param window:
    :param divider:
    :return:
    """
    wav_file_name = file_name + ".wav"
    data_file_name = file_name + ".pickle"

    rate, data = read(wav_file_name)
    true_areas = numpy.array([])  # pickle.load(open(data_file_name, 'r'))) // divider

    new_data = list()
    for i in range(0, len(data), divider):
        new_data.append(data[i])

    for position, fft in generate_spectogram_iterator(new_data, window=window):
        c = 0
        for r in true_areas:
            if r[0] + window * 2 < position < r[1] - window * 2:
                c = 1
                break

        yield position, fft, c


if __name__ == "__main__":
    pass
    # rate, data = read(RING_01_TEST_DATA)

    # first test
    # result, c = prepare_tone_1_test_data(38685*5)
    # plot(range(len(result)), result)
    # plot(range(len(c)), c)
    # show()

    # second test
    # s = generate_spectogram_old(data, window=256, scale=None, filter=10, divider=2)  # 11khz
    # pickle.dump(s, open("save.p", "wb"))

    # third test
    # max = 2^16 = 65536 normalization multiplier = 1/2^16

    # spec = list()
    # for d in generate_spectogram_iterator(data):
    #     spec.append(d)

    # third test - p2
    # temp = list()
    # for position, fft, c in test_data_iterator(RING_01_TEST_DATA):
    #     blah = [position]
    #     blah.extend(fft.tolist())
    #     blah.append(c)
    #     temp.append(blah)

    # temp = numpy.array(temp)
    # numpy.savez_compressed('delete_me.npz', temp)

    # print "completed"
