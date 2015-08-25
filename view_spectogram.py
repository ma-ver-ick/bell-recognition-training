from scipy.io.wavfile import read, write
import numpy as np
from pylab import plot, show, subplot, specgram

rate, data = read('data/02_ring.wav')

data = np.delete(data, np.arange(0, data.size, 2))

subplot(511)
plot(range(len(data)), data)
subplot(512)
# NFFT is the number of data points used in each block for the FFT
# and noverlap is the number of points of overlap between blocks
specgram(data, NFFT=128, noverlap=0)  # small window
subplot(513)
specgram(data, NFFT=256, noverlap=0)
subplot(514)
specgram(data, NFFT=512, noverlap=0)
subplot(515)
specgram(data, NFFT=1024, noverlap=0)  # big window

show()
