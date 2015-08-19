from scipy.io.wavfile import read, write
from pylab import plot, show, subplot, specgram

rate, data = read('data/01_ring.wav')

print rate

# data = data[1050210:1088990]

plot(range(len(data)), data)
show()

