from scipy.io.wavfile import read, write
from pylab import plot, show, subplot, specgram
import pickle

rate, data = read('data/02_ring.wav')

print rate

data = data[1050210:1088990]

plot(range(len(data)), data)
show()

# r = [[13912900, 14138500], [14142600, 15195500], [15339800, 15557300]]
# pickle.dump(r, open('data/02_ring.pickle', 'w'))

