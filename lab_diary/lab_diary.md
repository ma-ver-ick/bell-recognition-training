
# Ring Tone Recoginition

## Introduction

## Restrictions

* Seems that the teensy can only do a fft-window of 256 samples (without running out of memory).

## Steps

### view_waveform.py

#### Data (01_ring)

Quiet, only a few noise in the beginning.

* Bell 1 (1050210-1277290)
    * Tone 1: 1050210 - 1088990
    * Tone 2:  - 1127350
    * Tone 3:  - 1277290
* Bell 2 (1455410-1682420)
    * Tone 1: 1455410 - 1493870
    * Tone 2:  - 1532500
    * Tone 3:  - 1682420
* Bell 3 (1777920-2005890)
    * Tone 1: 1777920 - 1816660
    * Tone 2:  - 1855110
    * Tone 3:  - 2005890
* Bell 4 (2068090-2295410)
    * Tone 1: 2068090 - 2106850
    * Tone 2:  - 2145600
    * Tone 3:  - 2295410

#### Data (02_ring)

Star Trek Voyager in the living room at room levels. A bit of 'silent' vacuum cleaning and then four bells in the end (with voyager still running, but the vacuum not).


# Mini-Analysis

* Avg tone 1 length (until tone 2 starts) 38780, 38460, 38740, 38760 = 38.685 
* Avg bell length: 227080, 227010, 227970, 227320 = 181.876 


### view_spectogram.py

![Spectogram FFT=128, 256, 512, 1024](00_figure.png)

* See the tones at around 600k, 800k, ...
* You can clearly see much higher frequencies in the lower plots, mainly because of the larger FFT window. The lower window also seems promising, because the tone is very clearly.
* The door bell is clearly overlapping three different tones each time. 


# Generating Input for the Neural Net

## Second experiment traindata_mix.generate_spectogram_old

Simple example to generate the FFT of a window from the data (window length=256) and a step size (sampling rate divider) of 2 resulted in a pickle file size of over 4 GB. This is unacceptable. 
Further ideas: 
* use a generator and feed the neural net directly (this makes scaling difficult, but the maximum FFT value could be determined upfront).
* try to reduce the amount of data somehow. We need this much data only for the first few tries, after that we can use the neural net to determine which data might be worth focusing on (because the weights would be highest along these data paths).

## Third experiment traindata_mix.generate_spectogram_iterator

* Only the first half of the frequencies are used
* Only the real part of the frequencies are used
* Amplitude is scaled to 1/2^16 (max amplitude in the source)

Resulting file size is still over 3GB when pickled, but can now be iterated

# Training the neural net

## test_001 - Neural net of 128 - 128*5 - 128*5 - 2

__Thoughts:__
* Used 01_ring.wav
* Uses a network with two hidden layers which count of neurons are multiples of the window size of the fft (also the frequencies).
* Two hidden layers just for fun.
* The train, val and test data are cut from the 01_ring.wav by taking a good look on it and using the easiest path to cut the data.
* So far, THEANO uses all CPU's and easily more than 19GB of RAM. Searching for something to cut down RAM usage (or using a larger machine...)

__Results:__
* Larger machine found a solution with a accuracy of 87.30 %
    * Used Amazon EC2, with 32 CPU's and a huge junk of memory (only around 10GB was used) and used around 3000 minutes of CPU time (top) = $12.11
* Large junks of no bell sounds where recognized also, will have to look at the data if these audio-frames can actually be differentiated with the given FFT data.
* Second figure shows the neural net on a new set of data (nn was not trained, validated, tested on). The four large (partly green) blocks at the end are the door bell.

![Results using the resulting NN. Green shows the output of the NN when it's considering the audio signal the bell it was trained on.](01_figure.png)
![New data set with four door bell sounds at the end. There was a vacuum cleaner and Star Trek Voyager as background or upfront.](02_figure.png)

## test_002 - Neural net of 512 - 256*2 - 256*2 - 2

__Setup:__

* There seems to be a plateau around 87% (test_002), this may have many many causes. I have picked the following: 
    * The remaining test samples look very similar to each other and it's very hard to differentiate between them.
    * The NN seems to have a harder time on the validation data vs the training data. So I switched both sets (and should rethink the distribution as it's entirely abitary).
* FFT uses now the full real and imaginary data. No scaling, no filter just plain FFT (as I now have access to a very large computer this is possible). See traindata_mix.generate_spectogram_iterator.

__First Try:__

* Overfitting: Choose examples randomly from the dataset and make sure that every set (train, val, test) have the same ratio of doorbell vs. not-doorbell.


## test_002-2 Neural net of 512 - 256 - 256 - 2


# Neural Nets and the Photon!

## Testing XOR

#include "math.h"

Code


    void setup() {
        Serial.begin(9600);
        Serial.println("Hello Computer");
        
    }
    
    
    void loop() {
        unsigned long before = millis();    
        for(int i = 0; i < 10000; i++) {
            evaluate(10, 100);
        }
        unsigned long after = millis();
        unsigned long diff = after-before;
        Serial.println("time for 10000 = " + String(diff));
    }
    
    int evaluate(float input_0, float input_1) {
        // layer: 0, name: None
        float n_out_0_0 = sigmoid( (-0.113556) * input_0 + (1.952280) * input_1 + (-0.420871));
        float n_out_0_1 = sigmoid( (-4.699653) * input_0 + (-4.362170) * input_1 + (1.333518));
        float n_out_0_2 = sigmoid( (0.372461) * input_0 + (-3.111637) * input_1 + (0.791882));
        float n_out_0_3 = sigmoid( (-0.955691) * input_0 + (-0.507858) * input_1 + (0.082308));
        float n_out_0_4 = sigmoid( (-4.293386) * input_0 + (2.813902) * input_1 + (-1.384903));
        // layer: 1, name: None
        float n_out_1_0 = (2.430914) * n_out_0_0 + (3.643128) * n_out_0_1 + (-2.860000) * n_out_0_2 + (-0.248841) * n_out_0_3 + (-3.294573) * n_out_0_4 + (0.221753);
        float n_out_1_1 = (-0.724163) * n_out_0_0 + (-4.231144) * n_out_0_1 + (2.033271) * n_out_0_2 + (0.699811) * n_out_0_3 + (2.863805) * n_out_0_4 + (-0.221753);
    
        if(n_out_1_0 > n_out_1_1) { return 1; }
        else if(n_out_1_1 > n_out_1_0) { return 0; }
    
        return -1;
    }
    
    float sigmoid(float in) {
        return 1 / (1 + exp(in));   
    }
´´´

Result:


    time for 10000 = 1768


* 10.000 Evaluations with 1/(1+exp(in)) need 1.768 seconds (it has to be <<< 1s!)
* 10.000.000 Evaluations with ´´´return in;´´´ take 0 seconds
* 10.000.000 Evaluations with ´´´1/(1+in)´´´ take 0 seconds
* 10.000.000 Evaluations of the 10000000 take 0 seconds
