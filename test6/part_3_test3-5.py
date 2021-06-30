import os
import numpy as np
import matplotlib.pyplot as plt
from thinkdsp import SinSignal
from thinkdsp import Chirp
from thinkdsp import normalize, unbias

PI2 = 2 * np.pi

class TromboneGliss(Chirp):
    """Represents a trombone-like signal with varying frequency."""
    
    def evaluate(self, ts):
        """Evaluates the signal at the given times.

        ts: float array of times
        
        returns: float wave array
        """
        l1, l2 = 1.0 / self.start, 1.0 / self.end
        lengths = np.linspace(l1, l2, len(ts))
        freqs = 1 / lengths
        
        dts = np.diff(ts, prepend=0)
        dphis = PI2 * freqs * dts
        phases = np.cumsum(dphis)
        ys = self.amp * np.cos(phases)
        return ys

low = 262
high = 349
signal = TromboneGliss(high, low)
wave1 = signal.make_wave(duration=1)
wave1.apodize()

signal = TromboneGliss(low, high)
wave2 = signal.make_wave(duration=1)
wave2.apodize()

wave = wave1 | wave2
sp = wave.make_spectrogram(1024)


plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False
sp.plot(high=1000)
plt.ylabel('频率(HZ)')
plt.xlabel('时间（s）')

wave.write(filename='output3-5.wav')
plt.show()