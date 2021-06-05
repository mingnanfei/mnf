from thinkdsp import CosSignal, SinSignal
from thinkdsp import decorate
import matplotlib.pyplot as plt
cos_sig = CosSignal(freq=330, amp=1.0, offset=0)
sin_sig = SinSignal(freq=880, amp=0.5, offset=0)
mix = sin_sig + cos_sig 
wave = mix.make_wave(duration=2, start=0, framerate=11025)
wave.play('temp3.wav')
spectrum = wave.make_spectrum()
spectrum.plot(high=5000)
decorate(xlabel='Frequency (Hz)')
plt.show()