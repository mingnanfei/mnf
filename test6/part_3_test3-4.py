import os
import numpy as np
import matplotlib.pyplot as plt
from thinkdsp import read_wave

wave = read_wave('C:/Users/38407/Desktop/2/数字信号处理/python/5python练习第三章/72475__rockwehrmann__glissup02.wav')

plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False
wave.make_spectrogram(512).plot(high=5000)
plt.ylabel('频率(HZ)')
plt.xlabel('时间（s）')
wave.write(filename='output3-4.wav')
plt.show()
