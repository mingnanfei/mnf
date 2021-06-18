import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

num = np.array([2,1])
den = np.array([1,-0.6])

ws, h = signal.freqz(num,den,whole=True)

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

plt.subplot(211)
plt.plot(ws, np.abs(h))
plt.ylabel('频幅特性')
plt.subplot(212)
plt.plot(ws, np.angle(h))
plt.ylabel('相频特性')
plt.show()