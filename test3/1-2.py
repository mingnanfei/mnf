from thinkdsp import read_wave
import matplotlib.pyplot as plt
from IPython.display import display
wave = read_wave('C:/Users/ming  nan  fei/Desktop/浙大宁理/大三（下）/数字信号教程/test3/mingnanfei.wav')
#整个波形
wave.normalize()
wave.make_audio()
plt.subplot(4,1,1)
wave.plot()


#具有恒定音高的片段
#截取片段
segment = wave.segment(start=1.1, duration=0.3)
segment.make_audio() 

plt.subplot(4,1,2)
segment.plot()
spectrum1 = segment.make_spectrum()
wave1 = spectrum1.make_wave()
wave1.play('temp1.wav')
#频谱
spectrum = segment.make_spectrum()
plt.subplot(4,1,3)
spectrum.plot(high=7000)



wave = read_wave('C:/Users/ming  nan  fei/Desktop/浙大宁理/大三（下）/数字信号教程/test3/mingnanfei.wav')
segment = wave.segment(1.1, 0.3)
spectrum = segment.make_spectrum()
spectrum.low_pass(5000)
spectrum.high_pass(500)
spectrum.band_stop(3000,5000)
#过滤谐音后的频谱
plt.subplot(4,1,4)
spectrum.plot(high=7000)
wave = spectrum.make_wave()
#转换成回声波 生成temp3.wav音频
wave.play('temp2.wav')
plt.show()
