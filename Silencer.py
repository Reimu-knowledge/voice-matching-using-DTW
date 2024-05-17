import numpy as np
from scipy.io import wavfile
from scipy.fftpack import dct
import warnings
import matplotlib.pyplot as plt
import webrtcvad
import struct
from scipy.io.wavfile import write
import os
import wave

def plot_spectrogram(spec, note):
    fig = plt.figure(figsize=(20, 5))
    heatmap = plt.pcolor(spec)
    fig.colorbar(mappable=heatmap)
    plt.xlabel('Time(s)')
    plt.ylabel(note)
    plt.tight_layout()
    plt.show()

warnings.filterwarnings('ignore')
# 绘制时域图
def plot_time(signal, sample_rate):
    time = np.arange(0, len(signal)) * (1.0 / sample_rate)
    plt.figure(figsize=(20, 5))
    plt.plot(time, signal)
    plt.xlabel('Time(s)')
    plt.ylabel('Amplitude')
    plt.grid()
    plt.show()


def cal_fbank_matrix(path):
    signal,sample_rate=silencer(path)
    print("signal:"+str(signal))
    signal = signal[0: int(3.5 * sample_rate)]  # Keep the first 3.5 seconds
    print('sample rate:', sample_rate, ', frame length:', len(signal))

    #plot_time(signal, sample_rate)

    #plot_freq(signal, sample_rate)

    pre_emphasis = 0.97
    emphasized_signal = np.append(signal[0], signal[1:] - pre_emphasis * signal[:-1])

    #plot_time(emphasized_signal, sample_rate)
    #plot_freq(emphasized_signal, sample_rate)

    frame_size, frame_stride = 0.025, 0.01
    frame_length, frame_step = int(round(frame_size * sample_rate)), int(round(frame_stride * sample_rate))
    signal_length = len(emphasized_signal)
    num_frames = int(np.ceil(np.abs(signal_length - frame_length) / frame_step)) + 1

    pad_signal_length = (num_frames - 1) * frame_step + frame_length
    z = np.zeros((pad_signal_length - signal_length))
    pad_signal = np.append(emphasized_signal, z)

    indices = np.arange(0, frame_length).reshape(1, -1) + np.arange(0, num_frames * frame_step, frame_step).reshape(-1, 1)
    frames = pad_signal[indices]
    print(frames.shape)

    hamming = np.hamming(frame_length)
    # hamming = 0.54 - 0.46 * np.cos(2 * np.pi * np.arange(0, frame_length) / (frame_length - 1))

    plt.figure(figsize=(20, 5))
    plt.plot(hamming)
    plt.grid()
    plt.xlim(0, 200)
    plt.ylim(0, 1)
    plt.xlabel('Samples')
    plt.ylabel('Amplitude')
    plt.show()

    frames *= hamming

    # plot_time(frames[1], sample_rate)

    # plot_freq(frames[1], sample_rate)

    NFFT = 512
    mag_frames = np.absolute(np.fft.rfft(frames, NFFT))
    pow_frames = ((1.0 / NFFT) * (mag_frames ** 2))
    plot_spectrogram(pow_frames,"hz")
    plt.figure(figsize=(20, 5))
    plt.plot(pow_frames[1])
    plt.grid()
    plt.show()

    low_freq_mel = 0
    high_freq_mel = 2595 * np.log10(1 + (sample_rate / 2) / 700)
    print(low_freq_mel, high_freq_mel)

    nfilt = 40
    mel_points = np.linspace(low_freq_mel, high_freq_mel, nfilt + 2)  # 所有的mel中心点，为了方便后面计算mel滤波器组，左右两边各补一个中心点
    hz_points = 700 * (10 ** (mel_points / 2595) - 1)

    fbank = np.zeros((nfilt, int(NFFT / 2 + 1)))  # 各个mel滤波器在能量谱对应点的取值
    bin = (hz_points / (sample_rate / 2)) * (NFFT / 2)  # 各个mel滤波器中心点对应FFT的区域编码，找到有值的位置
    f = open('./fbank/fbank.txt','a')
    f.write("fuck\n")
    for i in range(1, nfilt + 1):
        left = int(bin[i-1])
        center = int(bin[i])
        right = int(bin[i+1])
        for j in range(left, center):
            fbank[i-1, j+1] = (j + 1 - bin[i-1]) / (bin[i] - bin[i-1])
            f.write(str(fbank[i-1,j+1])+" ")
        for j in range(center, right):
            fbank[i-1, j+1] = (bin[i+1] - (j + 1)) / (bin[i+1] - bin[i])
            f.write(str(fbank[i-1,j+1])+" ")
        f.write("\n")
        

    f.close()

    print("pow_frames.shape"+str(pow_frames.shape))
    print("fbank.T"+str(fbank.T.shape))
    filter_banks = np.dot(pow_frames, fbank.T)
    filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)
    filter_banks = 20 * np.log10(filter_banks)  # dB

    filter_banks -= (np.mean(filter_banks, axis=0) + 1e-8)

    print(filter_banks.shape)

    plot_spectrogram(filter_banks.T, 'Filter Banks')
    print(type(filter_banks))
    return filter_banks

def silencer(path):
    mypath = path

    sample_rate, samples = wavfile.read(mypath)
    plot_time(samples,sample_rate)
    vad = webrtcvad.Vad()
    vad.set_mode(1.5)

    raw_samples = struct.pack("%dh" % len(samples), *samples)

    window_duration = 0.03 # duration in seconds0.03
    samples_per_window = int(window_duration * sample_rate + 0.3)
    bytes_per_sample = 2

    segments = []
        
    try:       
        for start in np.arange(0, len(samples), samples_per_window):
            stop = min(start + samples_per_window, len(samples))
            is_speech = vad.is_speech(raw_samples[start * bytes_per_sample: stop * bytes_per_sample], 
                                    sample_rate = sample_rate)
            segments.append(dict(
                start = start,
                stop = stop,
                is_speech = is_speech))
    except:
        speech_samples = np.concatenate([ samples[segment['start']:segment['stop']] for segment in segments if segment['is_speech']])

    return speech_samples,sample_rate
    

path=".\\Digital-Recognition-DTW_HMM_GMM-main\\Digital-Recognition-DTW_HMM_GMM-main\\records\\digit_6\\7_6.wav"