#.\\voices\\00_nativeSpeaker_fast\\native_fa_01_your01.wav
#".\\voices\\01_ZYJ_oneTime\\1_ZYJ_fa_01_your01.wav"
#".\\voices\\02_GJH_oneTime\\2_GJH_LAR1_fa_01_your01.wav"
#".\\voices\\02_GJH_oneTime\\2_GJH_LAR1_fa_07_them01.wav"
#".\\voices\\03_PX_oneTime\\03_PX_solo1_fa_14_to01.wav"
#"C:\Users\86153\Documents\luyin\luyin.wav"
import pandas as pd
import matplotlib.pyplot as plt
import webrtcvad
import struct
from scipy.io.wavfile import write
import os
import wave
import numpy as np
from scipy.io import wavfile
from scipy.fftpack import dct
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt

# 绘制时域图
def plot_time(signal, sample_rate):
    time = np.arange(0, len(signal)) * (1.0 / sample_rate)
    plt.figure(figsize=(20, 5))
    plt.plot(time, signal)
    plt.xlabel('Time(s)')
    plt.ylabel('Amplitude')
    plt.grid()
    plt.show()

# 绘制频域图
def plot_freq(signal, sample_rate, fft_size=512):
    xf = np.fft.rfft(signal, fft_size) / fft_size
    freqs = np.linspace(0, sample_rate/2,(int)(fft_size/2 + 1))
    xfp = 20 * np.log10(np.clip(np.abs(xf), 1e-20, 1e100))
    plt.figure(figsize=(20, 5))
    plt.plot(freqs, xfp)
    plt.xlabel('Freq(hz)')
    plt.ylabel('dB')
    plt.grid()
    plt.show()


# 绘制频谱图
def plot_spectrogram(spec, note):
    fig = plt.figure(figsize=(20, 5))
    heatmap = plt.pcolor(spec)
    fig.colorbar(mappable=heatmap)
    plt.xlabel('Time(s)')
    plt.ylabel(note)
    plt.tight_layout()
    plt.show()

def cal_fbank_matrix(path):
    signal,sample_rate=silencer(path)
    # print("signal:"+str(signal))
    signal = signal[0: int(3.5 * sample_rate)]  # Keep the first 3.5 seconds
    # print('sample rate:', sample_rate, ', frame length:', len(signal))

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
    #plt.show()

    frames *= hamming

    # plot_time(frames[1], sample_rate)

    # plot_freq(frames[1], sample_rate)

    NFFT = 512
    mag_frames = np.absolute(np.fft.rfft(frames, NFFT))
    pow_frames = ((1.0 / NFFT) * (mag_frames ** 2))
    #plot_spectrogram(pow_frames,"hz")
    plt.figure(figsize=(20, 5))
    plt.plot(pow_frames[1])
    plt.grid()
    #plt.show()

    low_freq_mel = 0
    high_freq_mel = 2595 * np.log10(1 + (sample_rate / 2) / 700)
    print(low_freq_mel, high_freq_mel)

    nfilt = 40
    mel_points = np.linspace(low_freq_mel, high_freq_mel, nfilt + 2)  # 所有的mel中心点，为了方便后面计算mel滤波器组，左右两边各补一个中心点
    hz_points = 700 * (10 ** (mel_points / 2595) - 1)

    fbank = np.zeros((nfilt, int(NFFT / 2 + 1)))  # 各个mel滤波器在能量谱对应点的取值
    bin = (hz_points / (sample_rate / 2)) * (NFFT / 2)  # 各个mel滤波器中心点对应FFT的区域编码，找到有值的位置
    for i in range(1, nfilt + 1):
        left = int(bin[i-1])
        center = int(bin[i])
        right = int(bin[i+1])
        for j in range(left, center):
            fbank[i-1, j+1] = (j + 1 - bin[i-1]) / (bin[i] - bin[i-1])
        for j in range(center, right):
            fbank[i-1, j+1] = (bin[i+1] - (j + 1)) / (bin[i+1] - bin[i])

    print("pow_frames.shape"+str(pow_frames.shape))
    print("fbank.T"+str(fbank.T.shape))
    filter_banks = np.dot(pow_frames, fbank.T)
    filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)
    filter_banks = 20 * np.log10(filter_banks)  # dB

    filter_banks -= (np.mean(filter_banks, axis=0) + 1e-8)

    print(filter_banks.shape)

    #plot_spectrogram(filter_banks.T, 'Filter Banks')
    print(type(filter_banks))
    return filter_banks


def silencer(path):
    mypath = path
    vad = webrtcvad.Vad(2)
    sample_rate, samples = wavfile.read(mypath)
    #plot_time(samples,sample_rate)


    raw_samples = struct.pack("%dh" % len(samples), *samples)

    window_duration = 0.03 # duration in seconds0.03
    samples_per_window = int(window_duration * sample_rate + 0.3)
    bytes_per_sample = 2

    segments = []
    speech_samples=[]
    try:       
        for start in np.arange(0, len(samples), samples_per_window):
            stop = min(start + samples_per_window, len(samples))
            is_speech = vad.is_speech(raw_samples[start * bytes_per_sample: stop * bytes_per_sample], 
                                    sample_rate = sample_rate)
            segments.append(dict(
                start = start,
                stop = stop,
                is_speech = is_speech))
        speech_samples = np.concatenate([ samples[segment['start']:segment['stop']] for segment in segments if segment['is_speech']])
    except:
        speech_samples = np.concatenate([ samples[segment['start']:segment['stop']] for segment in segments if segment['is_speech']])
    #plot_time(speech_samples,sample_rate)
    return speech_samples,sample_rate

def dtw_distance(fbank1, fbank2):
    # 计算距离矩阵
    distance_matrix = np.zeros((len(fbank1), len(fbank2)))
    for i in range(len(fbank1)):
        for j in range(len(fbank2)):
            distance_matrix[i, j] = np.linalg.norm(fbank1[i] - fbank2[j])  # 欧氏距离

    # 初始化累积距离矩阵
    accumulated_cost = np.zeros((len(fbank1), len(fbank2)))
    accumulated_cost[0, 0] = distance_matrix[0, 0]

    # 计算累积距离矩阵
    for i in range(1, len(fbank1)):
        accumulated_cost[i, 0] = accumulated_cost[i-1, 0] + distance_matrix[i, 0]
    for j in range(1, len(fbank2)):
        accumulated_cost[0, j] = accumulated_cost[0, j-1] + distance_matrix[0, j]
    for i in range(1, len(fbank1)):
        for j in range(1, len(fbank2)):
            accumulated_cost[i, j] = distance_matrix[i, j] + min(accumulated_cost[i-1, j],
                                                                 accumulated_cost[i, j-1],
                                                                 accumulated_cost[i-1, j-1])

    # 回溯最佳路径
    align_path = []
    i, j = len(fbank1) - 1, len(fbank2) - 1
    while i > 0 or j > 0:
        align_path.append((i, j))
        if i == 0:
            j -= 1
        elif j == 0:
            i -= 1
        else:
            if accumulated_cost[i-1, j] == min(accumulated_cost[i-1, j], 
                                               accumulated_cost[i, j-1], 
                                               accumulated_cost[i-1, j-1]):
                i -= 1
            elif accumulated_cost[i, j-1] == min(accumulated_cost[i-1, j], 
                                                 accumulated_cost[i, j-1], 
                                                 accumulated_cost[i-1, j-1]):
                j -= 1
            else:
                i -= 1
                j -= 1
    align_path.append((0, 0))
    align_path.reverse()

    # 对齐特征矩阵
    aligned_fbank1 = []
    aligned_fbank2 = []
    for i, j in align_path:
        aligned_fbank1.append(fbank1[i])
        aligned_fbank2.append(fbank2[j])

    return aligned_fbank1, aligned_fbank2

def standardization(matrix):
    mean = np.mean(matrix)
    std = np.std(matrix)
    standardized_matrix = (matrix - mean) / std
    return standardized_matrix

def min_max_normalization(matrix):
    min_val = np.min(matrix)
    max_val = np.max(matrix)
    normalized_matrix = (matrix - min_val) / (max_val - min_val)
    return normalized_matrix
list1=[]
df = []
f = open('./fbank/fbank.txt','w')
# for j in range (0,9):
#     f.write("number:"+str(j)+" ")
# f.write("\n")
for i in range (1,20):
    # f.write("第"+str(i)+"个声音 ")
    for j in range (0,9):
        # path1=".\\Digital-Recognition-DTW_HMM_GMM-main\\Digital-Recognition-DTW_HMM_GMM-main\\records\\digit_6\\"+str(i+1)+"_6.wav"
        # path2=".\\Digital-Recognition-DTW_HMM_GMM-main\\Digital-Recognition-DTW_HMM_GMM-main\\records\\digit_6\\"+str(i+2)+"_6.wav"
        path1=".\\Digital-Recognition-DTW_HMM_GMM-main\\Digital-Recognition-DTW_HMM_GMM-main\\records\\digit_"+str(j)+"\\"+str(i)+"_"+str(j)+".wav"#j是读的单词，i是人
        path2=".\\Digital-Recognition-DTW_HMM_GMM-main\\Digital-Recognition-DTW_HMM_GMM-main\\records\\digit_"+"1"+"\\1_"+"1"+".wav"
        # print("path1:"+path1)
        fbank1= cal_fbank_matrix(path1)
        fbank1=standardization(fbank1)
        fbank1=min_max_normalization(fbank1)
        #cal_fbank_matrix(path1)
        # print(fbank1)
        fbank2 =cal_fbank_matrix(path2)     # 假设fbank2是一个80x40的特征矩阵 cal_fbank_matrix(path2) np.random.rand(100, 40)
        fbank2=standardization(fbank2)
        fbank2=min_max_normalization(fbank2)
        # print(fbank2)
        aligned_fbank1, aligned_fbank2 = dtw_distance(fbank1, fbank2)

        # 对齐后的特征矩阵
        # print("Aligned fbank1 shape:", np.array(aligned_fbank1).shape)
        aligned_fbank1 = np.array(aligned_fbank1)
        aligned_fbank2 = np.array(aligned_fbank2)
        # 欧几里得距离
        euclidean_distance = np.linalg.norm(aligned_fbank1 - aligned_fbank2)
        print("Euclidean Distance:", euclidean_distance)
        f.write(str(euclidean_distance)+" ")
        # # 计算相关系数矩阵
        # correlation_matrix = np.corrcoef(aligned_fbank1.T, aligned_fbank2.T)

        # # 获取相关性系数
        # correlation_coefficient = correlation_matrix[0, 1]

        # print("Correlation coefficient:", correlation_coefficient)
        # # print("Aligned fbank2 shape:", np.array(aligned_fbank2).shape)
    f.write("\n")
f.close()
        
