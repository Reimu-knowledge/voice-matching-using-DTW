import wave

sampwidth = 1
framerate = 44100
path1=".\\Digital-Recognition-DTW_HMM_GMM-main\\Digital-Recognition-DTW_HMM_GMM-main\\records\\digit_6\\7_6.wav"
path2=".\\Digital-Recognition-DTW_HMM_GMM-main\\Digital-Recognition-DTW_HMM_GMM-main\\records\\digit_6\\8_6.wav"
with wave.open(path1,'rb') as f1:
    sampwidth = f1.getsampwidth()
    framerate = f1.getframerate()
    nframes1=f1.getnframes()
    data1=f1.readframes(nframes1)

with wave.open(path2,'rb') as f2:
    nframes2=f2.getnframes()
    data2=f2.readframes(nframes2)

with wave.open('result.wav','wb') as fw:
    fw.setnchannels(1)
    fw.setsampwidth(sampwidth)
    fw.setframerate(framerate)
    #fw.setnframes(nframes1+nframes2)
    fw.writeframesraw(data1)
    fw.writeframesraw(data2)
