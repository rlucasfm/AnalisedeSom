import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
import numpy as np
import sounddevice as sd

filename = 'worksample.wav'

# GRAVAÇÃO DO ARQUIVO
fs = 44100  # Sample rate
seconds = 3  # Duração da gravação

print('Gravação em Andamento...')
myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=1)
sd.wait()  # Espera até o fim do processo de gravamento
print('Gravação completa')
wav.write(filename, fs, myrecording)  # Salva como um arquivo WAV


# APLICAÇÃO DA FFT E GRÁFICOS
samplerate, data = wav.read(filename) # Lê o samplerate e informações do arquivo

nf=16384
Y = np.fft.fft(data,nf) # Aplicação da FFT no áudio

# Normalização da saída da FFT
ynorm = np.abs(Y[0:round(nf/2+1)])
ynorm = (ynorm - np.min(ynorm))/(np.max(ynorm) - np.min(ynorm))
f = samplerate/2*np.linspace(0,1,round(nf/2+1))

# Plotagem da transformada e do audio.
plt.figure(1)
plt.plot(f, ynorm, linewidth=0.3)
plt.figure(2)
plt.plot(data, linewidth=0.3)
plt.show()


