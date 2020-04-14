import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
import numpy as np
import sounddevice as sd
import librosa as lb
import librosa.display as lbdisp
import noisereduce as nr


filenamenoise = 'worksamplenoisy.wav'
filename = 'worksample.wav'

# GRAVAÇÃO DO ARQUIVO
fs = 44100  # Sample rate
seconds = 6  # Duração da gravação
noisesec = 3

# Gravação do Ruído
print('Permaneça 3 segundos em silêncio para detecção do ruído...')
noiserecording = sd.rec(int(noisesec * fs), samplerate=fs, channels=1)
sd.wait()
wav.write(filenamenoise, fs, noiserecording)

# Gravação do áudio de interesse
print('Gravação em Andamento...')
myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=1)
sd.wait()  # Espera até o fim do processo de gravamento
print('Gravação completa')
wav.write(filename, fs, myrecording)  # Salva como um arquivo WAV

# Filtragem do ruído ambiente
samplerate, dataraw = wav.read(filename) # Lê o samplerate e informações do arquivo
sampleratenoise, noisy_part = wav.read(filenamenoise) # Lê o samplerate e informações do arquivo de ruído
reduced_noise = nr.reduce_noise(audio_clip=dataraw, noise_clip=noisy_part, verbose=False) #Aplicação do filtro
wav.write(filename, fs, reduced_noise)

# APLICAÇÃO DA FFT E GRÁFICOS
nf=16384
Y = np.fft.fft(reduced_noise,nf) # Aplicação da FFT no áudio

# Normalização da saída da FFT
ynorm = np.abs(Y[0:round(nf/2+1)])
ynorm = (ynorm - np.min(ynorm))/(np.max(ynorm) - np.min(ynorm))
f = samplerate/2*np.linspace(0,1,round(nf/2+1))


# Plotagem da transformada e do audio (interesse e ruído).
plt.figure(1)
plt.plot(f, ynorm, linewidth=0.3)
plt.title('Espectro de Frequências')
plt.figure(2)
plt.plot(dataraw, linewidth=0.3)
plt.title('Áudio Gravado')
plt.figure(3)
plt.title('Ruído gravado')
plt.plot(noisy_part, linewidth=0.3)
plt.figure(4)
plt.title('Áudio após o filtro')
plt.plot(reduced_noise, linewidth=0.3)


# Obtenção dos coeficientes de frequência Mel cepstral e sua média
libload, fs1 = lb.load(filename)
mfccs=lb.feature.mfcc(libload, fs1)
media = np.mean(libload)
print(media)

# Plotagem dos CFMC
plt.figure(figsize=(10, 4))
lbdisp.specshow(mfccs, x_axis='time')
plt.colorbar()
plt.title('MFCC')
plt.tight_layout()
plt.show()
