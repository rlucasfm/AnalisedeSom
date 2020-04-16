import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
import numpy as np
import sounddevice as sd
import librosa as lb
import librosa.display as lbdisp
import noisereduce as nr
import pandas as pd
from sklearn.cluster import KMeans


filenamenoise = 'worksamplenoisy.wav'
filename = 'worksample.wav'
filereference = 'reference.wav'

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

# Carregamento do áudio de referência
sampleref, reference = wav.read(filereference)

# Filtragem do ruído ambiente
samplerate, dataraw = wav.read(filename) # Lê o samplerate e informações do arquivo
sampleratenoise, noisy_part = wav.read(filenamenoise) # Lê o samplerate e informações do arquivo de ruído
reduced_noise = nr.reduce_noise(audio_clip=dataraw, noise_clip=noisy_part, verbose=False) #Aplicação do filtro
wav.write(filename, fs, reduced_noise)

# APLICAÇÃO DA FFT E GRÁFICOS
#nf=20000
specFreq = np.fft.fft(reduced_noise) # Aplicação da FFT no áudio
specRef = np.fft.fft(reference) # Aplicação da FFT na referência


# Normalização da saída da FFT do áudio
specNorm = np.abs(specFreq[0:round(np.size(specFreq)/2+1)])
specNorm = (specNorm - np.min(specNorm))/(np.max(specNorm) - np.min(specNorm))
f = samplerate/2*np.linspace(0,1,round(np.size(specFreq)/2+1))
# Normalização da saída da FFT do áudio
specrefNorm = np.abs(specRef[0:round(np.size(specRef)/2+1)])
specrefNorm = (specrefNorm - np.min(specrefNorm))/(np.max(specrefNorm) - np.min(specrefNorm))
fRef = sampleref/2*np.linspace(0,1,round(np.size(specRef)/2+1))


# Plotagem da transformada e do audio (interesse e ruído).
plt.figure(1)
plt.plot(f, specNorm, linewidth=0.3)
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



# # Clusterização dos centros harmônicos
# freqData = pd.DataFrame(specNorm)
# freqData['Referência'] = pd.DataFrame(specrefNorm)
# freqArray = freqData.values
# kmeans = KMeans(n_clusters=8, init='k-means++', n_init=10, random_state=1234)
# freqData["clusters"] = kmeans.fit_predict(freqArray)
# freqData.rename(columns={0: 'Amostra'})
#
# centrosCluster = kmeans.cluster_centers_
#
# 
# # Plotagem dos clusters
# plt.figure(5)
# freqData.groupby("clusters").aggregate("mean").plot.bar(figsize=(10,7.5))
# plt.title("Clusters")




# Obtenção dos coeficientes de frequência Mel cepstral e sua média
libload, fs1 = lb.load(filename)
mfccs=lb.feature.mfcc(libload, fs1)
media = np.mean(mfccs)
# Normalização do MFCCS
mnorm = (mfccs - np.min(mfccs))/(np.max(mfccs) - np.min(mfccs))




# Obtenção dos Coeficientes de Predição Linear
lpc = lb.lpc(libload, 32)
# Normalização do LPC
lnorm = (lpc - np.min(lpc))/(np.max(lpc) - np.min(lpc))




# Plotagem dos LPC
plt.figure(6)
plt.plot(lpc)
plt.title('Coeficientes de Predição Linear')


# Plotagem dos MFCCS
plt.figure(figsize=(10, 4))
lbdisp.specshow(mfccs, x_axis='tempo')
plt.colorbar()
plt.title('Coeficientes de Frequência Mel Cepstrais')
plt.tight_layout()


plt.show()
