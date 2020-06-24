import glob
from pydub import AudioSegment
import os

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# mp3 to wav
DIR_PATH = "./sample_data/"
MP3_PATH = DIR_PATH + "*.mp3"

mp3_list = glob.glob(MP3_PATH)
for mp3 in mp3_list:
    dst = mp3.replace('mp3', 'wav')
    sound = AudioSegment.from_mp3(mp3)
    sound.export(dst, format="wav")
    
    os.remove(mp3)


# wav to png
WAV_PATH = DIR_PATH + "*.wav"
OUT_PATH = "./image/"

if not os.path.isdir(OUT_PATH):
	os.mkdir(OUT_PATH)

wav_list = glob.glob(WAV_PATH)
i = 0
for wav in wav_list:
    dst = OUT_PATH + wav.lstrip(DIR_PATH).replace('wav', 'png')
    y, sr = librosa.load(wav)

    # melspectrogram
    D = np.abs(librosa.stft(y))**2
    S = librosa.feature.melspectrogram(S=D, sr=sr)

    fig = plt.figure(figsize=(10, 4))
    S_dB = librosa.power_to_db(S, ref=np.max)
    librosa.display.specshow(S_dB, x_axis='time',
                            y_axis='mel', sr=sr,
                            fmax=8000)
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel-frequency spectrogram')
    plt.tight_layout()
    # plt.show()

    fig.savefig(dst)

    i +=1
    if i % 10 == 0:
        print(f"complete {i}th image...")


