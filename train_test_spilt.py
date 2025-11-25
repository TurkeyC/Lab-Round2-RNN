import glob
import shutil
import random

wav_files = glob.glob("./dataset/" + "*.wav")
random.shuffle(wav_files)

train_percent = 0.7
len_train = round(len(wav_files) * 0.7)
train_wav_files = wav_files[:len_train]
test_wav_files = wav_files[len_train:]

for file in train_wav_files:
    file_name = file.split("\\")[1]
    shutil.copy(file, "./train/" + file_name)

for file in test_wav_files:
    file_name = file.split("\\")[1]
    shutil.copy(file, "./test/" + file_name)
