import numpy as np
from torch.utils.data.dataloader import DataLoader
from src.Dataloader import AISTDataset
import os
import librosa as lib

audio_dir = '../all_music_wav'
dance_dir = '../keypoints3d'
trainset = AISTDataset(dance_dir, audio_dir)
train_dataloader = DataLoader(trainset,1, shuffle = False )
dancefiles = os.listdir(dance_dir)
audiofiles = os.listdir(audio_dir)
total  = len(dancefiles)
count = 0
for file in dancefiles:
    with open(dance_dir + '/' + file, 'rb') as f:
        mID = 'm' + file.split("m", 1)[1][0:0 + 3]
        result = list(filter(lambda x: mID in x, audiofiles))[0]
        audiodata, samplerate = lib.load(audio_dir + '/' + result)
        audiodata = np.transpose(audiodata)
        mfcc = lib.feature.mfcc(y=lib.to_mono(audiodata), sr=samplerate, win_length=1600, hop_length=800)
        mfcc = (np.transpose(mfcc)[0:360])
        with open('../mfcc.csv', 'a') as csv:
            np.savetxt(csv, mfcc, delimiter=",")




    # mean = (np.mean(data, axis=0))
    # std = (np.std(data, axis=0))
    # np.savetxt("data.csv",np.asarray(data), delimiter=",")
    count += 1
    print(count, ' of ', total)
