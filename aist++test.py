import pickle
#from mmae.multimodal_autoencoder import MultimodalAutoencoder
import tensorflow
import csv
import logging
import numpy as np
import requests
import librosa as lib
import io
import wave
import soundfile as sf
from mmae import MultimodalAutoencoder
import keras
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
logging.basicConfig(level=logging.INFO)

import os

x_train = []
y_train = []
with open('../all_music_wav.csv') as csvfile:
    audio = [item for sublist in list(csv.reader(csvfile)) for item in sublist]

for file in os.listdir('../keypoints3d'):
    with open('keypoints3d/'+file, 'rb') as f:
        skeldata = pickle.load(f)
        skeldata =  skeldata["keypoints3d"]
        length = len(skeldata)
        mID = 'm' + file.split("m", 1)[1][0:0+3]
        result = list(filter(lambda x: mID in x, audio))[0]
        content = requests.get(result).content
        print('audiofile loaded: ', mID)
        audiodata, samplerate = sf.read(io.BytesIO(content))
        audiodata = np.transpose(audiodata)
        mfcc = lib.feature.mfcc(y=lib.to_mono(audiodata), sr=samplerate, win_length=1600, hop_length=800)
        mfcc = (np.transpose(mfcc)[0:length])
        for timestep in mfcc: x_train.append(timestep)
        for frame in skeldata: y_train.append(frame)
        break

print('data loaded')
print('x: ',len(x_train))
print('y: ',len(y_train))
x_validation = x_train[int(len(x_train)/5):]
y_validation = y_train[int(len(y_train)/5):]


x_train = x_train[0:int(len(x_train)/5)]
y_train = y_train[0:int(len(y_train)/5)]
print('x: ',len(x_train))
print('y: ',len(y_train))
print('data split')

skel = np.transpose(y_train[1])
fig = plt.figure()
ax = plt.axes(projection='3d')

ax.scatter3D(skel[0], skel[1], skel[2])
plt.show()

data = [np.asarray(x_train), np.asarray(y_train)]
validation_data = [x_validation, y_validation]
input_shapes = [np.asarray(x_train).shape[1:], np.asarray(y_train).shape[1:]]
hidden_dims = [256, 128, 64, 16]

output_activations = ['sigmoid', 'relu']
# Name of Keras optimizer
optimizer = 'adam'
# Loss functions corresponding to a noise model for each modality
loss = ['bernoulli_divergence', 'poisson_divergence']
# Construct autoencoder network
autoencoder = MultimodalAutoencoder(input_shapes, hidden_dims,
                                    output_activations)
print('autoencoder loaded')
autoencoder.compile(optimizer, loss)
print('autoencoder compiled')
# Train model where input and output are the same

autoencoder.fit(data, epochs=100, batch_size=1,
                validation_data=validation_data)
print('autoencoder fit')








