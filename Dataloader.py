import os
import pickle
import librosa as lib
import numpy as np
from torch.utils.data import Dataset
import Labanotation
import sys
from SeqSplit import SplitSeq


class AISTDataset(Dataset):
    def __init__(self,skeletal_dir, audio_dir, transform=None,target_transform=None, seq_length = 10):
        self.skel_dir = skeletal_dir
        self.audio_dir = audio_dir
        self.transform = transform
        self.target_transform = target_transform
        self.seq_len = seq_length
        self.maxi = 0
        self.mini = 9999999


    def __len__(self):
        return len(os.listdir(self.skel_dir))

    def __getitem__(self, idx):
        audio = os.listdir(self.audio_dir)
        bodydata = os.listdir(self.skel_dir)[idx]

        with open(self.skel_dir + '/' + bodydata, 'rb') as f:
            skeldata = pickle.load(f)
            skeldata = skeldata["keypoints3d"][:360]


            if len(skeldata) < 360:
                print(bodydata, len(skeldata))
                os.remove(self. skel_dir + '/' + bodydata)
                sys.exit(0)
            mID = 'm' + bodydata.split("m", 1)[1][0:0 + 3]
            result = list(filter(lambda x: mID in x, audio))[0]
            audiodata, samplerate = lib.load(self.audio_dir + '/' + result)
            audiodata = np.transpose(audiodata)
            mfcc = lib.feature.mfcc(y=lib.to_mono(audiodata), sr=samplerate, win_length=1600, hop_length=800)
            mfcc = (np.transpose(mfcc)[0:360])
            output = []

            # if self.transform:
            #     sample = self.transform(sample)
            for idx,data in enumerate(skeldata):
                try:
                    x = list(Labanotation.f_1(data))
                    x.append(Labanotation.f_2(data))
                    x.append(Labanotation.f_9(data))
                    x.append(Labanotation.f_10(data))
                    output.append(np.asarray(x, dtype=float))
                except ValueError as e:
                    print (bodydata,len(skeldata), e)
                    # os.remove(self.skel_dir + '/' + bodydata)
                    # os.execv(sys.executable, ['python'] + sys.argv)
                    continue
                    # os.remove(self.skel_dir + '/' + bodydata)
                    # os.execv(sys.executable, ['python'] + sys.argv)



        dX,dY = SplitSeq(output,self.seq_len)
        mX, mY = SplitSeq(mfcc, self.seq_len)

        def DNormalize(input):
            mean = np.array([66.50385, 68.32886, 38.71090, 38.70888, 65.33032, 166.70784, 102561.16923, 33.85368])
            std = np.array([77.64474, 448.32667, 11.02547, 9.69077, 27.02228, 22.69702, 1242051.78412, 20.39479])
            return (np.asarray(input)-mean)/std
        def MNormalize(input):
            mean = np.array([-99.25795,65.42054,12.52503,35.54624,9.72540,23.42018,-0.63370,12.45946,-3.07848,9.48500,-3.29034,9.63652,-2.80752,9.05766,-3.43942,7.35616,-3.74244,7.79904,-3.22722,6.29393])
            std = np.array([98.89870,43.21727,28.85738,18.33283,14.70160,14.93426,11.50901,11.88072,9.93609,11.73378,10.21012,9.96216,9.39796,9.23207,8.58922,8.56636,7.96165,8.51490,8.08458,9.39848])
            return (np.asarray(input)-mean)/std


        return ((DNormalize(dX), DNormalize(dY)),(MNormalize(mX),MNormalize(mY)))


