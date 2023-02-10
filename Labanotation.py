from scipy.spatial import ConvexHull
import numpy as np

def f_2(bodydata):
    return max(bodydata[10][1],bodydata[11][1])

def f_9(bodydata):
    return ConvexHull(bodydata).volume
def f_10(bodydata):
    return bodydata[0][1] - f_2(bodydata)

def dist(p1,p2):
    return np.linalg.norm(p1 - p2)
def f_1(bodydata):
    leftLower = dist(bodydata[11],bodydata[15])
    rightLower = dist(bodydata[12],bodydata[16])
    leftUpper = dist(bodydata[9],bodydata[5])
    rightUpper = dist(bodydata[6],bodydata[10])
    hands = dist(bodydata[9],bodydata[10])
    return leftLower,rightLower,leftUpper,rightUpper,hands

class Labanotation(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):

        for idx, data in enumerate(sample):
            output = []
            try:
                x = list(Labanotation.f_1(sample))
                x.append(Labanotation.f_2(sample))
                x.append(Labanotation.f_9(sample))
                x.append(Labanotation.f_10(sample))
                output.append(np.asarray(x, dtype=float))
            except ValueError as e:
                print(e)
                # os.remove(self.skel_dir + '/' + bodydata)
                # os.execv(sys.executable, ['python'] + sys.argv)
                continue
                # os.remove(self.skel_dir + '/' + bodydata)
                # os.execv(sys.executable, ['python'] + sys.argv)

        return output


