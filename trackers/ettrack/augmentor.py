# set of functor classes that augment the data
import random
import numpy as np
import abc
from loguru import logger
import torch
#data x,y,w,h,delta x, delta y, delta w, delta h
# batch_abs [20,271,8]  frame_number, pedesetrian number, positions/delta position
class Augmentor(abc.ABC):
    """
    base class for augmentation methods.
    Override process(tracks).
    __call__ handles the probability of it been applied, so don't call process directly unless you want it always to be
    applied, use the class as a functor instead.
    eg
    f = FlipLR(probability=0.5)
    flip = f(test)

    tracks are x,y,w,h, delta x, delta y, delta w, delta h
    """
    def __init__(self, probability=0.5):
        self.probability = probability

    @abc.abstractmethod
    def process(self, tracks):
        pass

    def __call__(self, tracks):
        if self.do_augmentation():
            return self.process(tracks)
        else:
            return tracks


    def do_augmentation(self):
        if random.random() < self.probability:
            return True
        else:
            return False

class Rotate(Augmentor):
    def __init__(self, max_half_angle=np.pi, probability=0.5):
        super().__init__(probability)
        self.max_half_angle = max_half_angle

    def process(self, tracks):
        min_x = torch.min(tracks[:, :, 0], 0)
        max_x  =torch.max(tracks[:, :, 0], 0)
        mean_x = np.mean(tracks[:, :, 0], 0)
        #mean_x = (max_x-min_x)/2
        tracks[:, :, 0] -= mean_x
        tracks[:, :, 0] = - tracks[:, :, 0]

        mean_y = torch.mean(tracks[:, :, 1], 0)
        tracks[:, :, 1] -= mean_y
        tracks[:, :, 1] = - tracks[:, :, 1]
        theta = -self.max_half_angle + 2*self.max_half_angle*np.random.random()
        logger.debug(f'theta: {theta}')
        ct = np.cos(theta)
        st = np.sin(theta)
        tracks[:, :, 0] =  tracks[:, :, 0] * ct +  tracks[:, :, 1]*st
        tracks[:, :, 1] = tracks[:, :, 0] * st + tracks[:, :, 1] * ct
        tracks[:, :, 0] += mean_x
        tracks[:, :, 1] += mean_y

        #todo deltas

        return tracks

class Speed(Augmentor):
    def __init__(self, max_speed=2.0, probability=0.5):
        super().__init__(probability)
        self.max_speed = max_speed

    def process(self, tracks):
        pass


class FlipTime(Augmentor):
    def process(self, tracks):
        tracks = torch.flip(tracks, (0,))
        tracks[:,4:] = - tracks[:,4:]
        return tracks

class FlipLR(Augmentor):
    def process(self, tracks):
        #mean_x = np.mean(tracks[:, :, 0], axis=0)
        mean_x = torch.mean(tracks[ :, 0], 0)
        tracks[ :, 0] -= mean_x
        tracks[ :, 0] = - tracks[:, 0]
        tracks[ :, 4] = - tracks[:, 4]  # flip delta x
        tracks[ :, 0] += mean_x
        return tracks


class FlipUD(Augmentor):
    def process(self, tracks):
        mean_y = torch.mean(tracks[:, 1], 0)
        tracks[:, 1] -= mean_y
        tracks[:, 1] = - tracks[:, 1]
        tracks[:, 5] = - tracks[:, 5]  # flip delta y
        tracks[:, 1] += mean_y
        return tracks



if __name__ == '__main__':

    test = np.zeros((5,2,8))
    for ind in range(5):
        test[ind,0,:] = [ind,0,ind*2,0,0,0,0,0]
        test[ind, 1, :] = [-ind, 0, -ind * 2, 0, 0, 0, 0, 0]

    mean_x = np.mean(test[:,:,0],axis=0)
    test2 = test.copy()
    test2[:,:,0] = test[:,:,0]-mean_x
    print(test)
    f = FlipLR()
    flip = f(torch.tensor(test))
    print('processed...')
    print(flip)