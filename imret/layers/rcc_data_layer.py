# coding: utf-8
import os
import glob
import caffe
import random
import numpy as np
import scipy
from PIL import Image
from ..topology.rcc import Relation


class RCCDataLayer(caffe.Layer):
    """
       Load (input image, label image) pairs from PASCAL VOC
       one-at-a-time while reshaping the net to preserve dimensions.
       Use this to feed data to a fully convolutional network.
       """
    _rcc = Relation.get_preffix()
    _labels = ['above', 'below']

    def setup(self, bottom, top):
        """
        Setup data layer according to parameters:
        - ade_dir: path to ADE dir
        - split: train / val / test
        - mean: tuple of mean values to subtract
        - randomize: load in random order (default: True)
        - seed: seed for randomization (default: None / current time)

        """
        # config
        params = eval(self.param_str)
        self.rcc_dir = '/home/danilo/workspace/phd/imret/data/preposition'  # add the path to the dataset
        self.split = params['split']
        self.mean = np.array(params['mean'])
        self.random = params.get('randomize', True)
        self.seed = params.get('seed', None)

        # two tops: data, rcc) and label
        if len(top) != 3:
            raise Exception("Need to define three tops: data, rcc and label.")
        # data layers have no bottoms
        if len(bottom) != 0:
            raise Exception("Do not define a bottom.")

        # load indices for images and labels
        root_path = glob.glob(os.path.join(self.rcc_dir, '*'))
        indices = [img for dirname in root_path for img in glob.glob(os.path.join(dirname, '*.png'))]

        if self.split == 'train':
            split = int(len(indices) * .8)
            self.indices = indices[split:]
        else:
            split = int(len(indices) * .2)
            self.indices = indices[:split]

        self.idx = 0

        # make eval deterministic
        if 'train' not in self.split:
            self.random = False

        # randomization: seed and pick
        if self.random:
            random.seed(self.seed)
            self.idx = random.randint(0, len(self.indices) - 1)
        print("setuped")

    def reshape(self, bottom, top):
        img, rcc, label = self.load_image(self.indices[self.idx])
        self.data = img
        self.rcc = np.zeros((1, 8), dtype=np.float32)
        self.rcc[0, self._rcc.index(rcc)] = 1.

        self.label = np.array([self._labels.index(label)])

        # reshape tops to fit (leading 1 is for batch dimension)
        top[0].reshape(1, *self.data.shape)
        top[1].reshape(1, 8)
        top[2].reshape(1)

    def forward(self, bottom, top):
        # assign output
        top[0].data[...] = self.data
        top[1].data[...] = self.rcc
        top[2].data[...] = self.label

        # pick next input
        if self.random:
            self.idx = random.randint(0, len(self.indices) - 1)
        else:
            self.idx += 1
            if self.idx == len(self.indices):
                self.idx = 0

    def backward(self, top, propagate_down, bottom):
        pass

    def load_image(self, idx):
        """
        Load input image and preprocess for Caffe:
        - cast to float
        - switch channels RGB -> BGR
        - subtract mean
        - transpose to channel x height x width order
        """
        im = Image.open(idx)
        in_ = scipy.misc.imresize(im, (256, 256))
        in_ = np.array(in_, dtype=np.float32)
        if in_.ndim == 2:
            in_ = np.repeat(in_[:, :, None], 3, axis=2)

        in_ = in_[:, :, ::-1]
        in_ -= self.mean
        in_ = in_.transpose((2, 0, 1))

        label, rcc = os.path.basename(os.path.split(idx)[0]).split('-')
        return in_, rcc, label