# coding: utf-8
import os
import h5py
import cv2
import numpy as np

class Dataset(object):

    def __init__(self, path, suffix='train', image_path='images'):
        file_name = os.path.join(path, 'dataset_{}.hdf5'.format(suffix))
        if not os.path.exists(file_name):
            raise ValueError("Dataset {} not found.".format(file_name))
        self.coordinates = h5py.File(file_name, 'r')
        self.image_path = image_path

    @property
    def images(self):
        return self.coordinates.keys()

    def ground_truth_objects(self, image):
        return self.coordinates[image].keys()

    def ground_truth(self, image):
        gold_standard = self.coordinates[image]
        contour = {}
        for classname in gold_standard:
            bbox = gold_standard.get(classname)
            contour[classname] = np.vstack((bbox['x'], bbox['y'])).T
        return contour

    def get_im_array(self, image, rgb=False):
        img = cv2.imread(os.path.join(self.image_path, image))
        if img is None:
            return None
        return img if not rgb else img[:, :, (2, 1, 0)]

    def get_objects(self, image, classnames):
        imname = image.replace('.jpg', '.png')
        objects = cv2.imread(os.path.join(self.image_path, imname), 0)
        if objects is None:
            return {}

        classes = np.unique(objects)
        segmentation = {}

        for k in classes:
            x, y = np.where(objects == k)
            img = np.zeros(objects.shape[:2], dtype=np.uint8)
            img[x, y] = 255.
            _, binary = cv2.threshold(img, 127, 255, 0)
            _, contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            biggest = sorted([(len(cnt), nn) for nn, cnt in enumerate(contours)], key=lambda x: x[0], reverse=True)
            _, idx = biggest[0]
            segmentation[classnames[k]] = contours[idx]

        return segmentation