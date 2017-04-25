# coding: utf-8
import os
import h5py
import cv2


class Dataset(object):

    def __init__(self, path, suffix='train', image_path='images'):
        self.coordinates = h5py.File(os.path.join(path, 'dataset_{}.hdf5'.format(suffix)))
        self.image_path = image_path

    @property
    def images(self):
        return self.coordinates.keys()

    def ground_truth(self, image):
        gold_standard = self.coordinates[image]
        contour = {}
        for classname in gold_standard:
            bbox = gold_standard.get(classname)
            contour[classname] = np.vstack((bbox['x'], bbox['y'])).T
        return contour

    def get_im_array(self, image, rgb=False):
        if rgb:
            return cv2.imread(os.path.join(self.image_path, image))[:, :, (2, 1, 0)]
        return cv2.imread(os.path.join(self.image_path, image))

    def get_image_with_objects(self, image, obj_id=None, **kwargs):
        img = self.get_im_array(image, **kwargs)
        self.detector.get_image_with_objects(img, image, obj_id, **kwargs)
        return img