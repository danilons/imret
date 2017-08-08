# coding: utf-8
import numpy as np
import cv2
import caffe
import pandas as pd
from ..color.color_palette import ColorPalette


class Segment(object):

    def __init__(self, prototxt, weights, names, gpu=False):
        if gpu:
            print("Enabling GPU mode.")
            caffe.set_mode_gpu()

        print("Creating net with: \n{} \n{}".format(prototxt, weights))
        self.net = caffe.Net(prototxt, weights, caffe.TEST)
        self.color_palette = ColorPalette(name_conversion=names)
        self.transformer = caffe.io.Transformer({'data': self.net.blobs['data'].data.shape})
        self.transformer.set_transpose('data', (2, 0, 1))  # move image channels to outermost dimension

    @property
    def shape(self):
        return self.net.blobs['data'].data.shape[-2:]

    def segmentation_by_thresholds(self, image, thresholds):
        fx = image.shape[0] / float(self.shape[0])
        fy = image.shape[1] / float(self.shape[1])
        img = cv2.resize(image, self.shape, fx, fy)
        transformed_image = self.transformer.preprocess('data', img)
        self.net.blobs['data'].data[...] = transformed_image
        output = self.net.forward()[self.net.outputs[-1]][0]
        images = {}
        for threshold in thresholds:
            x, y = np.where(output.max(axis=0) > threshold)
            segmentation = np.zeros(self.shape, dtype=np.uint8)
            segmentation[x, y] = output[:, x, y].argmax(axis=0)
            images[threshold] = segmentation
        return images

    def segmentation(self, image, return_paletted=True, threshold=0.5, **kwargs):
        fx = image.shape[0] / float(self.shape[0])
        fy = image.shape[1] / float(self.shape[1])
        img = cv2.resize(image, self.shape, fx, fy)
        transformed_image = self.transformer.preprocess('data', img)
        self.net.blobs['data'].data[...] = transformed_image
        output = self.net.forward()[self.net.outputs[-1]][0]
        x, y = np.where(output.max(axis=0) > threshold)
        segmentation = np.zeros(self.shape, dtype=np.uint8)
        segmentation[x, y] = output[:, x, y].argmax(axis=0)
        if not return_paletted:
            return segmentation

        w, h = self.shape
        paletted = np.zeros((w, h, 3), dtype=np.uint8)
        for pixel_value in np.unique(segmentation):
            x, y = np.where(segmentation == pixel_value)
            r, g, b = self.color_palette.color_from_id(class_id=pixel_value)
            paletted[x, y, :] = np.array([r, g, b])
        return paletted

    def weighted_image(self, image, alpha=.7, **kwargs):
        fx = image.shape[0] / float(self.shape[0])
        fy = image.shape[1] / float(self.shape[1])
        img = cv2.resize(image, self.shape, fx, fy)
        segmented = self.segmentation(image, **kwargs)
        cv2.addWeighted(segmented, alpha, img, 1 - alpha, 0, img)
        return img[:, :, (2, 1, 0)]