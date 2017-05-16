# coding: utf-8
import numpy as np
import cv2
import caffe
import pandas as pd
from ..color.color_palette import ColorPalette


class Segment(object):

    def __init__(self, prototxt, weights, names, mean=None, gpu=True, objectInfo=None):
        if gpu:
            print("Enabling GPU mode.")
            caffe.set_mode_gpu()

        print("Creating net with: \n{} \n{}".format(prototxt, weights))
        self.net = caffe.Net(prototxt, weights, caffe.TEST)
        self.color_palette = ColorPalette(name_conversion=names)
        self.transformer = caffe.io.Transformer({'data': self.net.blobs['data'].data.shape})
        self.transformer.set_transpose('data', (2, 0, 1))  # move image channels to outermost dimension
        # self.transformer.set_channel_swap('data', (2, 1, 0))  # swap channels from RGB to BGR
        # self.transformer.set_raw_scale('data', 255)
        # self.transformer.set_mean('data', mu)  # subtract the dataset-mean value in each channel
        self.df = None
        if objectInfo:
            self.df = pd.read_csv(objectInfo, sep='\t').set_index('Idx')
        if mean:
            pass
        #     print("Loading image from file: {}".format(mean))
        #     data = open(mean, 'rb').read()
        #     blob = caffe.proto.caffe_pb2.BlobProto()
        #     blob.ParseFromString(data)
        #     mu = np.array(caffe.io.blobproto_to_array(blob))[0]
        #     mu = mu.mean(1).mean(1)
        #     self.transformer.set_mean('data', mu)  # subtract the dataset-mean value in each channel
        #
        # self.transformer.set_raw_scale('data', 255)


    @property
    def shape(self):
        return self.net.blobs['data'].data.shape[-2:]

    def segmentation(self, image, annot=None):
        fx = image.shape[0] / float(self.shape[0])
        fy = image.shape[1] / float(self.shape[1])
        img = cv2.resize(image, self.shape, fx, fy)
        transformed_image = self.transformer.preprocess('data', img)
        self.net.blobs['data'].data[...] = transformed_image
        output = self.net.forward()
        segmentation = output[self.net.outputs[-1]][0].argmax(axis=0)
        w, h = self.shape
        paletted = np.zeros((w, h, 3), dtype=np.uint8)
        for pixel_value in np.unique(segmentation):
            x, y = np.where(segmentation == pixel_value)
            if self.df is None:
                # b, g, r = self.color_palette.color_from_id(class_id=pixel_value)
                r, g, b = self.color_palette.color_from_id(class_id=pixel_value)

            else:
                class_name = self.df.ix[pixel_value].Name.split(',')[0]
                if class_name in annot.names:
                    r, g, b = self.color_palette.color_from_id(class_id=pixel_value)
                else:
                    r, g, b = (0, 0, 0)

            paletted[x, y, :] = np.array([r, g, b])
        return paletted

    def weighted_image(self, image, alpha=.7):
        fx = image.shape[0] / float(self.shape[0])
        fy = image.shape[1] / float(self.shape[1])
        img = cv2.resize(image, self.shape, fx, fy)
        segmented = self.segmentation(image)
        cv2.addWeighted(segmented, alpha, img, 1 - alpha, 0, img)
        return img[:, :, (2, 1, 0)]