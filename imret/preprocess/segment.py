# coding: utf-8
import numpy as np
import caffe
from ..color.color_palette import ColorPalette


class Segment(object):

    def __init__(self, prototxt, weights, mean=None, gpu=True):
        if gpu:
            print("Enabling GPU mode.")
            caffe.set_mode_gpu()

        print("Creating net with: \n{} \n{}".format(prototxt, weights))
        self.net = caffe.Net(prototxt, weights, caffe.TEST)
        self.color_palette = ColorPalette()

        self.transformer = caffe.io.Transformer({'data': self.net.blobs['data'].data.shape})
        self.transformer.set_transpose('data', (2, 0, 1))  # move image channels to outermost dimension
        self.transformer.set_channel_swap('data', (2, 1, 0))  # swap channels from RGB to BGR

        if mean:
            print("Loading image from file: {}".format(mean))
            data = open(mean, 'rb').read()
            blob = caffe.proto.caffe_pb2.BlobProto()
            blob.ParseFromString(data)
            mu = np.array(caffe.io.blobproto_to_array(blob))[0]
            mu = mu.mean(1).mean(1)
            self.transformer.set_mean('data', mu)  # subtract the dataset-mean value in each channel

        self.transformer.set_raw_scale('data', 255)


    @property
    def shape(self):
        return self.net.blobs['data'].data.shape[-2:]

    def segmentation(self, image):
        img = caffe.io.resize_image(image, self.shape)
        transformed_image = self.transformer.preprocess('data', img)
        self.net.blobs['data'].data[...] = transformed_image
        output = self.net.forward()
        segmentation = output[self.net.outputs[-1]][0].argmax(axis=0)
        w, h = self.shape
        paletted = np.zeros((w, h, 3), dtype=np.uint8)
        for pixel_value in np.unique(segmentation):
            x, y = np.where(segmentation == pixel_value)
            color = self.color_palette.color_from_id(class_id=pixel_value)
            paletted[x, y, :] = color
        return paletted