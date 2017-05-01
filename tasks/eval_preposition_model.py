#!/usr/bin/env python2
# Copyright (c) 2015-2017, NVIDIA CORPORATION.  All rights reserved.

"""
Classify an image using individual model files
Use this script as an example to build your own tool
"""

import argparse
import os
import glob
import PIL
import caffe
import numpy as np
import scipy.misc
from sklearn.metrics import classification_report, confusion_matrix
from click import progressbar

shape = (256, 256)


def path_iterator(path):
    pathname = glob.glob(os.path.join(path, '*'))
    for dirname in pathname:
        if os.path.isdir(dirname):
            fnames = glob.glob(os.path.join(dirname, '*.png'))
            for imname in fnames:
                if os.path.exists(imname):
                    yield imname, os.path.basename(dirname).replace('_', ' ')


def load_image(imname):
    image = PIL.Image.open(imname)
    image.load()
    return scipy.misc.imresize(image, shape, interp='bilinear')


def chunks(lst, size):
    """Yield successive n-sized chunks from lst."""
    for i in xrange(0, len(lst), size):
        yield lst[i:i + size]


def process(deploy_file, weights, mean_file, labels_file, impath, batch_size=32):

    net = caffe.Net(deploy_file, weights, caffe.TEST)

    with open(mean_file, 'rb') as fp:
        blob = caffe.proto.caffe_pb2.BlobProto()
        blob.MergeFromString(fp.read())
        mean_image = np.reshape(blob.data, (3, shape[0], shape[1]))

        data_shape = net.blobs['data'].data.shape
        mean_image = mean_image.astype(np.uint8)
        mean_image = mean_image.transpose(1, 2, 0)
        mean_image = scipy.misc.imresize(mean_image, (data_shape[2], data_shape[3]))
        mean_image = mean_image.transpose(2, 0, 1)
        mean_image = mean_image.astype('float')

    with open(labels_file) as fp:
        labels = [line.strip().replace('_', ' ') for line in fp.readlines()]

    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2, 0, 1))
    transformer.set_channel_swap('data', (2, 1, 0))
    transformer.set_mean('data', mean_image)

    names, y_test = zip(*[im for im in path_iterator(impath)])

    _, channels, w, h = net.blobs['data'].data.shape

    y_pred = []
    with progressbar(length=len(names), show_pos=True, show_percent=True) as bar:
        for chunk in chunks(names, batch_size):
            net.blobs['data'].reshape(len(chunk), channels, w, h)

            for index, name in enumerate(chunk):
                imarr = load_image(name)
                transformed_image = transformer.preprocess('data', imarr)
                net.blobs['data'].data[index] = transformed_image

            output = net.forward()
            prob = output['softmax']
            for p in prob:
                y_pred.append(labels[p.argmax()])
            bar.update(len(chunk))

    np.set_printoptions(precision=2)
    print classification_report(y_test, y_pred)
    print confusion_matrix(y_test, y_pred)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='IRRCC program')
    # Positional arguments
    parser.add_argument('-w', '--caffemodel', help='Path to a .caffemodel', type=str)
    parser.add_argument('-d', '--deploy_file', help='Path to the deploy file', type=str)
    parser.add_argument('-i', '--image_path', help='Path to images', type=str)

    # Optional arguments
    parser.add_argument('-m', '--mean', help='Path to a mean file (*.npy)')
    parser.add_argument('-l', '--labels', help='Path to a labels file')
    parser.add_argument('-b', '--batch-size', type=int)
    parser.add_argument('--nogpu', action='store_true', help="Don't use the GPU")

    opts = parser.parse_args()

    if opts.nogpu:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()

    process(deploy_file=opts.deploy_file,
            weights=opts.caffemodel,
            mean_file=opts.mean,
            labels_file=opts.labels,
            impath=opts.image_path,
            batch_size=opts.batch_size)

    print("Done.")