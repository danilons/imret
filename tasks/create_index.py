#!/usr/bin/env python
import os
import cv2
import caffe
import numpy as np
import itertools
import pandas as pd
import scipy
import argparse
from click import progressbar
from imret.query import StructuredQuery
from imret.dataset import Dataset
from imret.color import ColorPalette
from imret.topology import topology_relation


def create_mask(image, pixel_value, color):
    w, h = image.shape
    m1 = np.zeros((w, h), image.dtype)
    m2 = np.zeros((w, h, 3), image.dtype)

    x, y = np.where(image == pixel_value)
    m1[x, y] = 1
    m2[x, y, :] = color
    return m1, m2, np.array([x, y]).T[:, np.newaxis, :]


def chunks(lst, size):
    """Yield successive n-sized chunks from lst."""
    for i in xrange(0, len(lst), size):
        yield lst[i:i + size]

def process_image(image, segmented, sq, labels, transformer, cp, batch_size=32):
    alpha = .4
    index = []
    images = []
    objects = []
    relations = []
    image_ = scipy.misc.imresize(image, segmented.shape, interp='bilinear')

    for (x1, x2) in itertools.permutations(np.unique(segmented), 2):
        try:
            obj1 = cp.get_name(x1)
            obj2 = cp.get_name(x2)
        except IndexError:
            continue

        if obj1 in sq.names and obj2 in sq.names:
            mask1, overlay1, contour1 = create_mask(segmented, x1, color=(255, 0, 0))
            mask2, overlay2, contour2 = create_mask(segmented, x2, color=(0, 0, 255))

            img1 = cv2.bitwise_and(image_, image_, mask=mask1)
            img2 = cv2.bitwise_and(image_, image_, mask=mask2)
            img = cv2.bitwise_or(img1, img2)
            cv2.addWeighted(overlay1, alpha, img, 1 - alpha, 0, img)
            cv2.addWeighted(overlay2, alpha, img, 1 - alpha, 0, img)
            img = scipy.misc.imresize(img[:, :, (2, 1, 0)], (256, 256), interp='bilinear')
            relation = topology_relation(image.shape, {obj1: contour1, obj2: contour2})[0]['relation']
            relations.append(relation)
            images.append(img)
            objects.append((obj1, obj2))

    # feed net
    for n1, chunk in enumerate(chunks(zip(images, objects), size=batch_size)):
        _, channels, w, h = net.blobs['data'].data.shape
        net.blobs['data'].reshape(len(chunk), channels, w, h)

        for idx1, (im, _) in enumerate(chunk):
            transformed_image = transformer.preprocess('data', im)
            net.blobs['data'].data[idx1] = transformed_image

        output = net.forward()
        output_prob = output['softmax']
        for idx2, prob in enumerate(output_prob):
            prep = labels[prob.argmax()]
            score = prob.max()
            _, (obj1, obj2) = chunk[idx2]
            index.append([prep, obj1, obj2, score, relations[idx2]])
    return index

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='IRRCC program')
    parser.add_argument('-i', '--image_path', action="store", default='data/images')
    parser.add_argument('-s', '--segmentation_path', action="store", default='data/segmented')
    parser.add_argument('-n', '--names', action="store", default='data/model/topology/labels.txt')
    parser.add_argument('-d', '--deploy_file', action="store", default='data/model/topology/deploy.prototxt')
    parser.add_argument('-c', '--caffemodel', action="store", default='data/model/topology/snapshot.caffemodel')
    parser.add_argument('-m', '--mean_file', action="store", default='data/model/topology/mean.binaryproto')
    parser.add_argument('-a', '--annot_folder', action="store", default='data/datasets/Struct-Query-Test.mat')
    parser.add_argument('-o', '--output_file', action="store", default='data/preposition/index.csv')
    parser.add_argument('-p', '--dataset_path', action="store", default='data/datasets')
    parser.add_argument('-f', '--file_names_conversion', action="store", default='data/model/segmentation/name_conversion.csv')
    parser.add_argument('-b', '--batch_size', action="store", default=32, type=int)
    parser.add_argument('--gpu', dest='gpu', action="store_true", default=True)
    parser.add_argument('--no-gpu', dest='gpu', action='store_false')
    parser.set_defaults(feature=True)
    opts = parser.parse_args()

    if opts.gpu:
        print("GPU mode")
        caffe.set_mode_gpu()
    else:
        print("CPU mode")
        caffe.set_mode_cpu()

    net = caffe.Net(opts.deploy_file, opts.caffemodel, caffe.TEST)
    sq = StructuredQuery(opts.annot_folder)
    dset = Dataset(opts.dataset_path, 'test', opts.image_path)

    cp = ColorPalette(name_conversion=opts.file_names_conversion)

    with open(opts.names) as fp:
        labels = [line.strip().replace('_', ' ') for line in fp.readlines()]

    with open(opts.mean_file, 'rb') as f:
        blob = caffe.proto.caffe_pb2.BlobProto()
        blob.MergeFromString(f.read())
        mean_image = np.reshape(blob.data, (3, 256, 256))
        data_shape = tuple((1, 3, 227, 227))
        assert len(data_shape) == 4, 'Bad data shape.'
        mean_image = mean_image.astype(np.uint8)
        mean_image = mean_image.transpose(1, 2, 0)
        shape = list(mean_image.shape)
        mean_image = scipy.misc.imresize(mean_image, (data_shape[2], data_shape[3]))
        mean_image = mean_image.transpose(2, 0, 1)
        mean_image = mean_image.astype('float')

    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2, 0, 1))
    transformer.set_channel_swap('data', (2, 1, 0))
    transformer.set_mean('data', mean_image)

    index = {}
    with progressbar(length=len(dset.images), show_pos=True, show_percent=True) as bar:
        for imname in dset.images:
            segmented = cv2.imread(os.path.join(opts.segmentation_path, imname.replace('.jpg', '.png')), 0)
            image = dset.get_im_array(imname)
            if image is None or segmented is None:
                bar.update(1)
                continue

            idx = process_image(image, segmented, sq, labels, transformer, cp, opts.batch_size)
            for (prep, obj1, obj2, score, relation) in idx:
                index.setdefault('image', []).append(imname)
                index.setdefault('preposition', []).append(prep)
                index.setdefault('rcc', []).append(relation)
                index.setdefault('object1', []).append(obj1)
                index.setdefault('object2', []).append(obj2)
                index.setdefault('score', []).append(score)
            bar.update(1)

    pd.DataFrame(index).to_csv(opts.output_file)
    print("Done.")