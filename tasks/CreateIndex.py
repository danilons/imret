#!/usr/bin/env python
import matplotlib.pyplot as plt
import numpy as np
import caffe
import scipy
import glob
import os
import json
import cv2
import itertools
import pandas as pd

caffe.set_mode_cpu()
    

def create_mask(image, pixel_value, color):
    w, h = image.shape
    m1 = np.zeros((w, h), image.dtype)
    m2 = np.zeros((w, h, 3), image.dtype)

    x, y = np.where(image == pixel_value)
    m1[x, y] = 1
    m2[x, y, :] = color
    return m1, m2

def chunks(lst, size):
    for i in range(0, len(lst), size):
        yield lst[i:i + size]

def segment_objects(fname, scene):
    image = cv2.imread(fname)
    transformed_image = transformers[scene].preprocess('data', image[:, :, (2, 1, 0)])
    nets[scene].blobs['data'].data[0] = transformed_image
    output = nets[scene].forward()
    segmented = output['fc_final_up']
    return segmented[0].argmax(axis=0)


if __name__ == "__main__":
    fnames = [('indoor', fname) for fname in glob.glob('../data/scene/test/indoor/*.jpg')] + [('outdoor', fname) for fname in glob.glob('../data/scene/test/outdoor/*.jpg')]

    with open('../data/scene/scene.json', 'r') as fp:
        predictions = json.load(fp)

    nets = {'indoor': caffe.Net('../data/scene/indoor/deploy.prototxt',
                                '../data/scene/indoor/snapshot_iter_1025.caffemodel',
                                 caffe.TEST),
            
            'outdoor': caffe.Net('../data/scene/outdoor/deploy.prototxt',
                                 '../data/scene/outdoor/snapshot_iter_4230.caffemodel',
                                 caffe.TEST)}

    transformers = {}

    with open('../data/scene/indoor/mean.binaryproto', 'rb') as f:
        blob = caffe.proto.caffe_pb2.BlobProto()
        blob.MergeFromString(f.read())
        mean_image = np.reshape(blob.data, (3, 384, 384))
        data_shape = tuple((1, 3, 384, 384))
        assert len(data_shape) == 4, 'Bad data shape.'
        mean_image = mean_image.astype(np.uint8)
        mean_image = mean_image.transpose(1, 2, 0)
        shape = list(mean_image.shape)
        mean_image = scipy.misc.imresize(mean_image, (data_shape[2], data_shape[3]))
        mean_image = mean_image.transpose(2, 0, 1)
        mean_image = mean_image.astype('float')

    transformer = caffe.io.Transformer({'data': nets['indoor'].blobs['data'].data.shape})
    transformer.set_transpose('data', (2, 0, 1))
    transformer.set_channel_swap('data', (2, 1, 0))
    transformer.set_mean('data', mean_image)

    transformers['indoor'] = transformer

    with open('../data/scene/outdoor/mean.binaryproto', 'rb') as f:
        blob = caffe.proto.caffe_pb2.BlobProto()
        blob.MergeFromString(f.read())
        mean_image = np.reshape(blob.data, (3, 384, 384))
        data_shape = tuple((1, 3, 384, 384))
        assert len(data_shape) == 4, 'Bad data shape.'
        mean_image = mean_image.astype(np.uint8)
        mean_image = mean_image.transpose(1, 2, 0)
        shape = list(mean_image.shape)
        mean_image = scipy.misc.imresize(mean_image, (data_shape[2], data_shape[3]))
        mean_image = mean_image.transpose(2, 0, 1)
        mean_image = mean_image.astype('float')

    transformer = caffe.io.Transformer({'data': nets['outdoor'].blobs['data'].data.shape})
    transformer.set_transpose('data', (2, 0, 1))
    transformer.set_channel_swap('data', (2, 1, 0))
    transformer.set_mean('data', mean_image)

    transformers['outdoor'] = transformer

    net = caffe.Net('../data/scene/preposition/deploy.prototxt',
                    '../data/scene/preposition/snapshot_iter_780.caffemodel',
                    caffe.TEST)

    with open('../data/scene/preposition/mean.binaryproto', 'rb') as f:
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

    labels = {}
    with open('../data/scene/outdoor/labels.txt', 'r') as fp:
        labels['outdoor'] = dict([line.replace('#', '').replace(':', '').strip().split() for line in fp.readlines()])

    with open('../data/scene/indoor/labels.txt', 'r') as fp:
        labels['indoor'] = dict([line.replace('#', '').replace(':', '').strip().split() for line in fp.readlines()])
        
    inv_labels = {}
    inv_labels['outdoor'] ={v: k for k, v in labels['outdoor'].items()}
    inv_labels['indoor'] ={v: k for k, v in labels['indoor'].items()}

    labels['indoor'] = {int(k): v for k, v in labels['indoor'].items()}
    labels['outdoor'] = {int(k): v for k, v in labels['outdoor'].items()}



    with open('../data/scene/preposition/labels.txt') as fp:
        labels_preposition = [line.strip().replace('_', ' ') for line in fp.readlines()]

      
    for nn, (scene, fname) in enumerate(fnames):
        print "Processed {}/{}".format(nn, len(fnames))
        name = os.path.basename(fname)
        prediction = predictions.get(name)
        if prediction is None:
            continue
        if prediction != scene:
            segmented = segment_objects(fname, scene=prediction)
        else:
            segmented = cv2.imread(os.path.join("../data/scene/output-seg/", scene + "-bw", name.replace(".jpg", ".png")), 0)
        
        if segmented is None:
            print "Segmentation {} not found".format(fname)
            continue
        
        fullname = os.path.join('../data/scene/test/', scene, name.replace('.png', '.jpg'))
        image = cv2.imread(fullname)
        if image is None:
            continue

        image = scipy.misc.imresize(image, segmented.shape, interp='bilinear')
        
        alpha = .4
        indexed = []
        images = []
        objects = []
        batch_size = 32

        for (x1, x2) in itertools.permutations(np.unique(segmented), 2):
            obj1 = labels[prediction][x1]
            obj2 = labels[prediction][x2]

            mask1, overlay1 = create_mask(segmented, x1, color=(255, 0, 0))
            mask2, overlay2 = create_mask(segmented, x2, color=(0, 0, 255))
            
            mask1 = mask1.astype(image.dtype)
            mask2 = mask2.astype(image.dtype)
            overlay1 = overlay1.astype(image.dtype)
            overlay2 = overlay2.astype(image.dtype)

            img1 = cv2.bitwise_and(image, image, mask=mask1)
            img2 = cv2.bitwise_and(image, image, mask=mask2)
            img = cv2.bitwise_or(img1, img2)

            img = img.astype(image.dtype)

            cv2.addWeighted(overlay1, alpha, img, 1 - alpha, 0, img)
            cv2.addWeighted(overlay2, alpha, img, 1 - alpha, 0, img)
            img = scipy.misc.imresize(img[:, :, (2, 1, 0)], (256, 256), interp='bilinear')
            images.append(img)
            objects.append((obj1, obj2))
        
        for n1, chunk in enumerate(chunks(zip(images, objects), size=batch_size)):
            _, channels, w, h = net.blobs['data'].data.shape
            net.blobs['data'].reshape(len(chunk), channels, w, h)

            for idx1, (im, _) in enumerate(chunk):
                transformed_image = transformer.preprocess('data', im)
                net.blobs['data'].data[idx1] = transformed_image

            output = net.forward()
            output_prob = output['softmax']
            for idx2, prob in enumerate(output_prob):
                prep = labels_preposition[prob.argmax()]
                score = prob.max()
                _, (obj1, obj2) = chunk[idx2]
                indexed.append([name, scene, prediction, prep, obj1, obj2, score])

        df = pd.DataFrame(indexed, columns=['image', 'scene', 'est', 'prep', 'obj1', 'obj2', 'score'])
        df.to_csv('scene_index.csv', index=None, mode='a')

