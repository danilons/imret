#!/usr/bin/env python 
import caffe
import cv2
import numpy as np
import scipy
import cv2
import glob
import skimage
import os

if __name__ == "__main__":


    caffe.set_mode_cpu()
    net = caffe.Net('data/scene/outdoor/deploy.prototxt',
                    'data/scene/outdoor/snapshot_iter_4230.caffemodel',
                    caffe.TEST)

    with open('data/scene/outdoor/mean.binaryproto', 'rb') as f:
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


    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2, 0, 1))
    transformer.set_channel_swap('data', (2, 1, 0))
    transformer.set_mean('data', mean_image)

    with open("data/scene/outdoor/palette.txt", "r") as fp:
        colors = np.array([map(int, line.strip().split()) for line in fp.readlines()])

    def segment_image(image):
        transformed_image = transformer.preprocess('data', image)
        net.blobs['data'].data[0] = transformed_image
        output = net.forward()
        segmented = output['fc_final_up']
        segmentation = segmented[0].argmax(axis=0)
        # w, h = (384, 384)
        # paletted = np.zeros((w, h, 3), dtype=np.uint8)
        # for pixel_value in np.unique(segmentation).astype(int):
        #     x, y = np.where(segmentation == pixel_value)
        #     r, g, b = colors[pixel_value, :]
        #     paletted[x, y, :] = np.array([r, g, b])
        # return paletted
        return segmentation


    fnames = glob.glob('data/scene/test/outdoor/*.jpg')
    for nn, fname in enumerate(fnames):
        if nn % 100 == 0:
            print("Processing {}/{}".format(nn, len(fnames)))
        name = os.path.basename(fname).replace('.jpg', '.png')
        if os.path.exists(name):
            continue
            
        image = cv2.imread(fname)
        paletted = segment_image(image=image[:, :, (2, 1, 0)])
        skimage.io.imsave(os.path.join('data/scene/output-seg/outdoor-bw/', name), paletted)




