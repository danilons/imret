#!/usr/bin/env python2
import argparse
import os
import glob
import PIL
import caffe
import numpy as np
import scipy.misc
from sklearn.metrics import classification_report, confusion_matrix
from click import progressbar
from imret.topology.rcc import Relation


shape = (256, 256)


def path_iterator(path):
    pathname = glob.glob(os.path.join(path, '*'))
    for dirname in pathname:
        if os.path.isdir(dirname):
            fnames = glob.glob(os.path.join(dirname, '*.png'))
            for imname in fnames:
                if os.path.exists(imname):
                    preposition, rcc = os.path.basename(dirname).replace('_', ' ').split('-')
                    yield imname, rcc, preposition


def load_image(imname):
    image = PIL.Image.open(imname)
    image.load()
    return scipy.misc.imresize(image, shape, interp='bilinear')


def chunks(lst, size):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), size):
        yield lst[i:i + size]


def process(deploy_file, weights, mean_file, labels_file, impath, batch_size=32):
    _rcc = sorted([relation for relation in Relation.get_preffix() if relation.lower() != 'unk'])
    print(_rcc)

    net = caffe.Net(deploy_file, weights, caffe.TEST)

    # with open(mean_file, 'rb') as fp:
    #     blob = caffe.proto.caffe_pb2.BlobProto()
    #     blob.MergeFromString(fp.read())
    #     mean_image = np.reshape(blob.data, (3, shape[0], shape[1]))
    #
    #     data_shape = net.blobs['data'].data.shape
    #     mean_image = mean_image.astype(np.uint8)
    #     mean_image = mean_image.transpose(1, 2, 0)
    #     mean_image = scipy.misc.imresize(mean_image, (data_shape[2], data_shape[3]))
    #     mean_image = mean_image.transpose(2, 0, 1)
    #     mean_image = mean_image.astype('float')

    with open(labels_file) as fp:
        labels = [line.strip().replace('_', ' ') for line in fp.readlines()]

    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2, 0, 1))
    transformer.set_channel_swap('data', (2, 1, 0))
    # transformer.set_mean('data', mean_image)

    names, rcc, y_test = zip(*[im for im in path_iterator(impath)])

    _, channels, w, h = net.blobs['data'].data.shape

    y_pred = []
    with progressbar(length=len(names), show_pos=True, show_percent=True) as bar:
        for chunk in chunks(names, batch_size):
            net.blobs['data'].reshape(batch_size, channels, w, h)
            net.blobs['topology'].reshape(batch_size, 8)

            for index, name in enumerate(chunk):
                imarr = load_image(name)
                transformed_image = transformer.preprocess('data', imarr)
                net.blobs['data'].data[index] = transformed_image

                topology = np.zeros(len(_rcc), dtype=np.float32)
                topology[_rcc.index(rcc[index])] = 1.
                net.blobs['topology'].data[index] = topology

            output = net.forward()
            prob = output['prob']
            for p in prob[:len(chunk)]:
                y_pred.append(labels[p.argmax()])
            bar.update(len(chunk))

    np.set_printoptions(precision=2)
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='IRRCC program')

    parser.add_argument('-w', '--caffemodel', help='Path to a .caffemodel', type=str, default='snaps/fine-tunne-topology-snapshot_iter_4000.caffemodel')
    parser.add_argument('-d', '--deploy_file', help='Path to the deploy file', type=str, default='data/models/topology-preposition/deploy.prototxt')
    parser.add_argument('-i', '--image_path', help='Path to images', type=str, default='data/preposition/test')
    parser.add_argument('-m', '--mean', help='Path to a mean file (*.npy)', default='data/models/topology-preposition/mean.binaryproto')
    parser.add_argument('-l', '--labels', help='Path to a labels file', default='data/models/topology-preposition/labels.txt')
    parser.add_argument('-b', '--batch-size', type=int, default=32)
    parser.add_argument('--gpu', dest='gpu', action="store_true", default=True)
    parser.add_argument('--no-gpu', dest='gpu', action='store_false')
    parser.set_defaults(feature=True)
    opts = parser.parse_args()

    if opts.gpu:
        caffe.set_mode_gpu()
    else:
        caffe.set_mode_cpu()

    process(deploy_file=opts.deploy_file,
            weights=opts.caffemodel,
            mean_file=opts.mean,
            labels_file=opts.labels,
            impath=opts.image_path,
            batch_size=opts.batch_size)

    print("Done.")
