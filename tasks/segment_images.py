#!/usr/bin/env python
import os
import argparse
import skimage.io
import click
from imret.dataset import dataset
from imret.preprocess import segment
from imret.query import Annotation

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='IRRCC program')
    parser.add_argument('-d', '--dataset_path', action="store", default='data/datasets')
    parser.add_argument('-s', '--suffix', action="store", default='train')
    parser.add_argument('-i', '--image_path', action="store", default='data/images')
    parser.add_argument('-o', '--output_path', action="store", default='data/segmented')
    parser.add_argument('-p', '--prototxt', action="store", default='data/model/deploy.prototxt')
    parser.add_argument('-w', '--weights', action="store", default='data/model/snapshot.caffemodel')
    parser.add_argument('-n', '--names', action="store", default='data/query/name_conversion.csv')
    parser.add_argument('-l', '--annot', action='store', default='data/query')
    parser.add_argument('--info', action='store', default='data/model/objectInfo150.txt')
    parser.add_argument('--gpu', dest='gpu', action="store_true", default=True)
    parser.add_argument('--no-gpu', dest='gpu', action='store_false')
    parser.set_defaults(feature=True)
    parser.add_argument('--force', dest='force', action="store_true", default=False)
    parser.add_argument('--no-force', dest='force', action='store_false')
    parser.set_defaults(feature=True)
    params = parser.parse_args()

    dset = dataset.Dataset(params.dataset_path, params.suffix, params.image_path)
    smt = segment.Segment(prototxt=params.prototxt,
                          weights=params.weights,
                          names=params.names,
                          gpu=params.gpu,)

    annot = Annotation(os.path.join(params.annot, '{}_anno'.format(params.suffix.lower())))

    with click.progressbar(length=len(dset.images), show_pos=True, show_percent=True) as bar:
        for imname in dset.images:
            output_name = os.path.join(params.output_path, imname.replace('.jpg', '.png'))
            if os.path.exists(output_name) and not params.force:
                bar.update(1)
                continue

            image = dset.get_im_array(image=imname, rgb=True)
            if image is None:
                bar.update(1)
                continue
            segmentation = smt.segmentation(image, False)
            skimage.io.imsave(output_name, segmentation)
            bar.update(1)

    print("And we are done.")
