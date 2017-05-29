#!/usr/bin/env python
import os
import argparse
import skimage.io
import click
import cv2
import scipy.misc
import numpy as np
from imret.dataset import dataset
from imret.color import ColorPalette

size = (384, 384)

def create_label(scale, ground_truth, rgb=True):
    if rgb:
        label = np.zeros((w, h, 3), dtype=np.uint8)
    else:
        label = np.zeros((w, h), dtype=np.uint8)

    for (classname, contour) in ground_truth.items():
        try:
            color = color_palette[classname] if rgb else color_palette.class_id(classname)
        except ValueError:
            continue

        cnt = contour * scale
        cv2.drawContours(label, [cnt.astype(np.int32)], -1, color, -1)
    return label

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='IRRCC program')
    parser.add_argument('-d', '--dataset_path', action="store", default='data/datasets')
    parser.add_argument('-s', '--suffix', action="store", default='train')
    parser.add_argument('-i', '--image_path', action="store", default='data/images')
    parser.add_argument('-r', '--resized_path', action="store", default='data/imgs/resized')
    parser.add_argument('-l', '--label_path', action="store", default='data/imgs/labels')
    parser.add_argument('-m', '--mark_path', action="store", default='data/imgs/mark')
    parser.add_argument('-n', '--names', action="store", default='data/query/name_conversion.csv')
    parser.set_defaults(feature=True)
    params = parser.parse_args()

    dset = dataset.Dataset(params.dataset_path, params.suffix, params.image_path)
    color_palette = ColorPalette(name_conversion=params.names)

    with click.progressbar(length=len(dset.images), show_pos=True, show_percent=True) as bar:
        for imname in dset.images:
            image = dset.get_im_array(image=imname, rgb=True)
            if image is None:
                bar.update(1)
                continue

            w, h = size
            w1, h1 = image.shape[:2]
            fy = w / float(w1)
            fx = h / float(h1)
            scale = np.array([fx, fy])

            ground_truth = dset.ground_truth(imname)

            img_name = os.path.join(params.resized_path, imname.replace('.jpg', '.png'))
            if not os.path.exists(img_name):
                rsz = scipy.misc.imresize(image, size)
                skimage.io.imsave(img_name, rsz)

            label_name = os.path.join(params.label_path, imname.replace('.jpg', '.png'))
            if not os.path.exists(label_name):
                label = create_label(scale, ground_truth, rgb=True)
                skimage.io.imsave(label_name, label)

            mark_name = os.path.join(params.mark_path, imname.replace('.jpg', '.png'))
            if not os.path.exists(mark_name):
                label = create_label(scale, ground_truth, rgb=False)
                skimage.io.imsave(mark_name, label)

            bar.update(1)

    print("And we are done.")
