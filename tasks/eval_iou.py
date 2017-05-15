#!/usr/bin/env python
from __future__ import division
import os
import re
import argparse
import cv2
import click
import numpy as np
import skimage.io
from imret.dataset import Dataset
from imret.color import ColorPalette
from imret.query import Annotation

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='IRRCC program')
    parser.add_argument('-d', '--dataset_path', action="store", default='data/datasets')
    parser.add_argument('-i', '--image_path', action="store", default='data/images')
    parser.add_argument('-s', '--segmented_path', action="store", default='data/segmented')
    parser.add_argument('-n', '--names', action="store", default='data/query/name_conversion.csv')
    parser.add_argument('-a', '--annot', action="store", default='data/query/test_anno/')
    params = parser.parse_args()

    iou = {}
    color_palette = ColorPalette(name_conversion=params.names)
    dset = Dataset(params.dataset_path, 'test', params.image_path)
    annot = Annotation(params.annot)
    with click.progressbar(length=len(dset.images), show_pos=True, show_percent=True) as bar:
        for imname in dset.images:
            segmented_name = os.path.join(params.segmented_path, imname.replace('.jpg', '.png'))
            try:
                segmented = skimage.io.imread(segmented_name)
            except IOError:
                bar.update(1)
                continue

            img = dset.get_im_array(imname)
            if img is None:
                bar.update(1)
                continue

            w, h = img.shape[:2]
            fy = segmented.shape[0] / w
            fx = segmented.shape[1] / h
            scale = np.array([fx, fy])

            ground_truth = dset.ground_truth(imname)
            for object_name, contour in ground_truth.items():
                name = re.match('\D+', object_name).group()
                name = color_palette.class_names.get(name, name)
                try:
                    class_id = color_palette.class_id(name)
                    color = color_palette[name]
                except ValueError:
                    continue

                if name not in annot.normalized_names:
                    continue

                img1 = cv2.inRange(segmented, color, color)

                img2 = np.zeros(segmented.shape[:2], dtype=np.uint8)
                cnt = contour * scale
                cv2.drawContours(img2, [cnt.astype(np.int32)], -1, 255, -1)

                intersection = cv2.bitwise_and(img1, img2)
                union = cv2.bitwise_or(img1, img2)
                score = intersection.sum() / union.sum()

                iou.setdefault(name, []).append(score)

            bar.update(1)

    total = 0
    for object_name, scores in sorted(iou.items(), key=lambda x: x[0]):
        score = sum(scores) / len(scores)
        total += score
        print("{:20s} {:.2f}%".format(object_name, score * 100))
    print("{:20s} {:.2f}%".format('Total mean', (total / len(iou)) * 100))

    print("And we are done.")
