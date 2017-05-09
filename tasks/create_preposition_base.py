#!/usr/bin/env python
from sys import stdout
import os
import cv2
import numpy as np
import click
import argparse
import re
from imret.dataset import Dataset
from imret.query import StructuredQuery


def save_relations(dataset, sq, output_folder, alpha=.4):
    with click.progressbar(length=5, show_pos=True, show_percent=True) as bar:
        for query_type in sq.query_types:
            stdout.write("\n")
            for n1, query in enumerate(sq[query_type]):
                name = query['name'].split('&')
                for noun in name:
                    text = noun.split(',')[-1]
                    obj1, prep, obj2 = text.split('-')
                    imgs = [dataset.images[idx] for idx, ranked in enumerate(query['rank']) if ranked]
                    for nn, imname in enumerate(imgs):
                        step = "{:3d}/{:3d} Query type: {:1s} name: {:30s} step: {:3d}/{:3d}".format(n1, len(sq[query_type]),
                                                                                   query_type, text[:30], nn, len(imgs))
                        stdout.write("\r%s" % step)
                        stdout.flush()
                        contours = dataset.ground_truth(imname)
                        imarray = dataset.get_im_array(imname)
                        contour1 = contours.get(obj1)
                        contour2 = contours.get(obj2)
                        if contour1 is not None and contour2 is not None and imarray is not None:
                            mask = np.zeros(imarray.shape[:2], imarray.dtype)
                            m1 = np.zeros(imarray.shape, imarray.dtype)
                            m2 = np.zeros(imarray.shape, imarray.dtype)

                            cv2.drawContours(mask, [contour1.astype(np.int32)], -1, 255, -1)
                            cv2.drawContours(mask, [contour2.astype(np.int32)], -1, 255, -1)
                            cv2.drawContours(m1, [contour1.astype(np.int32)], -1, (255, 0, 0), -1)
                            cv2.drawContours(m2, [contour2.astype(np.int32)], -1, (0, 0, 255), -1)

                            objected = cv2.bitwise_and(imarray, imarray, mask=mask)
                            cv2.addWeighted(m1, alpha, objected, 1 - alpha, 0, objected)
                            cv2.addWeighted(m2, alpha, objected, 1 - alpha, 0, objected)
                            dirname = os.path.join(output_folder, prep)
                            imgfile = "{}-{}-{}.png".format(imname.replace('.jpg', ''), obj1, obj2)
                            if not os.path.exists(dirname):
                                os.makedirs(dirname)
                            cv2.imwrite(os.path.join(dirname, imgfile), objected)
            bar.update(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='IRRCC program')
    parser.add_argument('-d', '--dataset_path', action="store", default='data/datasets')
    parser.add_argument('-i', '--image_path', action="store", default='data/images')
    parser.add_argument('-n', '--names', action="store", default='data/name_conversion.csv')
    parser.add_argument('-s', '--sq_file', action="store", default='data/datasets/Struct-Query-Train.mat')
    parser.add_argument('-o', '--output', action="store", default='data')
    params = parser.parse_args()

    suffix = re.search('(train|test)', params.sq_file.lower()).group()
    print("running set: {}".format(suffix))
    dataset = Dataset(params.dataset_path, params.train, params.image_path)
    sq = StructuredQuery(params.sq_file)
    save_relations(dataset, sq, os.path.join(params.output, params.train))
    print("Done.")