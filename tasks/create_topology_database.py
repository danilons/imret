#!/usr/bin/env python
import os
import sys
import glob
import click
import argparse
import traceback
import h5py
import numpy
import signal
import re
from multiprocessing import Pool
from imret.dataset import Dataset
from imret.topology import topology_relation
from imret.color import ColorPalette


def _topology_relation(objects, image, shape):
    # return image, topology_relation(shape, objects)
    print("Im running.")
    relation = topology_relation(shape, objects)
    if not relation:
        print("No topology for image: {}".format(image))
    return image, relation


def init_worker():
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def store_topology(dset, mode, output_file, classnames):
    fname = output_file.replace('#', 'segment')
    if os.path.exists(fname):
        print("Removing file: {}".format(fname))
        os.remove(fname)

    pool = Pool(processes=8, initializer=init_worker)
    processes = []
    for imname in dset.images:
        objects = dset.ground_truth(imname) if mode == 'train' else dset.get_objects(imname, classnames)
        imarray = dset.get_im_array(imname)
        if imarray is None:
            continue
        # processes.append(pool.apply_async(func=_topology_relation, args=(objects, imname, imarray.shape[:2],)))
        processes.append((_topology_relation, objects, imname, imarray.shape[:2]))

    content = {}
    with click.progressbar(length=len(processes), show_pos=True, show_percent=True) as bar:
        try:
            for process in processes:
                try:
                    f, obj, imnam, shape = process
                    image, relation = f(obj, imnam, shape)
                    # image, relation = process.get()
                    content[image] = relation
                except KeyboardInterrupt:
                    raise
                except:
                    traceback.print_exc()
                bar.update(1)
        except KeyboardInterrupt:
            print "Caught KeyboardInterrupt, terminating workers"
            pool.terminate()
            pool.join()
            sys.exit(0)

    print("Saving")
    hdf5 = h5py.File(fname, 'w')
    with click.progressbar(length=len(content), show_pos=True, show_percent=True) as bar:
        for (image, relations) in content.iteritems():
            if image in hdf5:
                continue

            try:
                hdf5_group = hdf5.create_group(image)
            except RuntimeError:
                continue

            for topology in relations:
                try:
                    objects = '-'.join(topology['objects']).strip()
                    if objects in hdf5_group:
                        continue
                    hdf5_group.create_group(objects)
                    hdf5_group[objects].create_dataset('contours1', data=topology['contours'][0], compression="gzip")
                    hdf5_group[objects].create_dataset('contours2', data=topology['contours'][1], compression="gzip")
                    hdf5_group[objects]['relation'] = numpy.string_(topology['relation'])
                except ValueError:
                    print("Error when processing image {} and {}".format(image, objects))
                except RuntimeError:
                    print("Runtime error when processing image {} and {}".format(image, objects))
                    continue
                except IOError:
                    print("IOError error when processing image {} and {}".format(image, objects))
                except TypeError:
                    print("Fucking error I have no idea. image {} and {}".format(image, objects))
            bar.update(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='IRRCC program')
    parser.add_argument('-d', '--dataset_path', action="store", default='data/datasets')
    parser.add_argument('-i', '--image_path', action="store", default='data/images')
    parser.add_argument('-n', '--names', action="store", default='data/model/segmentation/name_conversion.csv')
    params = parser.parse_args()

    color_palette = ColorPalette(name_conversion=params.names)
    files = glob.glob(os.path.join(params.dataset_path, 'dataset_test*.hdf5'))
    for fname in files:
        suffix = re.search('(train|test)', fname.lower()).group()
        print("Start processing file: {}, mode: {}".format(fname, suffix))

        basename = os.path.basename(fname)
        filename, _ = os.path.splitext(basename)
        mode = filename.replace('dataset_', '')
        print("Processing mode: {}".format(mode))

        dset = Dataset(params.dataset_path, mode, params.image_path)
        output_file = fname.replace('dataset_', 'topology_#_')
        store_topology(dset, mode, output_file, classnames=color_palette.names)

    print("Done.")