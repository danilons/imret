#!/usr/bin/env python
import os
import glob
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='IRRCC program')
    parser.add_argument('-s', '--source', action="store", default='data/preposition/test')
    parser.add_argument('-d', '--destiny', action="store", default='data/test/preposition')
    params = parser.parse_args()
    
    db = {}
    folders = glob.glob(os.path.join(params.source, '*'))
    for folder in folders:
        files = glob.glob(os.path.join(folder, '*.png'))
        name = os.path.basename(folder).split('-')[0]
        for fn in files:
             db.setdefault(name, []).append(fn)

    for prep, imgs in db.items():
        dst = os.path.join(params.destiny, prep)
        if not os.path.exists(dst):
            os.makedirs(dst)

        for img in imgs:
            imname = os.path.join(dst, os.path.basename(img))
            if not os.path.exists(imname):
                os.symlink(os.path.join(os.getcwd(), img), imname)