# coding: utf-8
import os
from glob import glob

class Annotation:
    def __init__(self, path):
        self.db = {}
        self.imgs = {}
        for folder in glob(os.path.join(path, '*')):
            for fname in glob(os.path.join(folder, '*.txt')):
                key = os.path.basename(folder)
                with open(fname, 'r') as fp:
                    self.db[key] = [line.strip() for line in fp.readlines() if line.strip().endswith('.jpg')]

        names1, preposition, names2 = zip(*(query.split('-') for query in self.db))
        self.names = list(set(names1) | set(names2))
        self.preposition = list(set(preposition))

        for k, imlist in self.db.iteritems():
            for imname in imlist:
                self.imgs.setdefault(imname, []).append(k)

        self.objects = []
        self.objects_by_image = {}
        for img, queries in self.imgs.iteritems():
            for query in queries:
                noun1, _, noun2 = query.split('-')
                self.objects_by_image.setdefault(img, set()).add(noun1)
                self.objects_by_image.setdefault(img, set()).add(noun2)
                self.objects.append(noun1)
                self.objects.append(noun2)

        self.objects = list(set(self.objects))
        self.labels = {'books': 'book', 'cars': 'car', 'flowers': 'flower', 'seats': 'seat',
                       'rocks': 'rock', 'cupboard': 'cabinet', 'gate': 'bar'}

    def __getitem__(self, imname):
        return self.imgs.get(imname, [])

    @property
    def normalized_names(self):
        return [self.labels.get(name, name) for name in self.names]

    def features(self):
        for img, queries in self.imgs.iteritems():
            for query in queries:
                noun1, preposition, noun2 = query.split('-')
                yield (img, noun1, preposition, noun2)