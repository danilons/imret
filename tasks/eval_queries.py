#!/usr/bin/env python
from __future__ import division
import argparse
import click
import numpy as np
import pandas as pd
import cytoolz
import os
import cv2
from imret.query import Annotation
from imret.dataset import Dataset
from imret.color import ColorPalette


def apk(expected, predicted, k=4317):
    retrieved = np.array([ret in expected for ret in predicted[:k]]).astype(np.int32)
    if retrieved.sum() == 0:
        return 0.0

    scores = np.arange(len(retrieved)) + 1
    valids = np.where(retrieved > 0)
    return np.sum(np.cumsum(retrieved[valids]) / scores[valids].astype(np.float32)) / float(min(len(expected), k))



cp = ColorPalette('data/query/name_conversion.csv')
freqs = {}

def filter_by_area(retrieved, noun1, noun2):
    for imname in retrieved:
        name = os.path.join('data/segmented', imname.replace('.jpg', '.png'))
        freq = freqs.get(name, None)
        if freq is None:
            img = cv2.imread(name, 0)
            if img is None:
                continue
            freq = {cp.get_name(k): v for k, v in cytoolz.frequencies(img.flatten()).items()}

        freqs[name] = freq

        if freq.get(noun1, 0) > 10000 and freq.get(noun2, 0) > 10000:
            yield imname


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='IRRCC program')
    parser.add_argument('-d', '--dataset_path', action="store", default='data/datasets')
    parser.add_argument('-i', '--image_path', action="store", default='data/images')
    parser.add_argument('-f', '--index_file', action="store", default='data/preposition/index-0.900.csv')
    parser.add_argument('-a', '--annotation', action="store", default='data/query/test_anno')
    parser.add_argument('-o', '--output_file', action="store", default='data/query/map.json')
    parser.add_argument('-q', '--queries_path', action="store", default='data/query/query_equivalence.csv')
    parser.add_argument('-t', '--threshold', action="store", default=1.0, type=float)
    parser.add_argument('--all', dest='all', action="store_true", default=True)
    parser.add_argument('--no-all', dest='all', action='store_false')
    parser.set_defaults(feature=True)
    params = parser.parse_args()
    params = parser.parse_args()

    db = Dataset(params.dataset_path, 'test', params.image_path)
    qa = Annotation(params.annotation)
    print(params.index_file)
    df = pd.read_csv(params.index_file)
    print("Intersection size is {}".format(len(set(df.image) & set(qa.imgs))))
    print("QA imgs", len(set(qa.imgs)))
    print("DB imgs", len(set(df.image)))

    imagenames = db.images

    weights = df.image.value_counts().to_dict()
    negative = len(imagenames)
    avg_precision = []
    mean_average_precision = {}
    if not params.all:
        queries = pd.read_csv(params.queries_path)
        queries.dropna(inplace=True)
        queries = dict(zip(queries['Original'], queries['Equivalent']))
        query_db = {query: ground_truth for query, ground_truth in qa.db.items() if queries.get(query)}
        assert len(query_db) < len(qa.db), "Unable to filter queries"
    else:
        query_db = qa.db

    yy_true = []
    yy_pred = []
    precision_ = dict()
    recall_ = dict()
    count = 0
    apks = []
    queries_by_obj = {}
    queries_by_prep = {}
    retrieved_by_query = {}

    # import ipdb; ipdb.set_trace()
    # with click.progressbar(length=len(query_db), show_pos=True, show_percent=True) as bar:
    for nn, query in enumerate(sorted(query_db)):
        if len(qa.db[query]) == 0:
            continue

        ground_truth = [im for im in qa.db[query] if im.endswith('.jpg')]

        noun1, preposition, noun2 = query.split('-')

        preposition = preposition.replace('_', ' ')
        noun1 = noun1.replace('cars', 'car').replace('rocks', 'rock').replace('flowers', 'flower')
        noun2 = noun2.replace('cars', 'car').replace('rocks', 'rock').replace('flowers', 'flower')
        # if preposition in ['right of', 'inside of', 'left of', 'across from']:
        #     valid = df[(df.object1 == noun1) & (df.object2 == noun2)][['image', 'score']]
        # else:
        #     valid = df[(df.object1 == noun1) & (df.object2 == noun2) & (df.preposition == preposition)][['image', 'score']]

        if preposition in ['right of', 'inside of', 'left of', 'across from']:
            valid = df[(df.object1 == noun1) & (df.object2 == noun2)][['image', 'score']]
        else:
            valid = df[(df.object1 == noun1) & (df.object2 == noun2) & (df.preposition == preposition)][['image', 'score']]

        if noun1 == noun2:
            valid = df[(df.object1 == noun1) | (df.object2 == noun2)][['image', 'score']]

        retrieved = []
        relevance = []
        if len(valid) > 0:
            valid.drop_duplicates('image', inplace=True)
            retrieved, relevance = valid['image'].values, valid['score'].values

        tp = len(set(ground_truth) & set(retrieved))
        fp = len(retrieved) - tp
        tn = negative - len(set(ground_truth))
        fn = len(ground_truth) - tp

        y_test = np.zeros(negative, dtype=np.float32)
        for img in ground_truth:
            try:
                y_test[imagenames.index(img)] = 1.
            except ValueError:
                pass

        y_scores = np.zeros(negative, dtype=np.float32)
        for n1, img in enumerate(retrieved):
            y_scores[imagenames.index(img)] = relevance[n1]

        # retrieved = list(filter_by_area(retrieved, noun1, noun2))
        apk_ = apk(ground_truth, retrieved)  #, k=len(qa.db[query]))
        apks.append(apk_)

        retrieved_by_query[query] = retrieved

        n1, p, n2 = query.split('-')
        p = p.replace('_', ' ')
        sc = "{:.2f}".format(apk_).replace('.', ',')
        # print("\\textit{{{} {} {}}} & {} & {} & {} \\\\".format(n1, p, n2, len(retrieved), len(qa.db[query]), sc))
        print("\\textit{{{} {} {}}} & {} & {} & {} & {} & {} & {} & {} \\\\".format(n1, p, n2, len(retrieved), len(qa.db[query]), sc, tp, fp, tn, fn))
        # bar.update(1)

    print("{} queries".format(count))
    print("mAP {:.4f}".format(np.mean(apks)))