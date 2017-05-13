#!/usr/bin/env python
from __future__ import division
import argparse
import click
import numpy as np
import pandas as pd
import cytoolz
import json
from sklearn.metrics import average_precision_score, roc_curve
from imret.query import Annotation
from imret.dataset import Dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='IRRCC program')
    parser.add_argument('-d', '--dataset_path', action="store", default='data/datasets')
    parser.add_argument('-i', '--image_path', action="store", default='data/images')
    parser.add_argument('-f', '--index_file', action="store", default='data/preposition/index.csv')
    parser.add_argument('-a', '--annotation', action="store", default='data/query/test_anno')
    parser.add_argument('-o', '--output_file', action="store", default='data/query/map.json')
    parser.add_argument('-q', '--queries_path', action="store", default='data/query/query_equivalence.csv')
    parser.add_argument('-t', '--threshold', action="store", default=3.0, type=float)
    parser.add_argument('--all', dest='all', action="store_true", default=True)
    parser.add_argument('--no-all', dest='all', action='store_false')
    parser.set_defaults(feature=True)
    params = parser.parse_args()
    params = parser.parse_args()

    db = Dataset(params.dataset_path, 'test', params.image_path)
    qa = Annotation(params.annotation)
    df = pd.read_csv(params.index_file)

    print("Intersection size is {}".format(len(set(df.image) & set(qa.imgs))))
    print("QA imgs", len(set(qa.imgs)))
    print("DB imgs", len(set(df.image)))

    # assert len(set(df.images) & set(qa.imgs)) == len(qa.imgs), "Number of images different with query annotation"
    # assert len(set(df.images) & set(qa.imgs)) == len(set(df.images)), "Number of images different with dataset"
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

    with click.progressbar(length=len(query_db), show_pos=True, show_percent=True) as bar:
        for nn, query in enumerate(query_db):
            noun1, preposition, noun2 = query.split('-')
            l1 = df[(df['object1'] == noun1)].image.tolist()
            l2 = df[(df['object2'] == noun2)].image.tolist()
            l3 = df[(df['preposition'] == preposition)].image.tolist()

            retrieved = {k: v / weights[k] for k, v in cytoolz.frequencies(l1 + l2 + l3).items()}
            valid = [(k, retrieved[k]) for k in sorted(retrieved, key=retrieved.get, reverse=True) if
                     retrieved[k] >= params.threshold]
            retrieved = []
            relevance = []
            if valid:
                retrieved, relevance = zip(*valid)

            tp = len(set(qa.db[query]) & set(retrieved))
            fp = len(retrieved) - tp
            tn = negative - len(set(qa.db[query]))
            fn = len(qa.db[query]) - tp

            y_test = np.zeros(negative, dtype=np.float32)
            for img in qa.db[query]:
                y_test[imagenames.index(img)] = 1.

            y_scores = np.zeros(negative, dtype=np.float32)
            for nn, img in enumerate(retrieved):
                y_scores[imagenames.index(img)] = relevance[nn]

            average_precision = average_precision_score(y_test, y_scores)
            if np.isnan(average_precision):
                avg_precision.append(0.0)
            else:
                avg_precision.append(average_precision)

            mean_average_precision.setdefault(query, []).append({'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
                                                                 'y_test': y_test.tolist(),
                                                                 'y_scores': y_scores.tolist()})
            bar.update(1)

    with open(params.output_file, 'w') as fp:
        json.dump(mean_average_precision, fp)

    print("mAP {:.4f}".format(np.mean(avg_precision)))
