#!/usr/bin/env python
from __future__ import division
import argparse
import random
import click
import numpy as np
import pandas as pd
import cytoolz
import matplotlib.pyplot as plt
import matplotlib.colors
import json
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score, roc_curve, label_ranking_average_precision_score
from imret.query import Annotation
from imret.dataset import Dataset


def apk(expected, predicted, k=10):
    retrieved = np.array([ret in expected for ret in predicted]).astype(np.int32)
    scores = np.arange(len(retrieved)) + 1
    valids = np.where(retrieved > 0)
    if retrieved.sum() == 0:
        return 0.0

    # return np.sum(np.cumsum(retrieved[valids]) / scores[valids].astype(np.float32)) / float(len(expected))
    return np.sum(np.cumsum(retrieved[valids]) / scores[valids].astype(np.float32)) / float(len(retrieved))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='IRRCC program')
    parser.add_argument('-d', '--dataset_path', action="store", default='data/datasets')
    parser.add_argument('-i', '--image_path', action="store", default='data/images')
    parser.add_argument('-f', '--index_file', action="store", default='data/preposition/index.csv')
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

    with click.progressbar(length=len(query_db), show_pos=True, show_percent=True) as bar:
        for nn, query in enumerate(query_db):
            if len(qa.db[query]) == 0:
                print "invalid query {}".format(query)
                continue

            noun1, preposition, noun2 = query.split('-')
            preposition = preposition.replace('_', ' ')
            # preposition = preposition.replace('below', 'under').replace('across from', 'under')
            noun1 = noun1.replace('cars', 'car').replace('rocks', 'rock').replace('flowers', 'flower')
            noun2 = noun2.replace('cars', 'car').replace('rocks', 'rock').replace('flowers', 'flower')
            valid = df[(df.object1 == noun1) & (df.object2 == noun2) & (df.preposition == preposition)][['image', 'score']]
            # if len(valid) == 0:
            #     valid = df[((df.object1 == noun1) | (df.object2 == noun2)) | (df.preposition == preposition)][['image', 'score']]

            retrieved = []
            relevance = []
            if len(valid) > 0:
                # retrieved, relevance = zip(*valid)
                valid.drop_duplicates('image', inplace=True)
                retrieved, relevance = valid['image'].values, valid['score'].values

            tp = len(set(qa.db[query]) & set(retrieved))
            fp = len(retrieved) - tp
            tn = negative - len(set(qa.db[query]))
            fn = len(qa.db[query]) - tp

            y_test = np.zeros(negative, dtype=np.float32)
            for img in qa.db[query]:
                y_test[imagenames.index(img)] = 1.

            y_scores = np.zeros(negative, dtype=np.float32)
            for n1, img in enumerate(retrieved):
                y_scores[imagenames.index(img)] = relevance[n1]

            if len(retrieved) > 0:
                average_precision = average_precision_score(y_test, y_scores)
            else:
                average_precision = 0

            if np.isnan(average_precision):
                avg_precision.append(0.0)
            else:
                avg_precision.append(average_precision)

            queries_by_obj.setdefault(noun1, []).append(average_precision)
            queries_by_obj.setdefault(noun2, []).append(average_precision)
            queries_by_prep.setdefault(preposition, []).append(average_precision)
            # queries_by_obj[noun1] += average_precision
            # queries_by_obj[noun2] += average_precision
            # queries_by_prep[preposition] += average_precision

            apk_ = apk(qa.db[query], retrieved, k=len(qa.db[query]))
            apks.append(apk_)
            precision_[nn], recall_[nn], _ = precision_recall_curve(y_test, y_scores)
            count += 1

            # print("query {} returned {} images, ground-truth has {}. Score {:.4f}".format(query, len(retrieved),
            #                                                                               len(qa.db[query]),
            #                                                                               average_precision))
            print("query {} returned {} images, ground-truth has {}. Score {:.2f}".format(query, len(retrieved),
                                                                                          len(qa.db[query]),
                                                                                          apk_))
            mean_average_precision.setdefault(query, []).append({'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
                                                                 'y_test': y_test.tolist(),
                                                                 'y_scores': y_scores.tolist()})
            yy_true.append(y_test)
            yy_pred.append(y_scores)

            bar.update(1)

    print("{} queries".format(count))
    print('label ranking average precision {}'.format(label_ranking_average_precision_score(yy_true, yy_pred)))
    yy_true = np.array(yy_true)
    yy_pred = np.array(yy_pred)

    print("APK {:.2f}".format(np.mean(apks)))

    y1 = yy_true.flatten()
    y2 = yy_pred.flatten()
    micro = average_precision_score(y1, y2)

    # Plot Precision-Recall curve for each class
    precision_micro, recall_micro, _ = precision_recall_curve(y1, y2)

    print('Precision score {}'.format(micro))
    with open(params.output_file, 'w') as fp:
        json.dump(mean_average_precision, fp)

    print("mAP {:.4f}".format(np.mean(avg_precision)))

    with open("queries_by_element.json", 'w') as fp:
        json.dump({"prepositions": queries_by_prep, "objects": queries_by_obj}, fp)
