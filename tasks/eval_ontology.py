#!/usr/bin/env python
from __future__ import division
import os
import click
import argparse
import json
import pandas as pd
import numpy as np
from imret.query import KnowledgeBase, Annotation
# from imret.metrics import apk

def apk(expected, predicted, k=4317):
    retrieved = np.array([ret in expected for ret in predicted[:k]]).astype(np.int32)
    if retrieved.sum() == 0:
        return 0.0

    scores = np.arange(len(retrieved)) + 1
    valids = np.where(retrieved > 0)
    return np.sum(np.cumsum(retrieved[valids]) / scores[valids].astype(np.float32)) / float(min(len(expected), k))


def evaluate(queries, ground_truth, kb):
    mean_average_precision = []
    valid_queries = [v for v in queries.values() if v is not None]
    for nn, query in enumerate(ground_truth.db):
        print("Processed {}/{}".format(nn, len(ground_truth.db)))
        equivalent = queries.get(query)
        if equivalent:
            retrieved = []
            with click.progressbar(length=len(kb.images), show_pos=True, show_percent=True) as bar:
                for returned, found in kb.runquery(equivalent):
                    if found:
                        retrieved.append(returned)
                    bar.update(1)

            score = apk(ground_truth.db[query], retrieved)
            print(u"query {} returned {} images, ground-truth has {}. Score {:.4f}".format(query, len(retrieved),
                                                                                           len(ground_truth.db[query]),
                                                                                           score))
            mean_average_precision.append(score)

    return mean_average_precision


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='IRRCC program')
    parser.add_argument('-d', '--dataset_path', action="store", default='data/segmentation/predicted_relation.csv')
    parser.add_argument('-q', '--queries_path', action="store", default='data/segmentation/query_equivalence.csv')
    parser.add_argument('-t', '--test_anno', action="store", default='data/segmentation/test_anno/')
    parser.add_argument('--ltb_runner', action="store", default='data/segmentation/ltb_runner')
    parser.add_argument('--eprover', action="store", default='data/segmentation/eprover')
    parser.add_argument('--batch_config', action="store", default='EBatchConfig.txt')
    params = parser.parse_args()

    df = pd.read_csv(params.dataset_path)
    kb = KnowledgeBase(df, ltb_runner=params.ltb_runner, eprover=params.eprover, batch_config=params.batch_config)

    queries = pd.read_csv(params.queries_path)
    queries = dict(zip(queries['Original'], queries['Equivalent']))

    location = os.path.join(params.test_anno)
    qa = Annotation(location)
    mean_average_precision = evaluate(queries, qa, kb)
    with open(os.path.join('data/segmentation/map_prover.json'), 'w') as fp:
        json.dump(mean_average_precision, fp)
    print("mAP {:.4f}".format(np.mean(mean_average_precision)))