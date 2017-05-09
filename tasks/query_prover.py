#!/usr/bin/env python
import os
import argparse
import click
import json
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from imret.query import Annotation, KnowledgeBase


def evaluate(queries, ground_truth, kb):
    mean_average_precision = []
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

            score = average_precision_score(ground_truth.db[query], retrieved)
            print("query {} returned {} images, ground-truth has {}. Score {:.4f}".format(query, len(retrieved),
                                                                                          len(ground_truth.db[query]),
                                                                                          score))
            mean_average_precision.append(score)

    return mean_average_precision


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='IRRCC program')
    parser.add_argument('-d', '--dataset_path', action="store", default='data/preposition/index.csv')
    parser.add_argument('-q', '--queries_path', action="store", default='data/query/query_equivalence.csv')
    parser.add_argument('-t', '--test_anno', action="store", default='data/query/test_anno/')
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
