#!/usr/bin/env python
from __future__ import division
import os
import argparse
import json
import numpy as np
import pandas as pd
from imret.query import Annotation, KnowledgeBase
from imret.dataset import Dataset


def apk(expected, predicted, k=4317):
    retrieved = np.array([ret in expected for ret in predicted[:k]]).astype(np.int32)
    if retrieved.sum() == 0:
        return 0.0

    scores = np.arange(len(retrieved)) + 1
    valids = np.where(retrieved > 0)
    return np.sum(np.cumsum(retrieved[valids]) / scores[valids].astype(np.float32)) / float(min(len(expected), k))

def evaluate(queries, ground_truth, kb, dataset, verbose=False):
    mean_average_precision_ = []
    retrieved_by_query = {}
    for nn, query in enumerate(ground_truth):
        print("Processed {}/{}".format(nn, len(ground_truth)))
        equivalent = queries.get(query)
        if equivalent:
            retrieved = [imname for imname, found in kb.runquery(equivalent, verbose=verbose) if found]
            retrieved_by_query[equivalent] = retrieved
            y_true = [img for img in dataset.images if img in ground_truth[query]]
            y_pred = [img for img in dataset.images if img in retrieved]
            score = apk(y_true, y_pred)
            print("query {} returned {} images, ground-truth has {}. Score {:.4f}".format(equivalent, len(retrieved),
                                                                                          len(ground_truth[query]),
                                                                                          score))
            mean_average_precision_.append(score)

    return mean_average_precision_, retrieved_by_query


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='IRRCC program')
    parser.add_argument('-d', '--dataset_path', action="store", default='data/datasets')
    parser.add_argument('-i', '--image_path', action="store", default='data/images')
    parser.add_argument('-p', '--path', action="store", default='data/preposition/index-0.900.csv')
    parser.add_argument('-q', '--queries_path', action="store", default='data/query/query_equivalence.csv')
    # parser.add_argument('-t', '--test_anno', action="store", default='data/query/test_anno/')
    parser.add_argument('-t', '--test_anno', action="store", default='data/corrected_test_anno/')
    parser.add_argument('-o', '--output_file', action="store", default='data/prover/map_prover-0900')
    parser.add_argument('--ltb_runner', action="store", default='data/prover/E/PROVER/e_ltb_runner')
    parser.add_argument('--eprover', action="store", default='data/prover/E/PROVER/eprover')
    # parser.add_argument('--sumo', action="store", default='/home/danilo/sigma_run/KBs/SUMO.tptp')
    parser.add_argument('--sumo', action="store", default='/Users/danilonunes/Documents/sigma/run/KBs/SUMO.tptp')
    parser.add_argument('--verbose', dest='verbose', action="store_true", default=False)
    parser.add_argument('--no-verbose', dest='verbose', action='store_false')
    parser.set_defaults(feature=True)
    params = parser.parse_args()

    db = Dataset(params.dataset_path, 'test', params.image_path)
    df = pd.read_csv(params.path)
    kb = KnowledgeBase(df,
                       ltb_runner=params.ltb_runner,
                       eprover=params.eprover,
                       sumo=params.sumo,
                       user_assertions='/Users/danilonunes/Documents/sigma/onto/SUMO_UserAssertions.tptp')

    queries = pd.read_csv(params.queries_path)
    queries.dropna(inplace=True)
    queries = dict(zip(queries['Original'], queries['Equivalent']))

    location = os.path.join(params.test_anno)
    qa = Annotation(location)

    query_db = {query: ground_truth for query, ground_truth in qa.db.items() if queries.get(query)}

    mean_average_precision, retrieved = evaluate(queries, query_db, kb, db, params.verbose)
    with open(params.output_file + '.json', 'w') as fp:
        json.dump(mean_average_precision, fp)

    with open(params.output_file + '_retrieved.json', 'w') as fp:
        json.dump(retrieved, fp)

    print("mAP {:.4f}".format(np.mean(mean_average_precision)))
