#!/usr/bin/env python
from __future__ import division
import os
import json
import argparse
import click
import numpy as np
import pandas as pd
import cytoolz
import json
from sklearn.metrics import average_precision_score
from imret.query import StructuredQuery
from imret.dataset import Dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='IRRCC program')
    parser.add_argument('-d', '--dataset_path', action="store", default='data/datasets')
    parser.add_argument('-i', '--image_path', action="store", default='data/images')
    parser.add_argument('-b', '--index_base', action="store", default='data/preposition/index.csv')
    parser.add_argument('-o', '--output_file', action="store", default='data/preposition/map.json')
    parser.add_argument('-t', '--threshold', action="store", default=3.0, type=float)
    parser.set_defaults(feature=True)
    params = parser.parse_args()

    dataset = Dataset(params.dataset_path, 'test', params.image_path)
    sq = StructuredQuery(os.path.join(params.dataset_path, 'Struct-Query-Test.mat'))
    df = pd.read_csv(params.index_base)

    imagenames = dataset.images  # {nn:img for nn, img in enumerate(dataset.images)}

    map_ = {}
    weights = df.image.value_counts().to_dict()
    negative = len(imagenames)
    
    for query_type in sorted(sq.query_types):
        avg_precision = []
        mean_average_precision = []
        with click.progressbar(length=len(sq[query_type]), show_pos=True, show_percent=True) as bar:
            for query in sq[query_type]:
                l1 = []
                l2 = []
                l3 = []
                l4 = []
                for term in query['name'].split('&'):
                    try:
                        unary, binary = term.split(',')
                    except ValueError:
                        binary = term
                        unary = ''

                    noun1, preposition, noun2 = binary.split('-')
                    l1 += list(df[df['object1'] == noun1].image)
                    l2 += list(df[df['object2'] == noun2].image)
                    l3 += list(df[df['preposition'] == preposition].image)
                    l4 += list(df[((df['object1'] == unary) | (df['object2'] == unary)) & (df['rcc'] == 'DC')].image)

                    # l1 += list(df[(df['object1'] == noun1) & (df['rcc'].notnull())].image)
                    # l2 += list(df[(df['object2'] == noun2) & (df['rcc'].notnull())].image)
                    # l3 += list(df[(df['preposition'] == preposition) & (df['rcc'].notnull())].image)
                    # l4 += list(df[((df['object1'] == unary) | (df['object2'] == unary)) & (df['rcc'] == 'DC')].images)

                retrieved = {k: v / weights[k] for k, v in cytoolz.frequencies(l1 + l2 + l3 + l4).items()}
                valids = [(k, retrieved[k]) for k in sorted(retrieved, key=retrieved.get, reverse=True) if
                          retrieved[k] >= 2.5]
                retrieved = []
                relevance = []
                if valids:
                    retrieved, relevance = zip(*valids)

                gs = [imagenames[idx] for idx, is_valid in enumerate(query['rank']) if is_valid]
                tp = len(set(gs) & set(retrieved))
                fp = len(retrieved) - tp
                tn = negative - len(set(gs))
                fn = len(gs) - tp

                y_test = query['rank'].astype(np.float32)

                y_scores = np.zeros(negative, dtype=np.float32)
                for nn, img in enumerate(retrieved):
                    y_scores[imagenames.index(img)] = relevance[nn]

                average_precision = average_precision_score(y_test, y_scores)
                if np.isnan(average_precision):
                    avg_precision.append(0.0)
                else:
                    avg_precision.append(average_precision)

                mean_average_precision.append({'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
                                               'y_test': y_test.tolist(),
                                               'y_scores': y_scores.tolist()})

                map_[query_type] = mean_average_precision
                bar.update(1)
            print("\nQuery {} mAP {:.4f}".format(query_type, np.mean(avg_precision)))

    with open(params.output_file, 'w') as fp:
        json.dump(map_, fp)