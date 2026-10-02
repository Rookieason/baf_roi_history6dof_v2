"""Compare archived v1 labels with author-centered v2 on the SAME cache origins.

This is a preprocessing diagnostic, NOT prediction accuracy or benchmark evidence.
"""
from pathlib import Path
import argparse
import importlib.util
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from baf_roi.data import Cache
from baf_roi.config import dump_json


def compare(cache_path):
    path = Path(__file__).resolve().parents[1] / 'reference/v1_supervision.py'
    spec = importlib.util.spec_from_file_location('_archived_v1_supervision', path)
    old = importlib.util.module_from_spec(spec); spec.loader.exec_module(old)
    cache = Cache(cache_path)
    results = []
    for ti, record in enumerate(cache.records):
        a = cache.arrays(ti)
        v1 = np.array([old.behavior(a['series'].history(int(i)))[0] for i in a['anchors']])
        v2 = a['labels']
        counts = lambda labels: {name: int(np.sum(labels == value)) for value, name in
                                  ((0, 'Focused'), (1, 'Explorer'), (-1, 'Unlabeled'))}
        matrix = np.zeros((3, 3), np.int64)
        order = {0: 0, 1: 1, -1: 2}
        for before, after in zip(v1, v2):
            matrix[order[int(before)], order[int(after)]] += 1
        results.append({'trajectory': record['key'], 'split': record['split'],
                        'same_input_origins': len(v1), 'changed_label_count': int(np.sum(v1 != v2)),
                        'v1': counts(v1), 'v2': counts(v2),
                        'matrix_rows_v1_columns_v2': matrix.tolist()})
    return {'purpose': 'PREPROCESSING DIAGNOSTIC ONLY; not accuracy',
            'matrix_label_order': ['Focused', 'Explorer', 'Unlabeled'],
            'total_origins': sum(r['same_input_origins'] for r in results),
            'total_changed_labels': sum(r['changed_label_count'] for r in results),
            'trajectory_results': results}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache', required=True); p.add_argument('--output', required=True)
    args = p.parse_args()
    report = compare(args.cache)
    dump_json(args.output, report)
    print(f"Changed {report['total_changed_labels']}/{report['total_origins']} labels; no accuracy claim.")
