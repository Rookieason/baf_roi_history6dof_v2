"""Reproduce v2 batch and oracle-label isolation tests on SYNTHETIC smoke output.

Never modifies the original smoke cache. The deliberately corrupted copy exists
only inside a fresh output directory, then is removed after the regression test.
"""
from pathlib import Path
import argparse
import json
import shutil
import sys
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from baf_roi.config import dump_json
from baf_roi.evaluate import evaluate
from baf_roi.runtime import Predictor


def main(smoke_dir, output):
    root, output = Path(smoke_dir).resolve(), Path(output).resolve()
    marker = json.loads((root / 'smoke_result.json').read_text(encoding='utf-8'))
    if not marker.get('evidence_type', '').startswith('SYNTHETIC') or marker.get('epochs_per_stage') != 1:
        raise ValueError('This regression script accepts only this project\'s synthetic smoke output.')
    if output.exists() and any(output.iterdir()):
        raise ValueError('Use a new output directory for smoke regression evidence.')
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    checkpoint = root / 'run/baf_roi.pt'
    a = evaluate(root / 'cache', checkpoint, output / 'batch16', 'cpu', batch_size=16)
    b = evaluate(root / 'cache', checkpoint, output / 'batch7', 'cpu', batch_size=7)
    comparisons = []
    for h in a['horizons_ms']:
        ma, mb = a['metrics'][str(h)], b['metrics'][str(h)]
        exact_keys = ['n_origins', 'tp', 'fp', 'fn', 'tn', 'exact_match']
        if not all(ma[k] == mb[k] for k in exact_keys):
            raise AssertionError(f'Batch regression changed classification counts at {h} ms')
        comparisons.append({'horizon_ms': h, 'identical_counts_and_exact_match': True,
                            'translation_mean_abs_delta_m': abs(ma['translation_mean_m'] - mb['translation_mean_m']),
                            'rotation_mean_abs_delta_deg': abs(ma['rotation_mean_deg'] - mb['rotation_mean_deg'])})
    # Deliberately preserve the identity in a disposable, label-corrupted COPY to
    # test whether model/evaluation logic accidentally consults those labels.
    # This is an adversarial test, not a valid cache for a scientific experiment.
    mutated = output / 'DISPOSABLE_corrupted_test_labels'
    shutil.copytree(root / 'cache', mutated)
    manifest = json.loads((mutated / 'manifest.json').read_text(encoding='utf-8'))
    try:
        for r in manifest['trajectories']:
            if r['split'] == 'test':
                for name in ('labels', 'next_labels', 'scores'):
                    path = mutated / r['key'] / f'{name}.npy'
                    v = np.load(path, allow_pickle=False)
                    np.save(path, np.full_like(v, -1 if name != 'scores' else np.nan), allow_pickle=False)
                (mutated / r['key'] / 'supervision_metadata.npz').unlink()
        c = evaluate(mutated, checkpoint, output / 'no_test_labels', 'cpu', batch_size=16)
        if c['metrics'] != a['metrics']:
            raise AssertionError('Test supervision affected evaluation predictions or sample selection')
    finally:
        shutil.rmtree(mutated)
    history = json.loads((root / 'history.json').read_text(encoding='utf-8'))
    pred = Predictor.load(checkpoint).predict_observations(history['timestamps_s'], history['positions'], history['rotations_deg'])
    assert pred['positions_m'].shape == (1, 5, 3)
    out = {'status': 'PASS', 'type': 'SYNTHETIC REGRESSION ONLY',
           'test_label_corruption_all_metrics_identical': True,
           'supervision_metadata_deleted_at_test_did_not_affect_predictions': True,
           'batch_size_16_vs_7': comparisons, 'history_only_prediction': 'PASS'}
    dump_json(output / 'extra_validation.json', out)
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--smoke-dir', required=True); p.add_argument('--output', required=True)
    args = p.parse_args(); main(args.smoke_dir, args.output)
