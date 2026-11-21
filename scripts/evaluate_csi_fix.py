"""Evaluate fixed checkpoints and a validation RF-permutation negative control."""
from collections import defaultdict
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from baf_roi_heatmap.contracts import new_directory, write_json
from baf_roi_heatmap.data import ForecastDataset
from roi_experiments.evaluation import evaluate
from roi_experiments.runner import resolve_device, assert_matched_reports


class ShuffledRF:
    def __init__(self, dataset):
        self.original = dataset
        self.windows = []
        groups = defaultdict(list)
        for i in range(len(dataset)):
            batch = dataset[i]; inputs = batch['inputs']
            self.windows.append({key: inputs[key] for key in ('maps', 'mask', 'timing')})
            ri = int(dataset.index[i, 0])
            groups[(dataset.records[ri]['session'], bool(inputs['rf_available']))].append(i)
        generator = np.random.default_rng(42)
        self.permutation = np.arange(len(dataset))
        for indices in groups.values():
            self.permutation[indices] = generator.permutation(indices)

    def __getattr__(self, name): return getattr(self.original, name)
    def __len__(self): return len(self.original)
    def __getitem__(self, index):
        batch = self.original[index]
        return {**batch, 'inputs': {**batch['inputs'], **self.windows[int(self.permutation[index])]}}


def score(report):
    pooled = []; macro = []
    for key, summary in report['summaries'].items():
        if key.endswith('margin0'):
            pooled.append(summary['all']['pooled']['iou_mean'])
            macro.append(summary['all']['recording_macro']['iou_mean'])
    return {'pooled_mean_over_horizons': float(np.mean(pooled)),
            'recording_macro_mean_over_horizons': float(np.mean(macro)),
            'correction_applied_fraction': report['correction_applied_fraction']}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experiments', nargs='+', required=True)
    p.add_argument('--variants', nargs='+')
    p.add_argument('--splits', nargs='+', choices=['valid', 'test'], default=['valid'])
    p.add_argument('--output', required=True); p.add_argument('--device', default='auto')
    a = p.parse_args(); torch.set_num_threads(2); device = resolve_device(a.device)
    out = new_directory(a.output); results = []; baseline_reports = {}
    for directory in a.experiments:
        experiments = json.loads((Path(directory)/'ablation_summary.json').read_text())
        for experiment in experiments:
            variant = experiment['variant']
            if a.variants and variant not in a.variants: continue
            checkpoint = experiment['checkpoint']
            bundle = torch.load(checkpoint, map_location='cpu', weights_only=True)
            option = {'alpha_checkpoint': checkpoint} if bundle.get('method') == 'baf_roi_heatmap_alpha' else {'checkpoint': checkpoint}
            destination = out/(Path(directory).name+'_'+variant)
            for split in a.splits:
                dataset = ForecastDataset(experiment['prepared'], split)
                key = (experiment['prepared'], split)
                if key not in baseline_reports:
                    baseline_reports[key] = evaluate(experiment['prepared'], out/f'baseline_{len(baseline_reports)}',
                                                     split=split, device=device, margins=[0.], _dataset=dataset)
                actual = evaluate(experiment['prepared'], destination/split, split=split, device=device,
                                  margins=[0.], _dataset=dataset, **option)
                baseline = baseline_reports[key]
                assert_matched_reports({'baf': baseline, 'adapter': actual})
                result = {'experiment': str(directory), 'variant': variant, 'split': split,
                          'checkpoint_selection': actual['checkpoint_selection'],
                          'baseline': score(baseline), 'actual': score(actual)}
                if split == 'valid' and bundle['selected_epoch'] > 0:
                    shuffled = evaluate(experiment['prepared'], destination/'valid_shuffled_rf', split='valid',
                                        device=device, margins=[0.], _dataset=ShuffledRF(dataset), **option)
                    shuffled['input_intervention'] = 'Within-session RF window permutation, seed 42; labels and HMD fixed. Diagnostic only.'
                    write_json(destination/'valid_shuffled_rf/report.json', shuffled)
                    result['shuffled_rf'] = score(shuffled)
                results.append(result)
                write_json(out/'comparison.json', results)
                print('COMPARISON', json.dumps(result), flush=True)


if __name__ == '__main__': main()
