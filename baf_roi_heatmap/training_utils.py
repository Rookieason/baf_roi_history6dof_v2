"""Recording-balanced optimization and explicit baseline-retained checkpoint status."""
from collections import Counter
from pathlib import Path

import torch


def recording_weights(dataset, aggregation):
    if aggregation not in ('origin', 'recording'):
        raise ValueError('aggregation must be origin or recording.')
    if aggregation == 'origin':
        return torch.ones(len(dataset.records))
    counts = Counter()
    for ri, record in enumerate(dataset.records):
        counts[dataset.recording_id(ri)] += record['count']
    # E_origin[w * loss] equals the recording-macro loss. Segmentation does
    # not give a recording extra votes. Do not renormalize within each batch.
    return torch.tensor([len(dataset)/(len(counts)*counts[dataset.recording_id(ri)])
                         for ri in range(len(dataset.records))])


class CheckpointSelection:
    def __init__(self, output, filename):
        self.output, self.filename = Path(output), filename
        self.best, self.best_trained = -float('inf'), -float('inf')
        self.selected_epoch = self.best_trained_epoch = None
        self.baseline = None

    def update(self, bundle):
        epoch, score = bundle['selected_epoch'], bundle['validation_recording_macro_iou']
        if epoch == 0:
            self.baseline = score
        else:
            torch.save(bundle, self.output/'last.pt')
            if score > self.best_trained + 1e-12:
                self.best_trained, self.best_trained_epoch = score, epoch
                torch.save(bundle, self.output/'best_trained.pt')
        if bundle.get('selection_eligible', True) and score > self.best + 1e-12:
            self.best, self.selected_epoch = score, epoch
            torch.save(bundle, self.output/self.filename)
        return {'selected_epoch': self.selected_epoch,
                'selected_validation_iou': self.best,
                'baseline_validation_iou': self.baseline,
                'best_trained_epoch': self.best_trained_epoch,
                'best_trained_validation_iou': self.best_trained if self.best_trained_epoch is not None else None,
                'selection_status': 'baseline_retained' if self.selected_epoch == 0 else 'trained_correction_selected'}
