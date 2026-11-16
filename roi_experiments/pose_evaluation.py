"""Pose-only view of the same matched-origin evaluation data contract.

No CSI manifest or pretrained BAF is required to evaluate Motion Transformer.
Only the data adapter differs; masks, rows, metrics and reports are produced by
roi_experiments.evaluation.evaluate for every method.
"""
import numpy as np
from torch.utils.data import Dataset

from baf_roi.config import validate_horizons
from baf_roi.data import Cache
from baf_roi.pose import ypr_to_quat
from baf_roi.roi import load_scene
from baf_roi.runtime import observed_weights
from baf_roi_heatmap.contracts import digest


class PoseEvaluationDataset(Dataset):
    def __init__(self, cache_path, predictor, split='test', horizons_ms=None, scene_config=None, batch_size=64):
        if split not in ('valid', 'test') or batch_size < 1:
            raise ValueError('Use valid/test and a positive batch size.')
        self.cache = Cache(cache_path)
        if predictor.bundle['cache_id'] != self.cache.manifest['cache_id'] or predictor.stats != self.cache.stats:
            raise ValueError('Checkpoint must use exactly this HMD cache and normalization.')
        config = self.cache.manifest['config']
        self.horizons = validate_horizons(config['evaluation']['horizons_ms'] if horizons_ms is None else horizons_ms)
        self.scene = load_scene(scene_config or config['scene'])
        self.split, self.records, self._arrays = split, [], []
        self.rf = {'grid_id': None, 'records': []}
        uses_alpha = callable(getattr(predictor, 'confidence', None))
        sources = {}
        for ci, record in enumerate(self.cache.records):
            source = record['source_sha256']
            if source in sources and sources[source] != record['split']:
                raise ValueError('HMD source bytes cross dataset splits.')
            sources[source] = record['split']
            if record['split'] != split:
                continue
            arrays = self.cache.arrays(ci)
            series = arrays['series']
            chosen, targets, weights, state = [], [], [], None
            for start in range(0, len(arrays['anchors']), batch_size):
                anchors = arrays['anchors'][start:start+batch_size]
                if uses_alpha:
                    histories = np.stack([series.history(int(i)) for i in anchors])
                    alpha, state = observed_weights(predictor.confidence(histories), series.t[anchors],
                                                    predictor.mu, series.max_gap, state)
                else:
                    alpha = np.zeros(len(anchors), np.float32)
                for j, anchor in enumerate(anchors):
                    try:
                        target = series.at(series.t[anchor] + np.asarray(self.horizons)/1000)
                    except ValueError:
                        continue
                    chosen.append(int(anchor)); targets.append(target); weights.append(alpha[j])
            if not chosen:
                continue
            anchors, targets = np.asarray(chosen, np.int64), np.stack(targets)
            tq = ypr_to_quat(targets[..., 3:])
            self._arrays.append({'anchors': anchors, 'origin_s': series.t[anchors]+record['epoch_start_s'],
                                 'alpha': np.asarray(weights, np.float32),
                                 'target_p': targets[..., :3], 'target_q': tq,
                                 'target_mask': self.scene.mask(targets[..., :3], tq)})
            self.records.append({'key': record['key'], 'cache_index': ci, 'split': split,
                                 'session': record['session'], 'group_id': record['session'],
                                 'recording_id': f"{record['session']}:{record['source_sha256']}",
                                 'scenario': 'unspecified', 'count': len(anchors)})
        self.index = np.asarray([(ri, oi) for ri, r in enumerate(self.records)
                                 for oi in range(r['count'])], np.int64).reshape(-1, 2)
        if not len(self.index):
            raise ValueError(f'No common-horizon origins in {split}.')
        identity = {'hmd_cache_id': self.cache.manifest['cache_id'], 'horizons_ms': self.horizons,
                    'scene_id': self.scene.snapshot()['scene_id'], 'rf': 'not_requested'}
        self.manifest = {**identity, 'experiment_id': digest(identity), 'scene': self.scene.snapshot()}

    def __len__(self):
        return len(self.index)

    def arrays(self, record_index):
        return self._arrays[record_index]

    def recording_id(self, record_index):
        return self.records[record_index]['recording_id']

    def __getitem__(self, index):
        ri, oi = map(int, self.index[index])
        return {'record_index': ri, 'origin_index': oi,
                'inputs': {'rf_available': False, 'alpha': self._arrays[ri]['alpha'][oi]}}
