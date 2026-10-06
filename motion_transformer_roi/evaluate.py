"""Shared evaluator ensures identical BAF/Transformer origins, geometry and metrics."""
from baf_roi.evaluate import evaluate as evaluate_shared
from .runtime import Predictor


def evaluate(cache_path, checkpoint, output, device='cpu', split='test', batch_size=128,
             scene_cfg=None, save_masks=True):
    return evaluate_shared(cache_path, checkpoint, output, device, split, batch_size,
                           scene_cfg, save_masks, predictor_class=Predictor)
