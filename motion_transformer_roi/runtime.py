"""Causal next-step Transformer with recursive rollout and exact-horizon interpolation."""
import math
import numpy as np
import torch
from baf_roi.data import Series
from baf_roi.pose import model_inputs, ypr_to_quat, quat_to_ypr, interpolate_predictions
from .models import build_model, source_hashes


class Predictor:
    def __init__(self, model, stats, device='cpu', data_config=None):
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.models = {'motion_transformer': self.model}
        self.stats = stats
        self.data_config = data_config or {'position_unit': 'm', 'max_gap_ms': 100}

    @classmethod
    def load(cls, checkpoint, device='cpu'):
        bundle = torch.load(checkpoint, map_location='cpu', weights_only=True)
        if bundle.get('format_version') != 2 or bundle.get('method') != 'motion_transformer_roi':
            raise ValueError('Expected adapted motion_transformer_roi v2 checkpoint; train on the shared cache first.')
        if bundle.get('source_sha256') != source_hashes():
            raise ValueError('Motion Transformer source differs from checkpoint.')
        model = build_model(bundle['model_config'])
        model.load_state_dict(bundle['model_state_dict'], strict=True)
        result = cls(model, bundle['normalization'], device, bundle['config']['data'])
        result.bundle = bundle
        return result

    @torch.inference_mode()
    def forecast_batch(self, histories, horizons_ms=(50, 100, 150, 200, 250)):
        h = np.asarray(histories, np.float64).copy()
        if h.ndim != 3 or h.shape[1:] != (30, 6) or not len(h) or not np.isfinite(h).all():
            raise ValueError('Need finite, nonempty history [B,30,6].')
        hs = list(horizons_ms)
        if not hs or len(set(hs)) != len(hs) or any(not np.isfinite(x) or not 50 <= x <= 250 for x in hs):
            raise ValueError('Distinct horizons must be within [50,250] ms.')
        steps = math.ceil(max(hs) * 30 / 1000 - 1e-10)
        ps, qs = [h[:, -1, :3].copy()], [ypr_to_quat(h[:, -1, 3:])]
        mean, std = np.asarray(self.stats['mean']), np.asarray(self.stats['std'])
        for _ in range(steps):
            _, pose = model_inputs(h, self.stats)
            out = self.model(torch.as_tensor(pose, device=self.device))
            p = out['p_hat_tr'].cpu().numpy() * std + mean
            q = out['q_hat_norm'].cpu().numpy().astype(np.float64)
            norms = np.linalg.norm(q, axis=-1, keepdims=True)
            if not np.isfinite(p).all() or not np.isfinite(q).all() or np.any(norms < 1e-8):
                raise FloatingPointError('Model produced an invalid predicted pose.')
            q /= norms
            ps.append(p); qs.append(q)
            generated = np.concatenate((p, quat_to_ypr(q)), axis=-1)
            h = np.concatenate((h[:, 1:], generated[:, None]), axis=1)
        rollout_p, rollout_q = np.stack(ps, 1), np.stack(qs, 1)
        positions, quaternions = interpolate_predictions(rollout_p, rollout_q, hs)
        return {'horizons_ms': hs, 'positions_m': positions, 'quaternions_wxyz': quaternions,
                'rotations_ypr_deg': quat_to_ypr(quaternions),
                'rollout_positions_m': rollout_p, 'rollout_quaternions_wxyz': rollout_q}

    def predict_observations(self, timestamps_s, positions, rotations_deg,
                             horizons_ms=(50, 100, 150, 200, 250), stream_id=None):
        ts, pos, rot = (np.asarray(v, np.float64) for v in (timestamps_s, positions, rotations_deg))
        if ts.ndim != 1 or len(ts) < 2 or pos.shape != (len(ts), 3) or rot.shape != pos.shape:
            raise ValueError('Need timestamps[N], positions[N,3], rotations_deg[N,3].')
        scale = .01 if self.data_config['position_unit'] == 'cm' else 1.
        series = Series(ts - ts[0], np.concatenate((pos * scale, rot), 1), self.data_config['max_gap_ms'] / 1000)
        return self.forecast_batch(series.history(len(ts) - 1)[None], horizons_ms)
