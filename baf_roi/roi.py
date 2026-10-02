"""Deterministic pose -> partition visibility. Static scene geometry is not model input."""
from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np
from scipy.spatial.transform import Rotation
from .pose import finite_array
from .config import fingerprint


def grid_centers(center, size, grid_dim):
    center = finite_array(center, (3,))
    size = np.asarray(size, np.float64)
    if size.ndim == 0:
        size = np.repeat(size, 3)
    if size.shape != (3,) or np.any(size <= 0) or not np.isfinite(size).all() or grid_dim <= 0:
        raise ValueError('Positive object size and grid dimension required.')
    axes = [center[i] - size[i] / 2 + (np.arange(grid_dim) + .5) * size[i] / grid_dim for i in range(3)]
    return np.array([[x, y, z] for z in axes[2] for y in axes[1] for x in axes[0]])


@dataclass
class Scene:
    centers_m: np.ndarray
    partition_ids: list[str]
    cfg: dict

    def __post_init__(self):
        self.centers_m = finite_array(self.centers_m, (3,))
        if self.centers_m.ndim != 2 or not len(self.centers_m):
            raise ValueError('Scene must contain partition centers [N,3].')
        if len(self.partition_ids) != len(self.centers_m) or len(set(self.partition_ids)) != len(self.partition_ids):
            raise ValueError('Partition IDs must be unique and align with centers.')
        mode = self.cfg['visibility_mode']
        keys = ['fov_deg'] if mode == 'legacy_center_cone' else ['horizontal_fov_deg', 'vertical_fov_deg']
        if mode not in ('legacy_center_cone', 'rectangular_center_frustum'):
            raise ValueError(f'Unsupported visibility_mode: {mode}')
        if any(not np.isfinite(self.cfg.get(k, np.nan)) or not 0 < self.cfg[k] < 180 for k in keys):
            raise ValueError(f'Calibrated finite FoV angles in (0,180) required: {keys}')
        if mode == 'rectangular_center_frustum':
            near, far = self.cfg.get('near_m', .01), self.cfg.get('far_m')
            if not np.isfinite(near) or near < 0 or (far is not None and (not np.isfinite(far) or far <= near)):
                raise ValueError('Invalid frustum near/far planes.')
        distance = self.cfg.get('max_distance_m')
        if distance is not None and (not np.isfinite(distance) or distance <= 0):
            raise ValueError('max_distance_m must be positive.')

    def mask(self, positions_m, quaternions_wxyz):
        p, q = finite_array(positions_m, (3,)), finite_array(quaternions_wxyz, (4,))
        prefix = p.shape[:-1]
        p, q = p.reshape(-1, 3), q.reshape(-1, 4)
        if len(p) != len(q) or np.any(np.linalg.norm(q, axis=-1) < 1e-10):
            raise ValueError('Invalid pose batch for visibility.')
        r = Rotation.from_quat(q[:, [1, 2, 3, 0]]).as_matrix()
        direction = self.centers_m[None] - p[:, None]
        distance = np.linalg.norm(direction, axis=-1)
        if self.cfg['visibility_mode'] == 'legacy_center_cone':
            forward = r[:, :, 0]
            projection = np.einsum('bnc,bc->bn', direction, forward)
            inside = projection >= distance * np.cos(np.deg2rad(self.cfg['fov_deg'] / 2))
            inside |= distance <= 1e-11  # same 1e-9 cm degeneracy rule as original, converted to m
        else:
            local = np.einsum('bnc,bcd->bnd', direction, r)
            depth = local[:, :, 0]
            inside = (depth > 0) & (depth >= self.cfg.get('near_m', .01))
            inside &= np.abs(local[:, :, 1]) <= depth * np.tan(np.deg2rad(self.cfg['horizontal_fov_deg'] / 2))
            inside &= np.abs(local[:, :, 2]) <= depth * np.tan(np.deg2rad(self.cfg['vertical_fov_deg'] / 2))
            if self.cfg.get('far_m') is not None:
                inside &= depth <= self.cfg['far_m']
        if self.cfg.get('max_distance_m') is not None:
            inside &= distance <= self.cfg['max_distance_m']
        return inside.reshape(*prefix, len(self.centers_m))

    def snapshot(self):
        result = {'config': self.cfg, 'partition_ids': self.partition_ids, 'centers_m': self.centers_m.tolist(),
                  'definition': 'partition-center membership; no occlusion or box/mesh intersection'}
        result['scene_id'] = fingerprint(result)
        return result


def load_scene(cfg):
    if not cfg.get('geometry_confirmed', False):
        raise ValueError('Confirm partition geometry, FoV and visibility definition; set scene.geometry_confirmed=true.')
    direct = cfg.get('partition_centers_file')
    if direct:
        data = json.loads(Path(direct).read_text(encoding='utf-8'))
        if data.get('coordinate_convention') != 'legacy_ypr_xforward' or data.get('unit') != 'm':
            raise ValueError('Centers file must explicitly use unit=m and coordinate_convention=legacy_ypr_xforward.')
        return Scene(np.asarray(data['centers_m']), data['partition_ids'], cfg)
    path = cfg.get('adaptive_config_path')
    if not path:
        raise ValueError('Supply partition_centers_file or adaptive_config_path; no scene was included in the uploaded ZIP.')
    if not cfg.get('allow_uniform_grid_approximation', False):
        raise ValueError('Uniform bounding cubes need explicit allow_uniform_grid_approximation=true; prefer real world-space centers.')
    root = json.loads(Path(path).read_text(encoding='utf-8'))
    defaults = (root.get('defaults') or {}).get('config') or {}
    streams = root.get('streams') or root.get('objects') or []
    if not streams or any(not isinstance(s, dict) for s in streams):
        raise ValueError('Expected a nonempty list of stream/object dictionaries.')
    grid_dim = int(cfg['grid_dim'])
    centers, ids = [], []
    for obj_index, stream in enumerate(streams):
        config = defaults.copy()
        if isinstance(stream.get('config'), dict):
            config.update(stream['config'])
        else:
            config.update({k: v for k, v in stream.items() if k.startswith('obj_transform_')})
        # No false claims that this adapter reproduces unprovided server transforms.
        for key, value in config.items():
            if key.startswith('obj_transform_rot') and float(value) != 0:
                raise ValueError('Nonzero object rotation: export actual world-space partition centers instead.')
            if key.startswith('obj_transform_scale') and float(value) != 1:
                raise ValueError('Object scale: export actual world-space partition centers instead.')
        center_cm = [float(config.get(f'obj_transform_pos_{axis}_cm', 0)) for axis in 'xyz']
        points = grid_centers(np.array(center_cm) / 100, np.asarray(cfg['object_size_cm']) / 100, grid_dim)
        centers.extend(points)
        ids.extend(f'object_{obj_index}:part_{i}' for i in range(len(points)))
    return Scene(np.asarray(centers), ids, cfg)
