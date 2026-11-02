"""Two testable extensions: horizon-conditioned pose residuals and expert gating."""
from dataclasses import asdict
import math
import torch
from torch import nn
import torch.nn.functional as F
from .contracts import ModelConfig


def unit_quaternion(q):
    return F.normalize(q, dim=-1, eps=1e-8)


def quaternion_product(a, b):
    aw, av, bw, bv = a[..., :1], a[..., 1:], b[..., :1], b[..., 1:]
    return torch.cat((aw * bw - (av * bv).sum(-1, keepdim=True),
                      aw * bv + bw * av + torch.cross(av, bv, dim=-1)), dim=-1)


def rotation_vector_quaternion(vector):
    angle = torch.linalg.vector_norm(vector, dim=-1, keepdim=True)
    return torch.cat((torch.cos(angle / 2), 0.5 * torch.sinc(angle / (2 * math.pi)) * vector), dim=-1)


def bounded_vector(vector, bound):
    radius = torch.linalg.vector_norm(vector, dim=-1, keepdim=True).clamp_min(1e-8)
    return bound * torch.tanh(radius) / radius * vector


def slerp(a, b, weight):
    a, b = unit_quaternion(a), unit_quaternion(b)
    dot = (a * b).sum(-1, keepdim=True)
    b = torch.where(dot < 0, -b, b)
    dot = dot.abs().clamp(0, 1)
    theta = torch.acos(dot.clamp(max=1 - 1e-6))
    w = weight[..., None]
    spherical = (torch.sin((1 - w) * theta) * a + torch.sin(w * theta) * b) / torch.sin(theta).clamp_min(1e-8)
    linear = (1 - w) * a + w * b
    return unit_quaternion(torch.where(dot > .9995, linear, spherical))


def rotation_matrix(q):
    w, x, y, z = unit_quaternion(q).unbind(-1)
    return torch.stack((1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y),
                        2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x),
                        2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)), dim=-1).reshape(*q.shape[:-1], 3, 3)


def ablate_maps(maps, mask, mode):
    if mode == 'metadata_only':
        maps = torch.zeros_like(maps)
    elif mode == 'spatial_mean':
        maps = maps.mean((-2, -1), keepdim=True).expand_as(maps)
    elif mode == 'angle_only':
        maps = maps.mean(-1, keepdim=True).expand_as(maps)
    elif mode == 'delay_only':
        maps = maps.mean(-2, keepdim=True).expand_as(maps)
    elif mode == 'abs_change':
        maps = torch.cat((maps[:, :, :1], maps[:, :, 1:].abs()), dim=2)
    elif mode == 'last_frame':
        maps, mask = maps.clone(), mask.clone()
        maps[:, :-1] = 0
        mask[:, :-1] = False
        # Remove channels that already contain temporal information.
        maps[:, :, 1:] = 0
    elif mode != 'full':
        raise ValueError(mode)
    return maps, mask


class ForecastAdapter(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = (config or ModelConfig()).validate()
        h = self.config.hidden
        self.hmd = nn.GRU(7, h, batch_first=True)
        self.spatial = nn.Sequential(nn.Conv2d(3, 8, 3, stride=2, padding=1), nn.GELU(),
                                     nn.Conv2d(8, 16, 3, stride=2, padding=1), nn.GELU(), nn.AdaptiveAvgPool2d((2, 2)), nn.Flatten())
        self.temporal = nn.GRUCell(66, h)
        dimension = 6 if self.config.method == 'residual' else 1
        self.head = nn.Sequential(nn.Linear(2*h + 1, 64), nn.GELU(), nn.Linear(64, dimension))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def encode(self, inputs):
        _, head = self.hmd(inputs['hmd'])
        maps, mask = ablate_maps(inputs['maps'], inputs['mask'], self.config.ablation)
        batch, frames = maps.shape[:2]
        if self.config.modality == 'hmd':
            maps, mask = torch.zeros_like(maps), torch.zeros_like(mask)
        features = self.spatial(maps.reshape(-1, *maps.shape[2:])).reshape(batch, frames, -1)
        features = torch.cat((features, inputs['timing']), dim=-1)
        state = features.new_zeros((batch, self.config.hidden))
        for j in range(frames):
            update = self.temporal(features[:, j], state)
            state = torch.where(mask[:, j, None], update, state)
        return torch.cat((head[-1], state), dim=-1)

    def forward(self, inputs):
        """Only causal inputs are accepted here; future supervision is a separate object."""
        context = self.encode(inputs)
        horizons = inputs['horizon_s']
        context = context[:, None].expand(-1, horizons.shape[1], -1)
        raw = self.head(torch.cat((context, horizons[..., None]), dim=-1))
        use = inputs['rf_available'].bool() if self.config.modality == 'rf' else torch.ones_like(inputs['rf_available'], dtype=torch.bool)
        base_p, base_q = inputs['base_p'], inputs['base_q']
        prior = inputs['alpha'][:, None].expand_as(horizons)
        if self.config.method == 'residual':
            dp = bounded_vector(raw[..., :3], self.config.max_translation_m)
            dr = bounded_vector(raw[..., 3:], math.radians(self.config.max_rotation_deg))
            p = base_p + dp
            q = unit_quaternion(quaternion_product(rotation_vector_quaternion(dr), base_q))
            gate = prior
            penalty = (dp / self.config.max_translation_m).square().sum(-1) + (dr / math.radians(self.config.max_rotation_deg)).square().sum(-1)
            changed = (dp.abs().sum(-1) + dr.abs().sum(-1)) > 0
        else:
            delta = self.config.max_logit_delta * torch.tanh(raw[..., 0])
            gate = torch.sigmoid(torch.logit(prior.clamp(1e-5, 1-1e-5)) + delta)
            p = (1-gate[..., None])*inputs['focused_p'] + gate[..., None]*inputs['explorer_p']
            q = slerp(inputs['focused_q'], inputs['explorer_q'], gate)
            penalty = (delta / self.config.max_logit_delta).square()
            changed = torch.ones_like(gate, dtype=torch.bool)
        applied = use[:, None] & changed
        return {'p': torch.where(use[:, None, None], p, base_p),
                'q': torch.where(use[:, None, None], q, base_q),
                'gate': gate, 'penalty': penalty * use[:, None], 'applied': applied}

    def specification(self):
        return {'config': asdict(self.config), 'parameters': sum(p.numel() for p in self.parameters()),
                'note': 'HMD control keeps the same module structure; RF-specific parameters are inactive.'}
