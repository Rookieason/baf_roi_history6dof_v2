"""Task supervision with a differentiable surrogate; final ROI uses original hard geometry."""
import math
import numpy as np
import torch
import torch.nn.functional as F
from .model import rotation_matrix, unit_quaternion


def soft_visibility(p, q, scene, angular_temperature=.02, distance_temperature_m=.03):
    centers = torch.as_tensor(scene.centers_m, dtype=p.dtype, device=p.device)
    direction = centers[None, None] - p[..., None, :]
    distance = torch.linalg.vector_norm(direction, dim=-1)
    rotation = rotation_matrix(q)
    cfg = scene.cfg
    if cfg['visibility_mode'] == 'legacy_center_cone':
        forward = rotation[..., :, 0]
        cosine = (direction * forward[..., None, :]).sum(-1) / distance.clamp_min(1e-8)
        result = torch.sigmoid((cosine - math.cos(math.radians(cfg['fov_deg']/2))) / angular_temperature)
        result = torch.where(distance <= 1e-11, torch.ones_like(result), result)
    else:
        local = torch.einsum('bhkc,bhcd->bhkd', direction, rotation)
        depth = local[..., 0]
        tx = math.tan(math.radians(cfg['horizontal_fov_deg']/2))
        ty = math.tan(math.radians(cfg['vertical_fov_deg']/2))
        result = torch.sigmoid((depth-cfg.get('near_m', .01)) / distance_temperature_m)
        result = result * torch.sigmoid((depth*tx-local[..., 1].abs()) / distance_temperature_m)
        result = result * torch.sigmoid((depth*ty-local[..., 2].abs()) / distance_temperature_m)
        if cfg.get('far_m') is not None:
            result = result * torch.sigmoid((cfg['far_m']-depth) / distance_temperature_m)
    if cfg.get('max_distance_m') is not None:
        result = result * torch.sigmoid((cfg['max_distance_m']-distance) / distance_temperature_m)
    return result


def task_loss(prediction, supervision, scene, roi_weight=.25, regularization=.001,
              *, objective='pose_roi', sample_weight=None):
    if objective not in ('pose_roi', 'roi_iou'):
        raise ValueError('objective must be pose_roi or roi_iou.')
    p, q = prediction['p'], unit_quaternion(prediction['q'])
    target_p, target_q = supervision['target_p'], unit_quaternion(supervision['target_q'])
    translation = F.smooth_l1_loss(p / .1, target_p / .1, reduction='none').mean((-2, -1))
    dot = (q * target_q).sum(-1).clamp(-1, 1)
    # Sign invariant, finite-gradient small-angle surrogate, approximately angle^2 / (10deg)^2.
    rotation = (4 * (1-dot.square()) / math.radians(10)**2).mean(-1)
    probability = soft_visibility(p, q, scene)
    truth = supervision['target_mask']
    tp = (probability * truth).sum(-1)
    fp = (probability * (1-truth)).sum(-1)
    fn = ((1-probability) * truth).sum(-1)
    tversky = 1 - (tp+1e-6) / (tp+.3*fp+.7*fn+1e-6)
    if objective == 'roi_iou':
        # Match the unexpanded hard-IoU checkpoint criterion, including empty GT.
        # The legacy Tversky (.3 FP / .7 FN) optimizes a different tradeoff.
        roi = 1 - (tp+1e-6)/(tp+fp+fn+1e-6)
    else:
        roi = tversky
    roi = torch.where(truth.sum(-1) > 0, roi, probability.mean(-1)).mean(-1)
    value = roi if objective == 'roi_iou' else translation + rotation + roi_weight*roi
    value = value + regularization*prediction['penalty'].mean(-1)
    if sample_weight is not None:
        value = value * sample_weight.to(value)
    return value.mean()


def numpy_prediction(prediction, original_base_p, original_base_q):
    """Restore original float64 baseline on fallback/zero-residual, including boundary cases."""
    p = prediction['p'].detach().cpu().numpy().astype(np.float64)
    q = prediction['q'].detach().cpu().numpy().astype(np.float64)
    applied = prediction['applied'].detach().cpu().numpy()
    return np.where(applied[..., None], p, original_base_p), np.where(applied[..., None], q, original_base_q)


def motion_task_loss(prediction, supervision, scene, roi_weight=.25, regularization=.001,
                     *, objective='pose_roi', sample_weight=None, inputs):
    """Pose + ROI with a label-only penalty for regression against frozen BAF.

    CSI may contain unrelated body motion. Penalize excess pose error and missed
    visibility instead of rewarding every large RF response. This is a training
    preference, not a per-example or test-set non-regression guarantee.
    """
    value = task_loss(prediction, supervision, scene, roi_weight, regularization,
                      objective=objective, sample_weight=sample_weight)
    target_p, target_q = supervision['target_p'], unit_quaternion(supervision['target_q'])
    pose_excess = (torch.linalg.vector_norm(prediction['p']-target_p, dim=-1)
                   - torch.linalg.vector_norm(inputs['base_p']-target_p, dim=-1)).clamp_min(0)/.1
    rotation_error = lambda q: 4*(1-(unit_quaternion(q)*target_q).sum(-1).square().clamp(0, 1))/math.radians(10)**2
    rotation_excess = (rotation_error(prediction['q'])-rotation_error(inputs['base_q'])).clamp_min(0)
    probability = soft_visibility(prediction['p'], prediction['q'], scene)
    with torch.no_grad():
        baseline = soft_visibility(inputs['base_p'], inputs['base_q'], scene)
    truth = supervision['target_mask']
    missed_excess = ((baseline-probability)*truth).sum(-1).clamp_min(0)/truth.sum(-1).clamp_min(1)
    regret = (pose_excess+rotation_excess+missed_excess).mean(-1)
    if sample_weight is not None:
        regret = regret*sample_weight.to(regret)
    return value+.1*regret.mean()
