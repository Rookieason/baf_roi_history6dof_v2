"""Thesis Sections 3.2/5.1 proxy labels reconstructed from history alone.
The supplied ZIP has no script generating its *_with_conf_supervision CSVs.
F=0, E=1, ambiguous=-1. None of these labels is an inference input.
"""
import numpy as np


def motion_features(history, dt=1 / 30):
    x = np.asarray(history, dtype=np.float64)
    if x.shape != (30, 6) or not np.all(np.isfinite(x)):
        raise ValueError('Behavior supervision requires finite [30,6] history.')
    speed = np.linalg.norm(np.diff(x[:, :3], axis=0), axis=1) / dt
    yaw_rate = ((np.diff(x[:, 3]) + 180) % 360 - 180) / dt
    # Paper does not specify how exact zeros are counted. Explicit convention:
    # remove exact-zero rates, then count sign changes; no denoising threshold.
    signed = np.sign(yaw_rate)
    signed = signed[signed != 0]
    crossings = np.count_nonzero(signed[1:] != signed[:-1])
    duration = (len(x) - 1) * dt  # actual elapsed duration, 29/30 s
    return np.array([speed.mean(), np.quantile(np.abs(yaw_rate), .9),
                     crossings / duration, np.mean(np.abs(yaw_rate) > 60)], np.float64)


def behavior(history):
    speed, q90, reversal, hi = motion_features(history)
    tr = min(1., speed / .2)
    rot = .5 * min(1., q90 / 180) + .3 * min(1., reversal / 3) + .2 * min(1., hi / .5)
    score = float(1 / (1 + np.exp(-6 * (.6 * rot + .4 * tr - .5))))
    return (1 if score >= .55 else 0 if score <= .45 else -1), score
