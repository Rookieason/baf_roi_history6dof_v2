"""Regression checks for center-frame age, recording weighting and ROI selection."""
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from baf_roi.pose import ypr_to_quat
from baf_roi.roi import Scene
from baf_roi_heatmap.contracts import InputConfig
from baf_roi_heatmap.data import HeatmapSeries, encode_window
from baf_roi_heatmap.losses import task_loss
from baf_roi_heatmap.training_utils import CheckpointSelection, recording_weights


def delayed_maps(path, *, measurement=True, invalid=False):
    end = np.arange(.3, 1.6, .05)
    values = dict(maps=np.ones((len(end), 4, 4), np.float32), valid=np.ones(len(end), bool),
                  support_start_s=end-.25, support_end_s=end, available_at_s=end,
                  theta_deg=np.arange(4), tau_s=np.arange(4)*1e-9,
                  metadata_json=json.dumps({'format_version': 1, 'representation': 'aoa_tof', 'unit': 'db',
                                            'clock_domain': 'unix_seconds', 'availability_source': 'test',
                                            'physical_axes_verified': True}))
    if measurement:
        values['measurement_s'] = end + .01 if invalid else end-.13
    np.savez(path, **values)
    return HeatmapSeries(path)


def test_center_frame_age_does_not_remove_background_availability_delay(tmp_path):
    cfg = InputConfig(size=4)
    old = delayed_maps(tmp_path/'old.npz', measurement=False)
    new = delayed_maps(tmp_path/'new.npz')
    x = encode_window(old, 1.2, {'median': 0., 'scale': 1.}, cfg)
    y = encode_window(new, 1.2, {'median': 0., 'scale': 1.}, cfg)
    np.testing.assert_array_equal(x['maps'], y['maps'])
    np.testing.assert_array_equal(x['mask'], y['mask'])
    np.testing.assert_allclose(y['timing'][:, 0]-x['timing'][:, 0], .13/cfg.max_age_s, atol=1e-6)
    legacy = encode_window(new, 1.2, {'median': 0., 'scale': 1.}, cfg, age_basis='support_end')
    np.testing.assert_array_equal(legacy['timing'], x['timing'])
    with pytest.raises(ValueError, match='requires explicit center'):
        encode_window(old, 1.2, {'median': 0., 'scale': 1.}, cfg, age_basis='center_measurement')
    # A measurement time in the past does not permit using undelivered inputs.
    assert not encode_window(new, 1.2, {'median': 0., 'scale': 1.}, cfg, delivery_delay_s=2.)['rf_available']
    with pytest.raises(ValueError, match='inside its raw support'):
        delayed_maps(tmp_path/'invalid.npz', invalid=True)


def test_macro_weights_give_each_recording_equal_total_weight_after_segmentation():
    dataset = SimpleNamespace(records=[{'count': 800}, {'count': 200}, {'count': 10}],
                              recording_id=lambda ri: ['long', 'long', 'short'][ri])
    class Dataset:
        records = dataset.records
        recording_id = staticmethod(dataset.recording_id)
        def __len__(self): return 1010
    w = recording_weights(Dataset(), 'recording')
    assert float(w[0]*800+w[1]*200) == pytest.approx(float(w[2]*10))
    assert float((w*torch.tensor([800, 200, 10])).sum()/1010) == pytest.approx(1.)


def test_roi_objective_respects_cone_roll_invariance_while_legacy_pose_loss_does_not():
    scene = Scene(np.array([[2., 0, 0], [0., 2, 0]]), ['front', 'side'],
                  {'visibility_mode': 'legacy_center_cone', 'fov_deg': 90.})
    p = torch.zeros(1, 1, 3)
    q0 = torch.tensor(ypr_to_quat([[0., 0, 0]])[None], dtype=torch.float32)
    qr = torch.tensor(ypr_to_quat([[0., 0, 80.]])[None], dtype=torch.float32)
    y = {'target_p': p, 'target_q': q0, 'target_mask': torch.tensor([[[1., 0.]]])}
    a = {'p': p, 'q': q0, 'penalty': torch.zeros(1, 1)}
    b = {**a, 'q': qr}
    torch.testing.assert_close(task_loss(a,y,scene,objective='roi_iou'), task_loss(b,y,scene,objective='roi_iou'))
    assert task_loss(b,y,scene,objective='pose_roi') > task_loss(a,y,scene,objective='pose_roi')+1


def test_baseline_retained_is_explicit_and_trained_checkpoints_are_still_available(tmp_path):
    selection = CheckpointSelection(tmp_path, 'adapter.pt')
    for epoch, score in [(0, .8), (1, .7), (2, .75), (3, .6)]:
        status = selection.update({'selected_epoch': epoch, 'validation_recording_macro_iou': score,
                                   'model': {'weight': torch.tensor([epoch])}})
    assert status['selection_status'] == 'baseline_retained'
    assert status['selected_epoch'] == 0 and status['best_trained_epoch'] == 2
    assert torch.load(tmp_path/'adapter.pt', weights_only=True)['selected_epoch'] == 0
    assert torch.load(tmp_path/'best_trained.pt', weights_only=True)['selected_epoch'] == 2
    assert torch.load(tmp_path/'last.pt', weights_only=True)['selected_epoch'] == 3
