"""Scientific reporting regressions shared by all four prediction methods."""
import numpy as np
import pytest
from torch.utils.data import DataLoader

from baf_roi.evaluate import Metrics
from baf_roi_heatmap.contracts import InputConfig, write_json
from baf_roi_heatmap.data import empty_window
from roi_experiments.evaluation import balanced_summary, compare, roi_counts, summarize


def test_common_metrics_equal_original_ppt_metrics_including_empty_sets():
    prediction = np.array([[0, 0, 0, 0], [1, 0, 0, 0], [1, 1, 0, 0], [0, 1, 0, 0]], bool)
    truth = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 1, 0], [0, 1, 0, 0]], bool)
    tr, rot = np.array([0., .01, .03, .1]), np.array([0., 2., 7., 12.])
    original = Metrics()
    counts, ious = original.add(prediction, truth, tr, rot)
    rows = [dict(zip(('tp', 'fp', 'fn', 'tn'), c)) | {
        'iou': float(iou), 'translation_error_m': float(t), 'rotation_error_deg': float(r)}
        for c, iou, t, r in zip(counts, ious, tr, rot)]
    shared = summarize(rows)
    for name, value in original.result().items():
        if value is None:
            assert shared[name] is None
        else:
            assert shared[name] == pytest.approx(value)
    assert shared['coverage_micro'] == shared['recall_micro']
    # A false positive on empty GT must remain in the nonempty-union statistic.
    assert shared['iou_mean_nonempty_union'] == pytest.approx((0.+1/3+1.)/3)
    assert shared['iou_mean_nonempty_gt'] == pytest.approx((1/3+1.)/2)


def test_empty_coverage_matches_recall_without_claiming_a_defined_quality_constraint():
    row = {'tp': 0, 'fp': 0, 'fn': 0, 'tn': 3, 'iou': 1.,
           'translation_error_m': 0., 'rotation_error_deg': 0.}
    result = summarize([row])
    assert result['coverage_micro'] == result['recall_micro'] == 0.
    assert not result['coverage_defined']
    assert result['iou_mean_nonempty_union'] is None


def test_recording_macro_does_not_overweight_a_file_split_by_sensor_gaps():
    rows = []
    for segment, recording, good in [('a1', 'a', True), ('a2', 'a', True), ('b1', 'b', False)]:
        rows.append({'record': segment, 'recording_id': recording, 'scenario': 'walk',
                     'tp': int(good), 'fp': 0, 'fn': int(not good), 'tn': 1,
                     'iou': float(good), 'translation_error_m': 0., 'rotation_error_deg': 0.})
    report = balanced_summary(rows)
    assert report['pooled']['iou_mean'] == pytest.approx(2/3)
    assert report['recording_macro']['iou_mean'] == .5
    assert report['recordings'] == 2


@pytest.mark.parametrize('field,left,right', [
    ('delivery_delay_s', 0., .1), ('targets_sha256', 'truth-a', 'truth-b'),
    ('origin_keys_sha256', 'same-count-a', 'same-count-b'), ('rf_cohort_sha256', 'rf-a', 'rf-b'),
])
def test_comparison_refuses_equal_counts_with_different_experimental_conditions(tmp_path, field, left, right):
    common = {'experiment_id': 'same', 'scene_id': 'same', 'split': 'test', 'horizons_ms': [50.]}
    for folder, value in [('a', left), ('b', right)]:
        write_json(tmp_path/folder/'report.json', common | {field: value})
    with pytest.raises(ValueError, match=field):
        compare(tmp_path/'a', tmp_path/'b', tmp_path/'comparison.json')


def test_missing_rf_batches_collate_across_supported_numpy_and_torch_versions():
    # numpy.bool_ scalars fail default_collate in some supported NumPy/PyTorch pairs.
    batch = next(iter(DataLoader([empty_window(InputConfig()), empty_window(InputConfig())], batch_size=2)))
    assert batch['rf_available'].tolist() == [False, False]
    assert tuple(batch['maps'].shape) == (2, 8, 3, 24, 24)
