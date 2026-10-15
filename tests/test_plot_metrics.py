import csv
import pytest
from baf_roi.plot_metrics import read_metrics, plot_metrics


def write_metrics(path, rows):
    with path.open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_plot_uses_mean_iou_and_normalized_counts(tmp_path):
    path = tmp_path / 'metrics.csv'
    write_metrics(path, [dict(horizon_ms=1000, n_origins=10, tp=8, fp=12, fn=2,
                             iou_mean=.75, iou_micro=8 / 22),
                         dict(horizon_ms=200, n_origins=10, tp=0, fp=0, fn=0,
                              iou_mean=1., iou_micro=1.)])
    rows = read_metrics(path)
    assert [r['horizon_ms'] for r in rows] == [200, 1000]
    assert rows[1]['iou_mean'] == .75
    assert rows[1]['missed'] == .2
    assert rows[1]['redundant'] == 1.2
    assert rows[1]['selected'] == 2.
    assert rows[1]['coverage'] == .8
    assert rows[0]['coverage'] == 0.
    files = plot_metrics([path], tmp_path / 'plots', ['BAF-ROI'], True)
    assert all(p.stat().st_size > 1000 for p in files)
    with pytest.raises(ValueError, match='already exist'):
        plot_metrics([path], tmp_path / 'plots', ['BAF-ROI'])


@pytest.mark.parametrize('change', [dict(n_origins=0), dict(iou_mean=float('nan')),
                                    dict(fn=-1), dict(horizon_ms=1001)])
def test_invalid_plot_metrics_rejected(tmp_path, change):
    row = dict(horizon_ms=200, n_origins=10, tp=8, fp=12, fn=2, iou_mean=.75)
    path = tmp_path / 'metrics.csv'
    write_metrics(path, [dict(row, **change)])
    with pytest.raises(ValueError):
        read_metrics(path)


def test_incompatible_model_comparisons_rejected(tmp_path):
    row = dict(horizon_ms=200, n_origins=10, tp=8, fp=12, fn=2, iou_mean=.75)
    a, b = tmp_path / 'a.csv', tmp_path / 'b.csv'
    write_metrics(a, [row])
    write_metrics(b, [dict(row, horizon_ms=400)])
    with pytest.raises(ValueError, match='same horizons'):
        plot_metrics([a, b], tmp_path / 'plots', ['BAF-ROI', 'Motion Transformer'])
    write_metrics(b, [dict(row, n_origins=5)])
    with pytest.raises(ValueError, match='same number of origins'):
        plot_metrics([a, b], tmp_path / 'plots', ['BAF-ROI', 'Motion Transformer'])
    assert not (tmp_path / 'plots').exists()
