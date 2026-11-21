"""Recorder schema, date isolation and future dependency bounds for valder imports."""
import csv
import json
from pathlib import Path

import numpy as np
import pytest

from baf_roi.data import discover_recordings, read_recording
from baf_roi.pose import ypr_to_quat
from baf_roi_heatmap.recorded import dependency_times, inspect_session, selected_map_indices, session_settings

ROOT = Path(__file__).resolve().parents[2]
CONVENTION = 'unreal_world_cm_x_forward_y_right_z_up_pitch_positive_up'


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def sensor_row(t, **updates):
    rot = [20., 10., -5.]
    q = ypr_to_quat(rot)[[1,2,3,0]].tolist()
    hmd = {'position_cm': [100,200,150], 'yaw_deg': rot[0], 'pitch_deg': rot[1], 'roll_deg': rot[2],
           'quaternion_xyzw': q, 'canonical_available': True, 'base': {'pose_available': True},
           'coordinate_convention': CONVENTION}
    hmd.update(updates)
    payload = {'version':2, 'capture_rate_hz':30, 'rx_ts_iso':f'2026-10-31T00:00:00.{int(t*1000):03d}Z', 'hmd':hmd}
    return {'recv_unix':str(1793404800+t), 'payload_json':json.dumps(payload)}


def test_sensor_full_units_orientation_and_receive_clock(tmp_path):
    path = tmp_path/'pose.csv'; write_csv(path, [sensor_row(.01),sensor_row(.06)])
    t, x, _ = read_recording(path, {'position_unit':'cm','payload_format':'sensor_full_v2'})
    np.testing.assert_allclose(x, [[1,2,1.5,20,10,-5]]*2)
    assert t[1]-t[0] == pytest.approx(.05, abs=1e-6)


def test_hmd_delivery_queue_delay_uses_actual_later_availability(tmp_path):
    rows = [sensor_row(.01), sensor_row(.06)]
    rows[0]['recv_unix'] = str(1793404800+.05)
    path = tmp_path/'pose.csv'; write_csv(path, rows)
    t, _, audit = read_recording(path, {'position_unit':'cm','payload_format':'sensor_full_v2'})
    assert t[0] == pytest.approx(1793404800+.05, abs=1e-6)
    assert audit['receiver_to_recorder_delay_p50_max_ms'][1] == pytest.approx(40, abs=.01)


@pytest.mark.parametrize('updates', [{'canonical_available':False}, {'quaternion_xyzw':[0,0,0,1]},
                                     {'coordinate_convention':'unknown'}])
def test_sensor_full_rejects_invalid_or_mismatched_coordinates(tmp_path, updates):
    path = tmp_path/'pose.csv'; write_csv(path, [sensor_row(.01,**updates),sensor_row(.06)])
    with pytest.raises(ValueError):
        read_recording(path, {'position_unit':'cm','payload_format':'sensor_full_v2'})


def test_date_topic_filter_excludes_old_data_even_when_present(tmp_path):
    for session, topic in [('20260117_old','agent6_head'), ('20261031_new','sensor.full.50010'),
                           ('20261101_new','sensor.full.50010'), ('20261031_wrong_topic','agent6_head')]:
        write_csv(tmp_path/'db'/session/topic/'x.csv', [sensor_row(.01)])
    recordings = discover_recordings(tmp_path, {'recording_topic':'sensor.full.50010','session_prefix':'20261031'})
    assert set(recordings) == {'20261031_new'}
    with pytest.raises(ValueError, match='No sensor.full'):
        discover_recordings(tmp_path, {'recording_topic':'sensor.full.50010','session_prefix':'20261031',
                                       'exclude_sessions':['20261031_new']})


def matched_rows(n):
    return [{f'nic{nic}_array_saved':str(i) for nic in range(1,5)} for i in range(n)]


def test_interpolation_includes_future_real_packet_before_use():
    rows = matched_rows(5); rows[1]['nic4_array_saved'] = '-1'
    clocks = {str(nic):{str(i):float(i) for i in range(5)} for nic in range(1,5)}
    starts, ends, _ = dependency_times(rows, clocks, half_window=0)
    assert starts[1] == 0 and ends[1] == 2
    assert ends[0] == 0 and ends[2] == 2


def test_centered_background_is_delayed_until_its_last_packet_arrives():
    rows = matched_rows(9)
    clocks = {str(nic):{str(i):i*.01 for i in range(9)} for nic in range(1,5)}
    starts, ends, _ = dependency_times(rows, clocks, half_window=2)
    assert starts[4] == pytest.approx(.02) and ends[4] == pytest.approx(.06)
    # Additional NIC receive latency must also delay the map.
    clocks['4']['6'] = .5
    assert dependency_times(rows, clocks, half_window=2)[1][4] == .5


def test_center_measurement_time_is_separate_from_centered_background_completion():
    rows = matched_rows(9)
    clocks = {str(nic): {str(i): i*.01 for i in range(9)} for nic in range(1,5)}
    starts, ends, _, measurement = dependency_times(rows, clocks, half_window=2, return_measurement=True)
    assert measurement[4] == pytest.approx(.04)
    assert ends[4] == pytest.approx(.06)
    assert starts[4] <= measurement[4] <= ends[4]


def test_phantom_rows_are_not_selected_as_measurements():
    ends = np.array([.1,.15,.2,.25]); starts = ends-.05
    ids = selected_map_indices(starts, ends, np.array([True,False,True,True]), max_rate_hz=30)
    assert 1 not in ids and np.all(np.diff(ends[ids])>0)


def test_fixed_split_is_complete_disjoint_and_stratified():
    split = json.loads((ROOT/'configs/splits_20261031.json').read_text())
    assert [len(split[s]) for s in ('train','valid','test')] == [12,3,3]
    flat = sum(split.values(),[])
    added = '20260330-082303_record-5-minutes'
    assert len(flat)==len(set(flat))==18 and added in split['train']
    assert '20261031-084133_upperbody_part' not in flat
    for sessions in split.values():
        assert {s.split('_',1)[1] for s in sessions if s != added}=={'upperbody_part','lowerbody_part','longlong'}


def test_additional_session_uses_its_topic_and_is_required(tmp_path):
    cfg = {'recording_topic':'sensor.full.50010', 'session_prefix':'20261031',
           'additional_sessions':['20260330_old'],
           'session_overrides':{'20260330_old':{'recording_topic':'hmd.50000'}}}
    for session, topic in [('20261031_new','sensor.full.50010'), ('20260330_old','hmd.50000'),
                           ('20260330_unselected','hmd.50000')]:
        write_csv(tmp_path/'db'/session/topic/'x.csv', [sensor_row(.01)])
    assert set(discover_recordings(tmp_path, cfg)) == {'20261031_new', '20260330_old'}
    cfg['additional_sessions'].append('missing')
    with pytest.raises(ValueError, match='Additional sessions missing'):
        discover_recordings(tmp_path, cfg)


def test_flat_hmd_uses_collector_clock_and_cm(tmp_path):
    rows = []
    for t in (.01, .06):
        payload = {'pos_x_cm':100, 'pos_y_cm':200, 'pos_z_cm':150,
                   'yaw_deg':20, 'pitch_deg':10, 'roll_deg':-5,
                   'ts_iso':f'2026-10-31T00:00:00.{int(t*1000):03d}Z'}
        rows.append({'recv_unix':str(1793404800+t+.02), 'payload_json':json.dumps(payload)})
    path = tmp_path/'pose.csv'; write_csv(path, rows)
    t, x, audit = read_recording(path, {'position_unit':'cm','payload_format':'hmd_flat_cm'})
    np.testing.assert_allclose(x, [[1,2,1.5,20,10,-5]]*2)
    assert t[0] == pytest.approx(1793404800+.03, abs=1e-6)
    assert audit['timestamp_domain'] == 'collector_receive_unix'


def test_generation_override_preserves_other_sessions():
    cfg = json.loads((ROOT/'configs/recorded_20261031.json').read_text())
    settings = cfg['recorded_heatmaps']
    old = session_settings(settings, '20260330-082303_record-5-minutes')
    base = session_settings(settings, '20261031_fixture')
    assert old['generation']['subcarriers'] == 2025
    assert old['generation']['center_mhz'] == 5570
    assert old['generation']['source_shape'] == base['generation']['source_shape'] == [90,130]
    assert base['generation']['subcarriers'] == settings['generation']['subcarriers'] == 1001
    assert not base.get('allow_missing_run_params', False)


def test_import_uses_collector_clock_and_checks_grid(tmp_path):
    session='20261031_fixture'; folder=tmp_path/'intermediates'/session
    params={'subcarriers':1001,'center_mhz':5775,'antenna_order':[6,7,4,5,2,3,0,1]}
    folder.mkdir(parents=True); (folder/'run_params.json').write_text(json.dumps(params))
    rows=matched_rows(40)
    write_csv(folder/'matched_csi'/f'{session}_matched.csv',rows)
    for nic in range(1,5):
        write_csv(tmp_path/'db'/session/f'csi.rx.{nic}'/'packets.csv',
                  [{'array_saved':str(i),'recv_unix':str(1000+i*.01)} for i in range(40)])
    (folder/'heatmaps').mkdir()
    np.savez_compressed(folder/'heatmaps'/f'{session}_heatmaps.npz',
                        **{str(i):np.ones((90,130)) for i in range(40)})
    settings={'generation':{**params,'past_only':False,'background_half_frames':12,'source_shape':[90,130]},'map_rate_hz':30}
    _,_,starts,ends,selected,_=inspect_session(tmp_path,session,settings)
    assert len(selected)>0 and starts.min()>=1000 and ends.min()>=1000
    settings['generation']['source_shape']=[77,51]
    with pytest.raises(ValueError,match='AoA-ToF shape'):
        inspect_session(tmp_path,session,settings)
    settings['generation']['source_shape']=[90,130]
    (folder/'run_params.json').unlink()
    with pytest.raises(ValueError,match='missing run_params.json'):
        inspect_session(tmp_path,session,settings)
    settings['allow_missing_run_params']=True
    audit=inspect_session(tmp_path,session,settings)[-1]
    assert 'historical run_params.json absent' in audit['parameter_source']
