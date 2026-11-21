"""Create a new prepared experiment with center-row timing; preserve all original artifacts."""
import argparse
from copy import deepcopy
from pathlib import Path
import shutil
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from baf_roi.data import Cache
from baf_roi_heatmap.contracts import digest, file_hash, new_directory, read_json, write_json
from baf_roi_heatmap.data import HeatmapSeries
from baf_roi_heatmap.recorded import inspect_session, session_settings


def repair(prepared, raw_root, output):
    source = Path(prepared).resolve()
    m = read_json(source/'manifest.json')
    original_identity = deepcopy(m); original_identity.pop('experiment_id')
    if digest(original_identity) != m['experiment_id']:
        raise ValueError('Source manifest identity changed.')
    cache = Cache(m['pose_cache'])
    settings = cache.manifest['config']['recorded_heatmaps']
    out = new_directory(output)
    maps_dir=out/'maps'; maps_dir.mkdir()
    replacements = {}
    for entry in m['rf_manifest']['records']:
        path = entry.get('file')
        if not path or path in replacements:
            continue
        if file_hash(path) != entry['sha256']:
            raise ValueError('Source RF bytes changed.')
        with np.load(path, allow_pickle=False) as archive:
            arrays={k:archive[k] for k in archive.files}
        import json
        metadata=json.loads(str(arrays['metadata_json'].item()))
        session=metadata['session']
        _, matched, starts, ends, selected, audit, measurement = inspect_session(
            Path(raw_root), session, session_settings(settings,session), return_measurement=True)
        if file_hash(matched) != metadata['matched_csv_sha256']:
            raise ValueError(f'{session}: matched CSV changed since original import.')
        ids=arrays['source_map_index']
        # Require the original selection and dependency bounds, not just a similar count.
        np.testing.assert_array_equal(ids,selected)
        np.testing.assert_array_equal(arrays['support_start_s'],starts[ids])
        np.testing.assert_array_equal(arrays['support_end_s'],ends[ids])
        arrays['measurement_s']=measurement[ids]
        metadata['measurement_time_source']='maximum collector receive time of center-row interpolation inputs; not a hardware capture timestamp'
        metadata['timing_repair_source_sha256']=entry['sha256']
        arrays['metadata_json']=json.dumps(metadata)
        destination=maps_dir/Path(path).name
        np.savez_compressed(destination,**arrays)
        HeatmapSeries(destination,allow_exploratory=True)
        replacements[path]=(str(destination),file_hash(destination))
        print('REPAIRED',session,audit['completion_minus_measurement_p50_p99_s'],flush=True)
    for entry in m['rf_manifest']['records']:
        if entry.get('file'):
            entry['file'],entry['sha256']=replacements[entry['file']]
    shutil.copytree(source/'frozen',out/'frozen')
    for record in m['records']:
        if file_hash(out/'frozen'/f"{record['key']}.npz") != record['sha256']:
            raise ValueError('Frozen supervision changed during copy.')
    old_id=m.pop('experiment_id')
    m['timing_repair']={'source_prepared':str(source),'source_experiment_id':old_id,
                        'change':'RF age uses center-row receive completion; input availability and selection unchanged.'}
    m['experiment_id']=digest(m)
    write_json(out/'manifest.json',m)
    return out


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepared',required=True);p.add_argument('--raw-root',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();print(repair(a.prepared,a.raw_root,a.output))
