"""Create a review-required RF mapping template; never invent synchronization evidence."""
from pathlib import Path
from baf_roi.data import Cache
from .contracts import write_json


def manifest_template(cache_path, maps_directory, output_path):
    if Path(output_path).exists():
        raise ValueError('Refusing to overwrite an existing mapping.')
    cache = Cache(cache_path)
    directory = Path(maps_directory).resolve()
    records = []
    for r in cache.records:
        path = directory / (r['key']+'.npz')
        records.append({'key': r['key'], 'file': str(path) if path.is_file() else None,
                        'expected_map_path': str(path), 'hmd_source': r['source'],
                        'hmd_epoch_start_s': r['epoch_start_s'], 'group_id': r['session'],
                        'scenario': 'unspecified', 'source_clock_domain': 'unix_seconds',
                        'clock_offset_s': 0., 'clock_uncertainty_s': 999.})
    result = {'format_version': 1, 'hmd_cache_id': cache.manifest['cache_id'],
              'hmd_clock_domain': 'unix_seconds', 'synchronization_confirmed': False,
              'allow_exploratory': False, 'records': records,
              'instructions': 'Review file mapping, clock offset/uncertainty and true subject groups. Defaults are deliberately NOT synchronization evidence. Set synchronization_confirmed only after verification.'}
    write_json(output_path, result)
    return {'output': str(output_path), 'records': len(records), 'ready_for_training': False}
