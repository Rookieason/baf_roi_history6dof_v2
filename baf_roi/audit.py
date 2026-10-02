"""Machine-verifiable upstream provenance. Raises if any preserved source changed."""
import ast
import hashlib
import json
from pathlib import Path


def verify(root=None):
    root = Path(root) if root else Path(__file__).resolve().parents[1]
    manifest = json.loads((root / 'provenance.json').read_text())
    count = 0
    for group in ('vendor_files', 'reference_files'):
        for name, expected in manifest[group].items():
            actual = hashlib.sha256((root / name).read_bytes()).hexdigest()
            if actual != expected:
                raise AssertionError(f'Preserved source changed: {name}')
            count += 1
    text = (root / 'baf_roi/upstream_ops.py').read_text()
    nodes = {n.name: n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef)}
    for name, record in manifest['extracted_functions'].items():
        copied = ast.get_source_segment(text, nodes[name])
        if hashlib.sha256(copied.encode()).hexdigest() != record['sha256']:
            raise AssertionError(f'Copied function changed: {name}')
        original_text = (root / record['source']).read_text()
        original = next(n for n in ast.parse(original_text).body if isinstance(n, ast.FunctionDef) and n.name == name)
        if copied != ast.get_source_segment(original_text, original):
            raise AssertionError(f'Copied function no longer equals its original: {name}')
    return {'verified_source_files': count, 'verified_verbatim_functions': len(nodes), 'status': 'PASS'}
