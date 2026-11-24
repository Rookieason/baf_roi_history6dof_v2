"""The public method names are the packages that own their algorithms."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Method:
    name: str
    description: str
    uses_heatmap: bool = False
    requires_baf: bool = False


METHODS = {
    method.name: method for method in (
        Method('baf_roi', 'Original BAF-Pose confidence and two pose experts'),
        Method('motion_transformer_roi', 'History-only Motion Transformer'),
        Method('baf_roi_heatmap', 'CSI-conditioned BAF motion experts and per-step routing', True, True),
        Method('baf_roi_heatmap_alpha', 'CSI correction of BAF per-step alpha only', True, True),
    )
}


def resolve_methods(names=None, default=None):
    """No selection means every method; preserve an explicit user's order."""
    if names is None or names == ['all']:
        return list(METHODS if default is None else default)
    if not names or 'all' in names or len(set(names)) != len(names):
        raise ValueError('Choose distinct methods, or use all alone.')
    unknown = set(names) - METHODS.keys()
    if unknown:
        raise ValueError(f'Unknown methods: {sorted(unknown)}')
    return list(names)

