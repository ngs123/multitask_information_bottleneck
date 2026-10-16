"""Extend a compatible run into a new, self-contained output directory."""
import copy
import json
import shutil
from pathlib import Path


def comparable_config(config):
    result = copy.deepcopy(config)
    for key in ('representations', 'device', 'robustness', '_diagnostics_only', 'seeds'):
        result.pop(key, None)
    result.get('gaussian', {}).pop('rates', None)
    result.get('quantized', {}).pop('bits', None)
    return result


def validate_merge(previous, config):
    previous = Path(previous)
    old = json.loads((previous / 'config.json').read_text())
    if not set(old.get('seeds', [old['seed']])).issubset(config.get('seeds', [config['seed']])):
        raise ValueError('Merge seed list must include all previous seeds')
    if comparable_config(old) != comparable_config(config):
        raise ValueError('Cannot merge runs with different tasks, seeds, pretexts, fitting modes, data or training settings. Omit --config to inherit the original configuration.')
    rows = json.loads((previous / 'results.json').read_text())
    if not rows:
        raise ValueError('Previous run has no completed results')
    keys = [cell_key(r) for r in rows]
    if len(set(keys)) != len(keys):
        raise ValueError('Previous run has duplicate evaluation cells')
    for row in rows:
        directory = previous / row['directory']
        if not directory.resolve().is_relative_to(previous.resolve()) or not directory.is_dir():
            raise ValueError(f'Invalid/missing previous artifact directory: {directory}')
    return rows


def cell_key(row):
    return row['mode'], row['candidate'], row['budget'], row['seed']


def import_previous(previous, out, manifest, rows):
    previous, out = Path(previous), Path(out)
    old_manifest = json.loads((previous / 'feature_manifest.json').read_text())
    # Device is merely execution metadata. All feature-generating settings must match.
    old_manifest.pop('device', None)
    manifest = dict(manifest); manifest.pop('device', None)
    if old_manifest != manifest:
        raise ValueError('Cached feature manifest differs from the previous run; use the same cache/data settings')
    for row in rows:
        dest = out / row['directory']
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(previous / row['directory'], dest)
    if (previous / 'uncompressed').is_dir():
        shutil.copytree(previous / 'uncompressed', out / 'uncompressed')
    predictions = json.loads((previous / 'gaussian_predicted_distortions.json').read_text())
    certificates = json.loads((previous / 'certificates.json').read_text())
    return {'rows': rows, 'predicted': {float(k): v for k, v in predictions.items()},
            'certificates': certificates}


def combined_config(config, rows):
    result = copy.deepcopy(config)
    result['representations'] = sorted({r['mode'] for r in rows})
    for mode, section, key in [('gaussian', 'gaussian', 'rates'), ('quantized', 'quantized', 'bits')]:
        budgets = sorted({r['budget'] for r in rows if r['mode'] == mode})
        if budgets:
            result[section][key] = budgets
    return result
