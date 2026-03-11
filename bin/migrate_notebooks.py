#!/usr/bin/env python3
"""Migrate Jupyter notebooks from the old flat watershed_workflow namespace
to the new hierarchical namespace introduced in v2.x.

Usage
-----
    python migrate_notebooks.py notebook.ipynb [notebook2.ipynb ...]
    python migrate_notebooks.py examples/          # all notebooks under a directory
    python migrate_notebooks.py --dry-run examples/

The script rewrites notebooks in place and prints a summary of changes.
Original files are backed up as <name>.ipynb.bak unless --no-backup is given.
"""

import argparse
import glob
import json
import os
import re
import shutil
import sys


# ---------------------------------------------------------------------------
# Mapping: old module name (the flat part) -> new dotted path
# Ordered longest-first so that e.g. `land_cover_properties` is matched
# before any shorter prefix could fire.
# ---------------------------------------------------------------------------
MODULE_MAP = [
    # Already-updated deep paths that should now be flattened to subpackage level.
    # These must come first (longest match) so they fire before the shorter old names.
    ('mesh.triangulation',         'watershed_workflow.mesh'),
    ('mesh.river_mesh',            'watershed_workflow.mesh'),
    ('mesh.condition',             'watershed_workflow.mesh'),
    ('mesh.regions',               'watershed_workflow.mesh'),
    ('mesh.mesh',                  'watershed_workflow.mesh'),
    ('hydro.river',                'watershed_workflow.hydro'),
    ('hydro.watershed',            'watershed_workflow.hydro'),
    ('hydro.hydrography',          'watershed_workflow.hydro'),
    ('hydro.resampling',           'watershed_workflow.hydro'),
    ('hydro.angles',               'watershed_workflow.hydro'),
    ('io.plot',                    'watershed_workflow.plot'),
    ('io.colors',                  'watershed_workflow.plot'),
    ('io.dynamic_colormaps',       'watershed_workflow.plot'),
    ('io.vtk',                     'watershed_workflow.io'),
    ('io.hdf5',                    'watershed_workflow.io'),
    ('io.ui',                      'watershed_workflow.io'),
    ('io.bin_utils',               'watershed_workflow.io'),
    ('utils.crs',                  'watershed_workflow.crs'),
    ('utils.config',               'watershed_workflow.utils'),
    ('utils.warp',                 'watershed_workflow.utils'),
    ('utils.tinytree',             'watershed_workflow.utils'),
    ('utils.data',                 'watershed_workflow.utils'),
    ('utils.utils',                'watershed_workflow.utils'),
    # properties stays namespaced — keep sub-submodule paths
    ('properties.soil',            'watershed_workflow.properties.soil'),
    ('properties.land_cover',      'watershed_workflow.properties.land_cover'),
    ('properties.meteorology',     'watershed_workflow.properties.meteorology'),
    ('properties.stream_light',    'watershed_workflow.properties.stream_light'),
    # Old flat names (pre-reorganization)
    ('crs',                        'watershed_workflow.crs'),
    ('config',                     'watershed_workflow.utils'),
    ('warp',                       'watershed_workflow.utils'),
    ('tinytree',                   'watershed_workflow.utils'),
    ('data',                       'watershed_workflow.utils'),
    ('river_tree',                 'watershed_workflow.hydro'),
    ('split_hucs',                 'watershed_workflow.hydro'),
    ('hydrography',                'watershed_workflow.hydro'),
    ('resampling',                 'watershed_workflow.hydro'),
    ('angles',                     'watershed_workflow.hydro'),
    ('triangulation',              'watershed_workflow.mesh'),
    ('river_mesh',                 'watershed_workflow.mesh'),
    ('condition',                  'watershed_workflow.mesh'),
    ('regions',                    'watershed_workflow.mesh'),
    ('plot',                       'watershed_workflow.plot'),
    ('colors',                     'watershed_workflow.plot'),
    ('dynamic_colormaps',          'watershed_workflow.plot'),
    ('vtk_io',                     'watershed_workflow.io'),
    ('bin_utils',                  'watershed_workflow.io'),
    ('soil_properties',            'watershed_workflow.properties.soil'),
    ('land_cover_properties',      'watershed_workflow.properties.land_cover'),
    ('meteorology',                'watershed_workflow.properties.meteorology'),
    ('stream_light',               'watershed_workflow.properties.stream_light'),
    # ambiguous — must be last
    ('utils',                      'watershed_workflow.utils'),
    ('mesh',                       'watershed_workflow.mesh'),
    ('io',                         'watershed_workflow.io'),
    ('ui',                         'watershed_workflow.io'),
]

# Build one big alternation, longest-match first (already ordered above).
# Each old name is the bare module name after `watershed_workflow.`.
_OLD_NAMES = [old for old, _ in MODULE_MAP]
_NAME_TO_NEW = {old: new for old, new in MODULE_MAP}

# Regex that matches any of the old import forms in a single pass:
#
#   import watershed_workflow.OLD
#   from watershed_workflow.OLD import ...
#   watershed_workflow.OLD.attr   (attribute access)
#
# The alternation is tried longest-first so `land_cover_properties` wins
# over `land_cover` if we ever add the latter.
_PATTERN = re.compile(
    r'(?P<kw>import |from )?'           # optional keyword
    r'watershed_workflow\.'             # package prefix
    r'(?P<old>' + '|'.join(re.escape(n) for n in _OLD_NAMES) + r')'
    r'(?P<rest>(?=\s|$|;|\.))',         # followed by whitespace/end/dot
    re.MULTILINE,
)

# ---------------------------------------------------------------------------
# Class rewrites: long form -> canonical short form
# Applied as a second pass after module paths are fixed.
# ---------------------------------------------------------------------------
_CLASS_REWRITES = [
    # After module rewrites, all paths are in new form — collapse to short canonical.
    (r'watershed_workflow\.hydro\.river\.River',         'watershed_workflow.River'),
    (r'watershed_workflow\.hydro\.watershed\.Watershed', 'watershed_workflow.Watershed'),
    (r'watershed_workflow\.mesh\.mesh\.Mesh2D',          'watershed_workflow.Mesh2D'),
    (r'watershed_workflow\.mesh\.mesh\.Mesh3D',          'watershed_workflow.Mesh3D'),
    (r'watershed_workflow\.crs\.CRS',                    'watershed_workflow.CRS'),
    (r'watershed_workflow\.utils\.crs\.CRS',             'watershed_workflow.CRS'),
    # Subpackage-level (already-updated notebooks)
    (r'watershed_workflow\.hydro\.River',                'watershed_workflow.River'),
    (r'watershed_workflow\.hydro\.Watershed',            'watershed_workflow.Watershed'),
    (r'watershed_workflow\.mesh\.Mesh2D',                'watershed_workflow.Mesh2D'),
    (r'watershed_workflow\.mesh\.Mesh3D',                'watershed_workflow.Mesh3D'),
]
_CLASS_PATTERNS = [(re.compile(p), r) for p, r in _CLASS_REWRITES]


def _replace(m: re.Match) -> str:
    old = m.group('old')
    new = _NAME_TO_NEW[old]
    kw = m.group('kw') or ''
    rest = m.group('rest')
    return f'{kw}{new}{rest}'


def _migrate_source(src: str) -> tuple[str, int]:
    """Apply all migrations to a single source string.

    Returns (new_src, number_of_substitutions).
    """
    n = 0
    new_src = src

    # Pass 1: module path rewrites (expands flat names to new subpackage paths)
    new_src, k = _PATTERN.subn(_replace, new_src)
    n += k

    # Pass 2: SplitHUCs rename
    if 'SplitHUCs' in new_src:
        count = new_src.count('SplitHUCs')
        new_src = new_src.replace('SplitHUCs', 'Watershed')
        n += count

    # Pass 2b: function/class renames
    if 'setup_logging' in new_src:
        count = new_src.count('setup_logging')
        new_src = new_src.replace('setup_logging', 'setupLogging')
        n += count

    for old, new in [
        ('ComputeTargetLengthByDistanceToShape', 'TargetLengthByDistance'),
        ('ComputeTargetLengthByCallable',        'TargetLengthByCallable'),
        ('ComputeTargetLengthByProperty',        'TargetLengthByProperty'),
        ('ComputeTargetLengthFixed',             'TargetLengthFixed'),
        ('ComputeTargetLength',                  'TargetLength'),
        # vtk function renames (old flat .read/.write, or intermediate readVTK/writeVTK)
        ('.vtk.read(',                           '.readMeshVTK('),
        ('.vtk.write(',                          '.writeMeshVTK('),
        ('readVTK(',                             'readMeshVTK('),
        ('writeVTK(',                            'writeMeshVTK('),
        # warp function renames (flat names -> camelCase verbs)
        ('.warp.shplys(',                        '.warp.warpShplys('),
        ('.warp.shply(',                         '.warp.warpShply('),
        ('.warp.points(',                        '.warp.warpPoints('),
        ('.warp.bounds(',                        '.warp.warpBounds('),
        ('.warp.dataset(',                       '.warp.warpDataset('),
        ('.warp.xy(',                            '.warp.warpXY('),
    ]:
        if old in new_src:
            count = new_src.count(old)
            new_src = new_src.replace(old, new)
            n += count

    # Pass 3: class rewrites — collapse fully-qualified paths to short canonical form
    # By now all module paths are in new form, so e.g. split_hucs.Watershed has
    # already become hydro.watershed.Watershed, and we can match uniformly.
    for pattern, replacement in _CLASS_PATTERNS:
        new_src, k = pattern.subn(replacement, new_src)
        n += k

    # Pass 4: deduplicate bare `import X` lines that collapsed to the same target.
    # Only removes exact-duplicate lines of the form `import <module>` (no `from`).
    seen_imports: set[str] = set()
    deduped_lines = []
    for line in new_src.splitlines(keepends=True):
        stripped = line.strip()
        if re.match(r'^import\s+\S+$', stripped):
            if stripped in seen_imports:
                n += 1
                continue
            seen_imports.add(stripped)
        deduped_lines.append(line)
    new_src = ''.join(deduped_lines)

    return new_src, n


def migrate_notebook(path: str, dry_run: bool = False, backup: bool = True) -> int:
    """Migrate a single notebook. Returns total number of substitutions."""
    with open(path) as f:
        nb = json.load(f)

    total = 0
    for cell in nb.get('cells', []):
        if cell.get('cell_type') != 'code':
            continue
        src = ''.join(cell['source'])
        new_src, n = _migrate_source(src)
        if n:
            total += n
            if not dry_run:
                cell['source'] = new_src.splitlines(keepends=True)

    if total and not dry_run:
        if backup:
            shutil.copy2(path, path + '.bak')
        with open(path, 'w') as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)
            f.write('\n')

    return total


def collect_notebooks(paths: list[str]) -> list[str]:
    notebooks = []
    for p in paths:
        if os.path.isdir(p):
            notebooks.extend(sorted(glob.glob(os.path.join(p, '**', '*.ipynb'), recursive=True)))
        elif p.endswith('.ipynb'):
            notebooks.append(p)
        else:
            print(f'Warning: skipping {p}', file=sys.stderr)
    return notebooks


def main():
    parser = argparse.ArgumentParser(
        description='Migrate watershed_workflow notebooks to the new hierarchical namespace.')
    parser.add_argument('paths', nargs='+', help='Notebook files or directories to migrate')
    parser.add_argument('--dry-run', action='store_true',
                        help='Print what would change without modifying files')
    parser.add_argument('--no-backup', action='store_true',
                        help='Do not create .bak backup files')
    args = parser.parse_args()

    notebooks = collect_notebooks(args.paths)
    if not notebooks:
        print('No notebooks found.')
        return

    total_files = 0
    total_subs = 0
    for nb_path in notebooks:
        n = migrate_notebook(nb_path, dry_run=args.dry_run, backup=not args.no_backup)
        prefix = '[dry-run] ' if args.dry_run else ''
        if n:
            total_files += 1
            total_subs += n
            print(f'{prefix}{nb_path}: {n} substitution(s)')
        else:
            print(f'{nb_path}: no changes needed')

    print(f'\nDone. {total_subs} substitution(s) across {total_files} notebook(s).')


if __name__ == '__main__':
    main()
