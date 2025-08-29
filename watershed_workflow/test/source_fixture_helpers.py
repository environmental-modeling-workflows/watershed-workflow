"""One major problem of testing is that we need some real data, but
real data comes in big files (~GB) that we don't want to download, but
also don't want to save to our repo.

This helper works with source_fixtures.FileManagerMockNHDPlusSave to
save all needed HUCs to a couple of files, which can be put into the
repo, and are a much smaller file dataset.

To update the saved work files:

1. In source_fixtures.py:sources, replace FileManagerMockNHDPlus with FileManagerMockNHDPlusSave.
2. Run the tests, e.g. pytest watershed_workflow/test/
3. Run this script, e.g. python ./source_fixture_helpers.py within the test directory!
4. Change source_fixtures.py:sources back to FileManagerMockNHDPlus.

Note, you probably need to also commit the updated files to the repo!
"""

import fiona
import pickle
import watershed_workflow.io
import watershed_workflow.sources
import watershed_workflow.crs
import watershed_workflow.utils


def read_and_process_dump(pkl_dump_file):
    """Helper function to read, process, and save a new mock HUC file."""

    # read the pkl file saved when the tests were run
    with open(pkl_dump_file, 'rb') as fid:
        d = pickle.load(fid)

    # collect all HUCs needed
    nhdp = watershed_workflow.sources.FileManagerNHDPlus()
    hucs = dict()
    hydro_reqs = list()
    for huc, v in d.items():
        for level, _ in v.items():
            if level != 'hydro':
                print(f'reading {huc} level {level}')
                profile, these = nhdp.get_hucs(huc, int(level))
                name_key = f'HUC{level}'
                for this in these:
                    name = this['properties'][name_key]
                    hucs[name] = this
            else:
                hydro_reqs.append(huc)

    # convert to shapely
    for h, v in hucs.items():
        hucs[h] = watershed_workflow.utils.create_shply(v)
    for h, v in hucs.items():
        # normalize the properties
        v.properties = dict(HUC=h)

    # get the crs
    crs = watershed_workflow.crs.from_fiona(profile['crs'])

    # get the hydro, bounding as much as possible
    hydro = dict()
    for huc in hydro_reqs:
        bounds = hucs[huc].bounds
        print(f'reading {huc} hydro data')
        _, hydro[huc] = nhdp.get_hydro(huc, bounds, crs, include_catchments=False)

    # convert to shply -- and simplify, I think this should be safe!
    for h, v in hydro.items():
        print(v[0].keys())
        rivers = [watershed_workflow.utils.create_shply(r) for r in v]
        hydro[h] = [r.simplify(50) for r in rivers]

    # write the files
    watershed_workflow.io.write_to_shapefile('watershed_workflow/test/fixture_data/hucs.shp',
                                             list(hucs.values()), crs)
    for h, v in hydro.items():
        watershed_workflow.io.write_to_shapefile(
            f'watershed_workflow/test/fixture_data/river_{h}.shp', v, crs)


if __name__ == '__main__':
    import os
    if not os.path.isdir('watershed_workflow/test/fixture_data'):
        raise RuntimeError('Run this script from the top level directory')
    if not os.path.isfile('/tmp/my.pkl'):
        raise RuntimeError(
            'Modify tests to write and run the tests -- see source_fixture_helpers.__doc__')

    read_and_process_dump('/tmp/my.pkl')
