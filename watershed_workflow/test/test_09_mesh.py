import pytest
import numpy as np
import warnings

import watershed_workflow.mesh

from watershed_workflow.test.shapes import two_boxes


def check_2D_geometry(m2):
    assert (2 == m2.num_cells)
    assert (7 == m2.num_edges)
    assert (6 == m2.num_nodes)
    assert (6 == len(m2.boundary_edges))
    assert (6 == len(m2.boundary_nodes))
    assert (np.allclose(np.array([5, 0, 0]), m2.computeCentroid(0), 1.e-6))
    assert (np.allclose(np.array([15, 0, 0]), m2.computeCentroid(1), 1.e-6))


def test_2D(two_boxes):
    """Create a 2D mesh, extrude, write."""
    two_boxes = list(two_boxes.geometry)
    coords1 = np.array(two_boxes[0].exterior.coords)[:-1]
    coords2 = np.array(two_boxes[1].exterior.coords)[:-1]
    coords_xy = np.concatenate([coords1, coords2[1:3]], axis=0)
    coords_z = np.zeros((len(coords_xy), 1), 'd')

    coords = np.concatenate([coords_xy, coords_z], axis=1)
    print(coords)
    conn = [[0, 1, 2, 3], [1, 4, 5, 2]]

    m2 = watershed_workflow.mesh.Mesh2D(coords, conn)
    check_2D_geometry(m2)


def test_from_transect():
    m2 = watershed_workflow.mesh.Mesh2D.from_Transect(np.array([0, 10, 20]),
                                                      np.array([0, 0, 0]),
                                                      width=10)
    check_2D_geometry(m2)


def test_extrude():
    m2 = watershed_workflow.mesh.Mesh2D.from_Transect(np.array([0, 10, 20]),
                                                      np.array([0, 0, 0]),
                                                      width=10)
    m3 = watershed_workflow.mesh.Mesh3D.extruded_Mesh2D(m2, ['constant', ], [5.0, ], [10, ],
                                                        [1001, ])
    assert (20 == m3.num_cells)
    assert (7*10 + 2*11 == m3.num_faces)
    assert (6 * 11 == m3.num_nodes)


def test_write():
    m2 = watershed_workflow.mesh.Mesh2D.from_Transect(np.array([0, 10, 20]),
                                                      np.array([0, 0, 0]),
                                                      width=10)
    m3 = watershed_workflow.mesh.Mesh3D.extruded_Mesh2D(m2, ['constant', ], [5.0, ], [10, ],
                                                        [1001, ])

    import os
    if os.path.isfile('./mesh.exo'):
        os.remove('./mesh.exo')
    try:
        m3.writeExodus('./mesh.exo')
    except ImportError:
        warnings.warn('ExodusII is not enabled with this python.')
    else:
        assert (os.path.isfile('./mesh.exo'))

        import xarray
        with xarray.open_dataset('./mesh.exo') as fid:
            assert 20 == fid.sizes['num_el_in_blk1']
