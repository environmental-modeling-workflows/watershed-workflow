import pytest
import numpy as np
import warnings

import watershed_workflow.mesh.mesh

from watershed_workflow.test.shapes import two_boxes


def check_2D_geometry(m2):
    assert (2 == m2.num_cells)
    assert (7 == m2.num_edges)
    assert (6 == m2.num_vertices)
    assert (6 == len(m2.boundary_edges))
    assert (6 == len(m2.boundary_vertices))
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

    m2 = watershed_workflow.mesh.mesh.Mesh2D(coords, conn)
    check_2D_geometry(m2)


def test_from_transect():
    m2 = watershed_workflow.mesh.mesh.Mesh2D.from_Transect(np.array([0, 10, 20]),
                                                      np.array([0, 0, 0]),
                                                      width=10)
    check_2D_geometry(m2)


def test_extrude():
    m2 = watershed_workflow.mesh.mesh.Mesh2D.from_Transect(np.array([0, 10, 20]),
                                                      np.array([0, 0, 0]),
                                                      width=10)
    m3 = watershed_workflow.mesh.mesh.Mesh3D.extruded_Mesh2D(m2, ['constant', ], [5.0, ], [10, ],
                                                        [1001, ])
    assert (20 == m3.num_cells)
    assert (7*10 + 2*11 == m3.num_faces)
    assert (6 * 11 == m3.num_vertices)


def test_reorder():
    m2 = watershed_workflow.mesh.mesh.Mesh2D.from_Transect(np.array([0, 10, 20, 30, 40, 50]),
                                                      np.array([0,  0,  0,  0,  0,  0]),
                                                      width=10)
    assert m2.num_cells == 5
    m2.labeled_sets.append(watershed_workflow.mesh.mesh.LabeledSet('myset', 101, 'CELL', [3,]))
    assert np.allclose(m2.centroids[m2.labeled_sets[0].ent_ids[0], 0:2], [35.,0.], 1.e-10)

    # reorder
    new_order = [0, 4, 3, 1, 2]
    m2new = m2.reorder(new_order)

    # check the new mesh
    assert m2new.num_cells == 5
    assert np.allclose(m2new.centroids[0,0:2], [5.,0.], 1.e-10)
    assert np.allclose(m2new.centroids[1,0:2], [45.,0.], 1.e-10)
    assert np.allclose(m2new.centroids[4,0:2], [25.,0.], 1.e-10)
    assert np.allclose(m2new.centroids[m2new.labeled_sets[0].ent_ids[0], 0:2], [35.,0.], 1.e-10)


    # partition
    m2newpart = m2new.partition(2, True)

    assert m2new.num_cells == 5
    # -- the partitioned mesh is in random order, but partitioned
    diff = np.linalg.norm(m2newpart.centroids[0] - m2newpart.centroids[1])
    assert abs(diff - 10.0) < 1.e-9

    diff2 = np.linalg.norm(m2newpart.centroids[-1] - m2newpart.centroids[-2])
    assert abs(diff - 10.0) < 1.e-9
    
    # check that the partition is good
    assert 'partition' in m2newpart.cell_data
    for i in range(1, len(m2newpart.cell_data['partition'])):
        assert m2newpart.cell_data.loc[i-1, 'partition'] <= m2newpart.cell_data.loc[i,'partition']

    
def test_refine_triangle_labeled_sets():
    # a center triangle (cell 0) surrounded by 3 triangle neighbors,
    # plus one untouched triangle glued onto neighbor cell 1, so
    # something survives refinement unmodified.
    coords = np.array([
        [0, 0, 0],
        [2, 0, 0],
        [1, 2, 0],
        [1, -2, 0],
        [3, 1, 0],
        [-1, 1, 0],
        [1, -4, 0],
    ], dtype=float)
    conn = [
        [0, 1, 2],  # c = 0, the cell to refine
        [0, 3, 1],  # neighbor across edge (0,1)
        [1, 4, 2],  # neighbor across edge (1,2)
        [2, 5, 0],  # neighbor across edge (2,0)
        [3, 6, 1],  # untouched, glued onto neighbor cell 1
    ]
    m2 = watershed_workflow.mesh.mesh.Mesh2D(coords, conn, check_handedness=True)

    # VERTEX set with both endpoints of edge (0,1) -- midpoint should be added
    m2.labeled_sets.append(
        watershed_workflow.mesh.mesh.LabeledSet('myverts', 1, 'VERTEX', [0, 1]))
    # FACE set containing edge (0,1) -- should split into its two new edges
    m2.labeled_sets.append(
        watershed_workflow.mesh.mesh.LabeledSet(
            'myfaces', 2, 'FACE', [watershed_workflow.mesh.mesh.Edge(0, 1)]))
    # CELL set containing only the untouched cell -- should shift index, not split
    m2.labeled_sets.append(
        watershed_workflow.mesh.mesh.LabeledSet('mycells', 3, 'CELL', [4]))

    m2r = watershed_workflow.mesh.mesh.refineTriangle(m2, 0)

    assert m2r.num_cells == 4 - 4 + 10 + 1  # 4 removed, 10 added, 1 untouched survivor
    assert len(m2r.coords) == len(coords) + 3

    ls_by_name = {ls.name: ls for ls in m2r.labeled_sets}

    # the new midpoint vertex on edge (0,1) is the first of the 3 new vertices
    vm = len(coords)
    assert set(ls_by_name['myverts'].ent_ids) == { 0, 1, vm }

    ea = watershed_workflow.mesh.mesh.Edge(0, vm)
    eb = watershed_workflow.mesh.mesh.Edge(1, vm)
    assert set(ls_by_name['myfaces'].ent_ids) == { ea, eb }

    # the untouched cell (originally index 4) survives, now at index 0
    # because all 4 refined cells (indices 0-3) were removed first
    assert ls_by_name['mycells'].ent_ids == [0]
    assert np.allclose(m2r.centroids[ls_by_name['mycells'].ent_ids[0], 0:2],
                        m2.centroids[4, 0:2], 1.e-10)


def test_write():
    m2 = watershed_workflow.mesh.mesh.Mesh2D.from_Transect(np.array([0, 10, 20]),
                                                      np.array([0, 0, 0]),
                                                      width=10)
    m2.cell_data['partition'] = [0,1]
    m3 = watershed_workflow.mesh.mesh.Mesh3D.extruded_Mesh2D(m2, ['constant', ], [5.0, ], [10, ],
                                                        [1001, ])

    import os
    if os.path.isfile('./mesh.exo'):
        os.remove('./mesh.exo')
    try:
        m3.writeExodus('./mesh.exo', element_block_mode="material id")
    except ImportError:
        warnings.warn('ExodusII is not enabled with this python.')
    else:
        assert (os.path.isfile('./mesh.exo'))

        import xarray
        with xarray.open_dataset('./mesh.exo') as fid:
            assert 20 == fid.sizes['num_el_in_blk1']
