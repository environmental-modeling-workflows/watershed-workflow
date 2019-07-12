"""Functions useful within scripts, or helpers for the user interface."""
import os
import logging
import argparse
import fiona

import workflow.conf
import workflow.source_list

verb_to_level = {0:logging.WARNING,
                 1:logging.INFO,
                 2:logging.DEBUG,
                 3:logging.DEBUG}

def setup_logging(verbosity, logfile=None):
    """Sets the log level and log file."""
    level = verb_to_level[verbosity]
    
    if type(logfile) is str:
        raise RuntimeError("Developer error: use 'with open() as fid' construct instead.")

    if logfile is not None:
        logging.basicConfig(filename=logfile, level=level,
                            format='%(asctime)s - %(name)s - %(levelname)s: %(message)s')
    else:
        logging.basicConfig(level=level,
                            format='%(asctime)s - %(name)s - %(levelname)s: %(message)s')

def get_basic_argparse(docstring):
    """Gets a basic argparse class with basic options for all scripts."""
    doclines = docstring.split('\n')
    try:
        first_empty = next(i for i,line in enumerate(doclines) if line.strip() == '')
    except StopIteration:
        description = docstring
        epilog = ''
    else:
        description = '\n'.join(doclines[0:first_empty])
        if len(doclines) > first_empty:
            epilog = '\n'.join(doclines[first_empty+1:])
        else:
            epilog = ''

    parser = argparse.ArgumentParser(description=description, epilog=epilog)
    parser.add_argument('-v', '--verbosity', action='count', default=1,
                        help='Increase output verbosity.  (default=1)')
    parser.add_argument('--logfile', type=str,
                        help='Write logging to file instead of stdout')
    def valid_epsg(x):
        """Note this validator does the work, no need for more"""
        try:
            workflow.conf.set_default_crs(x)
        except ValueError as err:
            raise argparse.ArgumentTypeError("In parsing EPSG: '%s'"%str(err))
        return x
    parser.add_argument('--projection', type=valid_epsg, default=workflow.conf.default_crs(),
                        help='Output coordinate system.  Default is "{}"'.format(workflow.conf.default_crs()['init']))
    return parser


def valid_hucstr(hucstr):
    try:
        huc_valid = workflow.huc_str(hucstr)
    except RuntimeError as err:
        raise argparse.ArgumentTypeError("In parsing HUC string: '%s'"%str(err))
    else:
        return huc_valid

def huc_arg(parser):
    """Adds a HUC argument to the parser."""
    parser.add_argument('HUC', type=valid_hucstr,
                        help='HUC code, for example "060102080101"')

def huc_args(parser):
    """Adds a HUC argument to the parser."""
    parser.add_argument('HUCS', type=valid_hucstr, nargs='+',
                        help='A list of HUC codes, for example, "060102080101"')
    
def simplify_options(parser):
    """Adds a simplify tolerance option to the parser."""
    simp = parser.add_argument_group('Shape Simplification')
    simp.add_argument('--simplify', type=float, default=10.0,
                        help='Tolerance for calls to GIS simplify [m] (default=10m)')
    simp.add_argument('--prune-reach-size', type=int, default=2,
                        help='Keep only rivers with at least this many reaches (default=2).')
    simp.add_argument('--cut-intersections', action='store_true',
                        help='Cut boundaries at river intersections.')

def default_simplify_options():
    """Returns a refine options struct for use in scripts."""
    class Struct:
        def __init__(self, **kwargs):
            self.__dict__.update(**kwargs)

    args = Struct(simplify=10.,
                  prune_reach_size=2,
                  cut_intersections=False,
                  verbosity=1)
    return args
    
    
def triangulate_options(parser):
    """Adds triangulation options to the parser."""
    group = parser.add_argument_group('Triangle Refinement')
    refine_max_area_options(group)
    refine_distance_options(group)
    refine_min_angle(group)
    refine_max_edge_length(group)
    enforce_delaunay(parser)

def default_triangulate_options():
    """Returns a refine options struct for use in scripts."""
    class Struct:
        def __init__(self, **kwargs):
            self.__dict__.update(**kwargs)

    args = Struct(refine_max_area=None,
                  refine_distance=None,
                  refine_min_angle=None,
                  refine_max_edge_length=None,
                  enforce_delaunay=False,
                  verbosity=1)
    return args

def refine_max_area_options(parser):
    parser.add_argument('--refine-max-area', type=float, 
                        help='Refine based upon max area of triangles [m^2]  (see note)')

def refine_distance_options(parser):
    parser.add_argument('--refine-distance', type=float, nargs=4,
                       metavar=('CLOSE_DISTANCE', 'CLOSE_AREA','FAR_DISTANCE', 'FAR_AREA'),
                        help='\n'.join(['Refine based upon a distance from stream.',
                                        ' Distances are in [m] and areas in [m^2].  A',
                                        ' triangle is refined if its area is less CLOSE_AREA',
                                        ' when its distance is less than CLOSE_DISTANCE, or',
                                        ' FAR_AREA if its distance is greater than FAR_DISTANCE,',
                                        ' or a linear interpolant between those two otherwise.']))
def refine_min_angle(parser):
    parser.add_argument('--refine-min-angle', type=float,
                        help='Refine to set a minimum angle constraint in [degrees]')

def refine_max_edge_length(parser):
    parser.add_argument('--refine-max-edge-length', type=float,
                        help='Refine based upon a max edge length [m]')
    
def enforce_delaunay(parser):
    parser.add_argument('--enforce-delaunay', action='store_true',
                        help='Enforce Delaunay, and not just constrained Delaunay')
    
def inshape_args(parser):
    """Sets input filename shapefile options."""
    def shapefile(x):
        """
        'Type' for argparse - checks that file exists but does not open.
        """
        if not os.path.exists(x):
            raise argparse.ArgumentTypeError("Input shapefile '{0}' does not exist".format(x))

        # check now that fiona can open the file.  immediate close to avoid resource issues
        with fiona.open(x,'r') as fid:
            pass
        return x
    
    parser.add_argument('input_file',
                        type=shapefile, help='filename including shape to be meshed')
    parser.add_argument('--shape-index', type=int, default=-1,
                        help='index of desired shape in shapefile, (default=all in file)')

def outmesh_args(parser):
    """Sets output filename and format options."""
    parser.add_argument('output_file', type=str,
                        help='VTK Filename for mesh output.')
    parser.add_argument('-p', '--plot', action='store_true',
                        help='Save mesh image to file.')

def huc_hint_options(parser):
    """Adds a HUC hint option for searching for shapes in HUCs"""
    parser.add_argument('--hint', type=valid_hucstr,
                        help="Hint for searching for a HUC, i.e. '06' for Tennessee River.  Note currently required to avoid downloading all HUCs!",
                        required=True)

def center_options(parser):
    """Adds option to center the output mesh"""
    parser.add_argument('--center', '-c', action='store_true',
                        help="Center the output mesh for cleaner simulation but harder provenance.  Note centroid is written to the output mesh's readme file")


def huc_source_options(parser):
    """Add options for sources."""
    parser.add_argument('--source-huc', type=str, default=workflow.source_list.default_huc_source,
                        choices=set(workflow.source_list.huc_sources.keys()),
                        help='Hydrologic unit shapefile dataset.  (default = "{}")'.format(workflow.source_list.default_huc_source))

def dem_source_options(parser):
    """Add options for sources."""
    parser.add_argument('--source-dem', type=str, default=workflow.source_list.default_dem_source,
                        choices=set(workflow.source_list.dem_sources.keys()),
                        help='Digital Elevation Model dataset.  (default = "{}")'.format(workflow.source_list.default_dem_source))

def hydro_source_options(parser):
    """Add options for sources."""
    parser.add_argument('--source-hydro', type=str, default=workflow.source_list.default_hydrography_source,
                        choices=set(workflow.source_list.hydrography_sources.keys()),
                        help='Hydrography dataset.  (default = "{}"'.format(workflow.source_list.default_hydrography_source))
    
        
                        
