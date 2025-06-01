"""Namespace defining a bunch of standard names for properties of hydrologic units or reaches."""

# generic property names used everywhere
ID = 'ID'
NAME = 'name'

# watershed/polygon property names
HUC = 'huc'
AREA = 'area'
OUTLET = 'outlet'

# reach property names
TARGET_SEGMENT_WIDTH = 'target_width'
TARGET_SEGMENT_LENGTH = 'target_length'
ORDER = 'stream_order'
DRAINAGE_AREA = 'drainage_area_sqkm'
HYDROSEQ = 'hydroseq'
UPSTREAM_HYDROSEQ = 'uphydroseq'
DOWNSTREAM_HYDROSEQ = 'dnhydroseq'
DIVERGENCE = 'divergence'
CATCHMENT = 'catchment'
CATCHMENT_AREA = 'catchment_area'
LENGTH = 'length'

# reach property names used in conditioning
PROFILE_ELEVATION = 'elev_profile'


# internal reach properties, not set by user
ELEMS = 'elems'
ELEMS_GID_START = 'elems_gid_start'
PARENT = 'parent_ID'
CHILDREN = 'children_IDs'
