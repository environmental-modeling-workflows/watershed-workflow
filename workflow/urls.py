"""URLs for collecting data."""


def dem(lat,lon):
    """URL, filename for DEM data

    10m proeduct from USGS NED.  Lat,lon are the upper-left corner, in format, e.g.:

         n37, w86
    """
    if type(lat) is int:
        lat = "n%d"%lat
    if type(lon) is int:
        lon = "w%d"%lon

    filename = "USGS_NED_13_%s%s_IMG.zip"%(lat,lon)
    return "https://prd-tnm.s3.amazonaws.com/StagedProducts/Elevation/13/IMG/%s"%filename, filename

def huc2(unit):
    """URL, filename for HUC02 data, which includes files for all contained HUCs down to HUC12."""
    filename = "WBD_%02d_HU2_Shape.zip"%unit
    return "https://prd-tnm.s3.amazonaws.com/StagedProducts/Hydrography/WBD/HU2/Shape/%s"%filename, filename
