"""This is in the github repo for rasterio, but not in my distribution..."""
import collections
import math

def rowcol(transform, xs, ys, op=math.floor, precision=None):
    """
    Returns the rows and cols of the pixels containing (x, y) given a
    coordinate reference system.
    Use an epsilon, magnitude determined by the precision parameter
    and sign determined by the op function:
        positive for floor, negative for ceil.
    Parameters
    ----------
    transform : Affine
        Coefficients mapping pixel coordinates to coordinate reference system.
    xs : list or float
        x values in coordinate reference system
    ys : list or float
        y values in coordinate reference system
    op : function
        Function to convert fractional pixels to whole numbers (floor, ceiling,
        round)
    precision : int, optional
        Decimal places of precision in indexing, as in `round()`.
    Returns
    -------
    rows : list of ints
        list of row indices
    cols : list of ints
        list of column indices
    """

    single_x = False
    single_y = False
    if not isinstance(xs, collections.Iterable):
        xs = [xs]
        single_x = True
    if not isinstance(ys, collections.Iterable):
        ys = [ys]
        single_y = True

    if precision is None:
        eps = 0.0
    else:
        eps = 10.0 ** -precision * (1.0 - 2.0 * op(0.1))

    invtransform = ~transform

    rows = []
    cols = []
    for x, y in zip(xs, ys):
        fcol, frow = invtransform * (x + eps, y - eps)
        cols.append(op(fcol))
        rows.append(op(frow))

    if single_x:
        cols = cols[0]
    if single_y:
        rows = rows[0]

    return rows, cols
