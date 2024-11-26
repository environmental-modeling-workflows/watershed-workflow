"""This module includes functions to increase the resolution of the
river network and the HUC boundary, using the original river network
and HUC.

"""

from typing import List, Optional, Tuple, Any, Iterable, Callable, overload

import numpy as np
import math
import shapely
import abc

from watershed_workflow.river_tree import River
from watershed_workflow.split_hucs import SplitHUCs


#
# Strategy pattern to compute a target length while resampling
#
class ComputeTargetLength(abc.ABC):
    @property
    @abc.abstractmethod
    def is_uniform(self) -> bool:
        pass

    @abc.abstractmethod
    def __call__(self, arg : Any) -> float:
        pass

    
class ComputeTargetLengthFixed(ComputeTargetLength):
    is_uniform = True
    def __init__(self, target_length):
        self._target_length = target_length

    def __call__(self, arg):
        return self._target_length

    
class ComputeTargetLengthByProperty(ComputeTargetLength):
    is_uniform = True
    def __call__(self, reach):
        return reach.properties[names.TARGET_SEGMENT_LENGTH]

    
class ComputeTargetLengthByCallable(ComputeTargetLength):
    is_uniform = True
    def __init__(self, func):
        self._func = func

    def __call__(self, reach):
        return self._func(reach)

    
class ComputeTargetLengthByDistanceToShape(ComputeTargetLength):
    is_uniform = False
    def __init__(self, dist_args, shp):
        self._d1, self._d2 = dist_args[0], dist_args[2]
        self._l1, self._l2 = dist_args[1], dist_args[3]
        self._shp = shp

    def _fromDistance(self, dist):
        if dist < self._d1:
            return self._l1
        elif dist > self._d2:
            return self._l2
        else:
            return self._l1 + (dist - self._d1) / (self._d2 - self._d1) * (self._l2 - self._l1)

    def __call__(self, point):
        return self._fromDistance(shapely.distance(self._shp, shapely.geometry.Point(point)))

    

#
# resampleRivers()
#
@overload
def resampleRivers(rivers : List[River]) -> None:
    """Resamples each reach based on the TARGET_SEGMENT_LENGTH property."""
    ...

@overload
def resampleRivers(rivers : List[River], target_length : float) -> None:
    """Resamples each reach based on a given target length."""
    ...

@overload
def resampleRivers(rivers : List[River], target_length : Callable[[River,],float]) -> None:
    """Resamples each reach based on a functor to provide the target length."""
    ...

def resampleRivers(rivers : List[River],
                   target_length : Optional[float | Callable[[River,],float]] = None) -> None:
    for river in rivers:
        _resampleRiverArgs(river, target_length)



#
# resampleRiver()
#
@overload
def resampleRiver(river : River) -> None:
    """Resamples each reach based on the TARGET_SEGMENT_LENGTH property."""
    ...

@overload
def resampleRiver(river : River, target_length : float) -> None:
    """Resamples each reach based on a given target length."""
    ...

@overload
def resampleRiver(river : River, target_length : Callable[[River,],float]) -> None:
    """Resamples each reach based on a functor to provide the target length."""
    ...

# could use functools singledispatch here but it seems unnecessary
def resampleRiver(river : River,
                  target_length : Optional[float | Callable[[River,],float]] = None) -> None:
    _resampleRiverArgs(river, target_length)


# only exists to quiet mypy    
def _resampleRiverArgs(river : River,
                       target_length : Optional[float | Callable[[River,],float]] = None) -> None:
    if target_length is None:
        _resampleRiver(river, ComputeTargetLengthByProperty())
    elif callable(target_length):
        _resampleRiver(river, ComputeTargetLengthByCallable(target_length))
    else:
        try:
            _resampleRiver(river, ComputeTargetLengthFixed(float(target_length)))
        except ValueError:
            raise ValueError('Unrecognized compute strategy argument provided for resampleRiver')


def _resampleRiver(river : River,
                   computeTargetLength : Callable[[River,],float]) -> None:
    """Resamples a river, in place, given a strategy to compute the target segment length."""
    for node in river.preOrder():
        node.linestring = resampleLineStringUniform(node.linestring, computeTargetLength(node))



#
# resampleHUCs()
#
@overload
def resampleHUCs(hucs : SplitHUCs,
                 target_length : float) -> None:
    """Resamples each HUC boundary segment based on a given target length."""
    ...

@overload
def resampleHUCs(hucs : SplitHUCs,
                 target_length : Tuple[float,float,float,float],
                 shp : shapely.geometry.base.BaseGeometry) -> None:
    """Resample each HUC boundary segment based on distance to a given shape.

    distance_args are [D1, L1, D2, L2], which provides for a linear
    function for target length, ranging from target length L1 at D1
    from shp to L2 to D2 from shp.
    """
    ...

    
def resampleHUCs(hucs : SplitHUCs,
                 target_length : float | Tuple[float,float,float,float],
                 shp : Optional[shapely.geometry.base.BaseGeometry] = None) -> None:
    if isinstance(target_length, list):
        target_length = tuple(target_length)

    if shp is None:
        if isinstance(target_length, tuple):
            raise ValueError('Unrecognized compute strategy arguments provided for resampleHUCs')
        else:
            try:
                target_length = float(target_length)
            except ValueError:
                raise ValueError('Unrecognized compute strategy arguments provided for resampleHUCs')
            else:
                _resampleHUCs(hucs, ComputeTargetLengthFixed(target_length))

    else:
        try:
            strat = ComputeTargetLengthByDistanceToShape(target_length, shp)
        except ValueError:
            raise ValueError('Unrecognized compute strategy arguments provided for resampleHUCs')
        else:
            _resampleHUCs(hucs, strat)


def _resampleHUCs(huc : SplitHUCs,
                  computeTargetLength : ComputeTargetLength) -> None:
    if computeTargetLength.is_uniform:
        for i, ls in enumerate(huc.linestrings):
            huc.linestrings[i] = resampleLineStringUniform(ls, computeTargetLength(ls))
    else:
        for i, ls in enumerate(huc.linestrings):
            huc.linestrings[i] = resampleLineStringNonuniform(ls, computeTargetLength)


            
#
# resampleLineString
#        
def _resampleLineStringUniform(linestring : shapely.geometry.LineString,
                              target_length : float) -> shapely.geometry.LineString:
    """Resample a linestring uniformly with no respect for previous discrete coords."""
    count = math.ceil(linestring.length / target_length) + 1
    s = np.linspace(0, 1, count)
    return shapely.line_interpolate_point(linestring, s, normalized=True)
                

def resampleLineStringUniform(linestring : shapely.geometry.LineString,
                              target_length : float) -> shapely.geometry.LineString:
    """Resample a linestring nearly uniformly, keeping previous discrete coords if possible."""
    assert linestring.length > 0
    coords = np.array(linestring.coords)
    dcoords = np.linalg.norm(coords[1:] - coords[:-1], axis=1)
    if np.max(dcoords) > target_length:
        new_coords = []
        i = 0
        while i < len(coords)-1:
            j = i + 1
            while j < len(coords)-1 and dcoords[j-1] < target_length:
                j += 1
            sublinestring = shapely.geometry.LineString(coords[i:j+1])
            extra_coords = _resampleLineStringUniform(sublinestring, target_length)
            if i == 0:
                new_coords.extend(extra_coords)
            else:
                new_coords.extend(extra_coords[1:])
            i = j
    else:
        new_coords = _resampleLineStringUniform(linestring, target_length)
    return shapely.geometry.LineString(new_coords)


def _resampleLineStringNonuniform(linestring : shapely.geometry.LineString,
                                  computeTargetLength : Callable[[Any,],float]) -> shapely.geometry.LineString:
    """Resample a linestring by distance with no respect for previous discrete coords."""
    # starting at the near coordinate, add segments of a given
    # arclength until they cover the linestring.
    coords = [linestring.coords[0]]
    arclens = [0.,]
    length = linestring.length
    s = 0.0
    while s < length:
        d = computeTargetLength(coords[-1])
        s = s + d
        arclens.append(s)
        coords.append(linestring.interpolate(s))

    # the new total arclength (s) is now longer than original length,
    # scale the arclens back to length
    arclens_a = np.array(arclens) * length / s

    # sample the linestring to get the new coordinates and return
    new_linestring_coords = shapely.line_interpolate_point(linestring, arclens_a)
    return new_linestring_coords
    

def resampleLineStringNonuniform(linestring : shapely.geometry.LineString,
                                 computeTargetLength : Callable[[Any,],float]) -> shapely.geometry.LineString:
    """Resample a linestring nonuniformly, keeping previous discrete coords if possible."""
    assert linestring.length > 0

    targets = [computeTargetLength(c) for c in linestring.coords]
    if max(targets) == min(targets):
        return resampleLineStringUniform(linestring, targets[0])

    if targets[0] > targets[-1]:
        # work from closest to furthest, so flip the linestring
        linestring = shapely.geometry.LineString(reversed(linestring.coords))
        reverse = True
    else:
        reverse = False

    # greedily add coords along the linestring
    coords = np.array(linestring.coords)
    dcoords = np.linalg.norm(coords[1:] - coords[:-1], axis=1)
    ml = targets[0]
    if np.max(dcoords) > ml:
        new_coords = []
        i = 0
        while i < len(coords)-1:
            j = i + 1
            while j < len(coords)-1 and dcoords[j-1] < ml:
                j += 1
                ml = computeTargetLength(coords[j])
            sublinestring = shapely.geometry.LineString(coords[i:j+1])

            extra_coords = _resampleLineStringNonuniform(sublinestring, computeTargetLength)
            if i == 0:
                new_coords.extend(extra_coords)
            else:
                new_coords.extend(extra_coords[1:])

            i = j
            if j < len(coords):
                ml = computeTargetLength(coords[j])

    else:
        new_coords = _resampleLineStringNonuniform(linestring, computeTargetLength)

    if reverse:
        new_coords = list(reversed(new_coords))
    return shapely.geometry.LineString(new_coords)

