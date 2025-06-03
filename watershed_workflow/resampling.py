"""This module includes functions to increase the resolution of the
river network and the HUC boundary, using the original river network
and HUC.

"""

from typing import List, Optional, Tuple, Any, Iterable, Callable, overload

import logging
import numpy as np
import math
import shapely
import abc
import scipy.optimize

from watershed_workflow.river_tree import River
from watershed_workflow.split_hucs import SplitHUCs
import watershed_workflow.utils


#
# Strategy pattern to compute a target length while resampling
#
class ComputeTargetLength(abc.ABC):
    @property
    @abc.abstractmethod
    def is_uniform(self) -> bool:
        pass

    @abc.abstractmethod
    def min(self) -> float:
        pass

    @abc.abstractmethod
    def max(self) -> float:
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

    def min(self):
        return self._target_length

    def max(self):
        return self._target_length
    

    
class ComputeTargetLengthByProperty(ComputeTargetLength):
    is_uniform = True
    def __call__(self, reach):
        return reach.properties[names.TARGET_SEGMENT_LENGTH]

    def min(self):
        return reach.properties[names.TARGET_SEGMENT_LENGTH]

    def max(self):
        return reach.properties[names.TARGET_SEGMENT_LENGTH]
    
    
class ComputeTargetLengthByCallable(ComputeTargetLength):
    is_uniform = True
    def __init__(self, func, min, max):
        self._func = func
        self._min = min
        self._max = max

    def __call__(self, reach):
        return self._func(reach)

    def min(self):
        return self._min

    def max(self):
        return self._max
    
    
class ComputeTargetLengthByDistanceToShape(ComputeTargetLength):
    is_uniform = False
    def __init__(self, dist_args, shp):
        self._d1, self._d2 = dist_args[0], dist_args[2]
        assert self._d1 <= self._d2
        self._l1, self._l2 = dist_args[1], dist_args[3]
        assert self._l1 <= self._l2
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

    def min(self):
        return self._l1

    def max(self):
        return self._l2
    

#
# resampleRivers()
#
@overload
def resampleRivers(rivers : List[River], *, keep_points : bool = False) -> None:
    """Resamples each reach based on the TARGET_SEGMENT_LENGTH property."""
    ...

@overload
def resampleRivers(rivers : List[River], target_length : float, *, keep_points : bool = False) -> None:
    """Resamples each reach based on a given target length."""
    ...

@overload
def resampleRivers(rivers : List[River], target_length : ComputeTargetLength, *, keep_points : bool = False) -> None:
    """Resamples each reach based on a functor to provide the target length."""
    ...

def resampleRivers(rivers: List[River],
                   target_length: float | ComputeTargetLength | None = None,
                   *, keep_points: bool = False) -> None:
    for river in rivers:
        _resampleRiverArgs(river, target_length, keep_points)

#
# resampleRiver()
#
@overload
def resampleRiver(river : River, *, keep_points : bool = False) -> None:
    """Resamples each reach based on the TARGET_SEGMENT_LENGTH property."""
    ...

@overload
def resampleRiver(river : River, target_length : float, *, keep_points : bool = False) -> None:
    """Resamples each reach based on a given target length."""
    ...

@overload
def resampleRiver(river : River, target_length : ComputeTargetLength, *, keep_points : bool = False) -> None:
    """Resamples each reach based on a functor to provide the target length."""
    ...

# could use functools singledispatch here but it seems unnecessary
def resampleRiver(river : River,
                  target_length : Optional[float | ComputeTargetLength] = None,
                  *, keep_points : bool = False) -> None:
    _resampleRiverArgs(river, target_length, keep_points)


# only exists to quiet mypy    
def _resampleRiverArgs(river : River,
                       target_length : Optional[float | ComputeTargetLength] = None,
                       keep_points : bool = False) -> None:
    if target_length is None:
        _resampleRiver(river, ComputeTargetLengthByProperty(), keep_points)
    elif isinstance(target_length, ComputeTargetLength):
        _resampleRiver(river, target_length, keep_points)
    else:
        try:
            _resampleRiver(river, ComputeTargetLengthFixed(float(target_length)), keep_points)
        except ValueError:
            raise ValueError('Unrecognized compute strategy argument provided for resampleRiver')


def _resampleRiver(river : River,
                   computeTargetLength : ComputeTargetLength,
                   keep_points : bool = False) -> None:
    """Resamples a river, in place, given a strategy to compute the target segment length."""
    for node in river.preOrder():
        ls2 = resampleLineStringUniform(node.linestring, computeTargetLength(node), keep_points)
        logging.debug(f'  - resampling ls: {node.linestring.length}, {min(watershed_workflow.utils.computeSegmentLengths(node.linestring))}, {min(watershed_workflow.utils.computeSegmentLengths(ls2))}')
        node.linestring = ls2


#
# resampleSplitHUCs()
#
@overload
def resampleSplitHUCs(hucs : SplitHUCs,
                      target_length : float,
                      *, keep_points : bool = False) -> None:
    """Resamples each HUC boundary segment based on a given target length."""
    ...

@overload
def resampleSplitHUCs(hucs : SplitHUCs,
                      target_length : Tuple[float,float,float,float],
                      shp : shapely.geometry.base.BaseGeometry,
                      *, keep_points : bool = False) -> None:
    """Resample each HUC boundary segment based on distance to a given shape.

    distance_args are [D1, L1, D2, L2], which provides for a linear
    function for target length, ranging from target length L1 at D1
    from shp to L2 to D2 from shp.
    """
    ...

    
def resampleSplitHUCs(hucs : SplitHUCs,
                      target_length : float | Tuple[float,float,float,float],
                      shp : Optional[shapely.geometry.base.BaseGeometry] = None,
                      *, keep_points : bool = False) -> None:
    if isinstance(target_length, list):
        target_length = tuple(target_length)

    if shp is None:
        if isinstance(target_length, tuple):
            raise ValueError('Unrecognized compute strategy arguments provided for resampleSplitHUCs')
        else:
            try:
                target_length = float(target_length)
            except ValueError:
                raise ValueError('Unrecognized compute strategy arguments provided for resampleSplitHUCs')
            else:
                _resampleSplitHUCs(hucs, ComputeTargetLengthFixed(target_length), keep_points)

    else:
        try:
            strat = ComputeTargetLengthByDistanceToShape(target_length, shp)
        except ValueError:
            raise ValueError('Unrecognized compute strategy arguments provided for resampleSplitHUCs')
        else:
            _resampleSplitHUCs(hucs, strat, keep_points)

    # recreate polygon shapes after changing all the linestrings
    hucs.update()


def _resampleSplitHUCs(huc : SplitHUCs,
                       computeTargetLength : ComputeTargetLength,
                       keep_points : bool = False) -> None:
    if computeTargetLength.is_uniform:
        for i, ls in enumerate(huc.linestrings):
            huc.linestrings[i] = resampleLineStringUniform(ls, computeTargetLength(ls), keep_points)
    else:
        for i, ls in enumerate(huc.linestrings):
            huc.linestrings[i] = resampleLineStringNonuniform(ls, computeTargetLength, keep_points)


            
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
                              target_length : float,
                              keep_points : bool = False) -> shapely.geometry.LineString:
    """Resample a linestring nearly uniformly, keeping previous discrete coords if possible."""
    assert linestring.length > 0
    if keep_points:
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
    else:
        new_coords = _resampleLineStringUniform(linestring, target_length)
    return shapely.geometry.LineString(new_coords)


def _resampleLineStringNonuniform(linestring : shapely.geometry.LineString,
                                  computeTargetLength : ComputeTargetLength) -> shapely.geometry.LineString:
    """Resample a linestring by distance with no respect for previous discrete coords."""
    # starting at the near coordinate, add segments of a given
    # arclength until they cover the linestring.
    coords = [linestring.coords[0]]
    arclens = [0.,]
    length = linestring.length
    s = 0.0

    while s < length - 1.e-4:
        def func1(s_itr):
            if s_itr > linestring.length:
                v = np.array(linestring.coords[-1]) - np.array(linestring.coords[-2])
                v /= np.linalg.norm(v)
                next_coord = np.array(linestring.coords[-1]) + (s_itr - linestring.length) * v
            else:
                next_coord = linestring.interpolate(s_itr).coords[0]
            midp = watershed_workflow.utils.computeMidpoint(next_coord, coords[-1])
            d = computeTargetLength(midp)
            return next_coord, d
        
        def func(s_itr):
            next_coord, d = func1(s_itr)
            ds = watershed_workflow.utils.computeDistance(next_coord, coords[-1])
            return abs(d - ds)

        logging.debug('Iterating to resample nonuniform:')
        res = scipy.optimize.minimize_scalar(func, bounds=[s + computeTargetLength.min(),
                                                           s + computeTargetLength.max()])
        s = res.x
        next_coord, d = func1(s)

        logging.debug(f'  coords = ( {coords[-1]}, {next_coord} ) at s = {s} gives d = {d}')
        if res.success:
            logging.debug(f'  converged: {res.message} with error {func(s)}, itrs = {res.nit}')
        else:
            logging.debug(f'  NOT CONVERGED: {res.message} with error {func(s)}, itrs = {res.nit}')

        arclens.append(s)
        coords.append(next_coord)

    # the new total arclength (s) is now longer than original length,
    # scale the arclens back to length
    logging.debug(f'arclens = {np.array(arclens) / length}')
    logging.debug(f'shrinking by factor of {length/s}')
    arclens_a = np.array(arclens) / arclens[-1] * length
    

    # sample the linestring to get the new coordinates and return
    new_linestring_coords = shapely.line_interpolate_point(linestring, arclens_a)
    return new_linestring_coords
    

def resampleLineStringNonuniform(linestring : shapely.geometry.LineString,
                                 computeTargetLength : ComputeTargetLength,
                                 keep_points : bool = False) -> shapely.geometry.LineString:
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

    if keep_points:
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

    else:
        new_coords = _resampleLineStringNonuniform(linestring, computeTargetLength)

    if reverse:
        new_coords = list(reversed(new_coords))
    return shapely.geometry.LineString(new_coords)

