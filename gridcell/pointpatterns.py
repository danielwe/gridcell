#!/usr/bin/env python

"""File: pointpatterns.py
Module to facilitate point pattern analysis in arbitrarily shaped 2D windows.

"""
# Copyright 2015 Daniel Wennberg
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import numpy
import pandas
from scipy import integrate, optimize, interpolate
from scipy.spatial import distance
from scipy.stats import percentileofscore
from shapely import geometry, affinity, ops, speedups
from matplotlib import pyplot, patches

from .utils import AlmostImmutable, sensibly_divide
from .external.memoize import memoize_method

if speedups.available:
    speedups.enable()


RSAMPLES = 80
QUADLIMIT = 480


class Window(geometry.Polygon):
    """
    Represent a polygon-shaped window in the Euclidean plane, and provide
    methods for computing quantities related to it.

    """

    def __reduce__(self):
        memcache = memoize_method.cache_name
        red = list(geometry.Polygon.__reduce__(self))
        red[2] = {'state': red[2],
                  memcache: getattr(self, memcache, {})}
        return tuple(red)

    def __setstate__(self, state):
        geometry.Polygon.__setstate__(self, state.pop('state'))
        for key in state:
            setattr(self, key, state[key])

    @property
    @memoize_method
    def lattice(self):
        """
        The lattice vectors of a Bravais lattice having the window as Voronoi
        unit cell

        The lattice vectors are stored as an n-by-2 array, with n the number of
        window edges, such that each row contains the coordinates of a lattice
        vector crossing a window edge

        If the window is not a simple plane-filling polygon (parallellogram or
        hexagon with reflection symmetry through its center), a ValueError
        is raised.

        """
        vertices = (numpy.asarray(self.boundary)[:-1] - self.centroid)
        l = vertices.shape[0]
        vrotated = numpy.roll(vertices, l // 2, axis=0)
        if not (l in (4, 6) and numpy.allclose(vertices, -vrotated)):
            raise ValueError("window must be a simple plane-filling polygon "
                             "(a parallellogram, or a hexagon with reflection "
                             "symmetry through its center) to compute lattice "
                             "vectors.")

        return vertices + numpy.roll(vertices, 1, axis=0)

    @property
    @memoize_method
    def inscribed_circle(self):
        """
        The center and radius of the largest circle that can be inscribed in
        the polygon

        """
        def d(p):
            point = geometry.Point(p)
            if self.contains(point):
                return -self.boundary.distance(point)
            else:
                return 0.0

        cent = self.centroid
        x, y = optimize.minimize(d, (cent.x, cent.y)).x
        r = -d((x, y))

        return {'center': (x, y), 'radius': r}

    @property
    @memoize_method
    def longest_diagonal(self):
        """
        The length of the longest diagonal across the polygon

        """
        bpoints = list(geometry.MultiPoint(self.boundary)[:-1])
        dmax = 0.0
        while bpoints:
            p1 = bpoints.pop()
            for p2 in bpoints:
                d = p1.distance(p2)
                if d > dmax:
                    dmax = d

        return dmax

    @memoize_method
    def dilate_by_this(self, other):
        """
        Dilate another polygon by this polygon

        :other: polygon to dilate
        :returns: dilated polygon

        NB! Don't know if this algorithm works in all cases

        """
        plist = []
        sbpoints = geometry.MultiPoint(self.boundary)[:-1]
        obpoints = geometry.MultiPoint(other.boundary)[:-1]
        for p in sbpoints:
            plist.append(affinity.translate(other, xoff=p.x, yoff=p.y))
        for p in obpoints:
            plist.append(affinity.translate(self, xoff=p.x, yoff=p.y))
        return ops.cascaded_union(plist)

    @memoize_method
    def erode_by_this(self, other):
        """
        Erode another polygon by this polygon

        :other: polygon to erode
        :returns: eroded polygon

        NB! Don't know if this algorithm is correct in all cases

        """
        eroded = self.__class__(other)
        sbpoints = geometry.MultiPoint(self.boundary)[:-1]
        for p in sbpoints:
            eroded = eroded.intersection(affinity.translate(other, xoff=-p.x,
                                                            yoff=-p.y))
        return eroded

    def translated_intersection(self, xoff, yoff):
        """
        Compute the intersection of the window with a translated copy of itself

        :xoff: distance to translate in the x direction
        :yoff: distance to translate in the y direction
        :returns: a Window instance corresponding to the intersection

        """
        return self.intersection(affinity.translate(self, xoff=xoff,
                                                    yoff=yoff))

    @memoize_method
    def _set_covariance_interpolator(self):
        """
        Compute a set covariance interpolator for the window

        Returns
        -------
        RectangularGridInterpolator
            Interpolator that computes the the set covariance of the window.

        """
        xoffs = numpy.linspace(-self.longest_diagonal, self.longest_diagonal,
                               3 * RSAMPLES)
        yoffs = numpy.linspace(-self.longest_diagonal, self.longest_diagonal,
                               3 * RSAMPLES)
        scarray = numpy.zeros((xoffs.size, yoffs.size))
        for (i, xoff) in enumerate(xoffs):
            for (j, yoff) in enumerate(yoffs):
                scarray[i, j] = self.translated_intersection(xoff, yoff).area
        return interpolate.RegularGridInterpolator((xoffs, yoffs), scarray,
                                                   bounds_error=False,
                                                   fill_value=0.0)

    def set_covariance(self, x, y):
        """
        Compute the set covariance of the window at given displacements

        This is a wrapper around self._set_covariance_interpolator, providing
        a user friendly call signature.

        Parameters
        ----------
        x, y : array-like
            Arrays of the same shape giving x and y values of the displacements
            at which to evaluate the set covariance.

        Returns
        -------
        ndarray
            Array of the same shape as `x` and `y` containing the set
            covariance at each displacement.

        """
        xi = numpy.concatenate((x[..., numpy.newaxis],
                                y[..., numpy.newaxis]), axis=-1)
        return self._set_covariance_interpolator()(xi)

    @memoize_method
    def _isotropised_set_covariance_interpolator(self):
        """
        Compute an isotropised set covariance interpolator for the window

        Returns
        -------
        interp1d
            Interpolator that computes the the isotropised set covariance of
            the window.

        """
        rvals = numpy.linspace(0.0, self.longest_diagonal, RSAMPLES)
        iso_set_cov = numpy.zeros_like(rvals)

        # Identify potentially problematic angles and a safe starting- and
        # ending angle for the quadrature integration
        xy = numpy.asarray(self.boundary)[:-1]
        problem_angles = numpy.sort(numpy.arctan2(xy[:, 1], xy[:, 0]))
        theta0 = 0.5 * (problem_angles[0] + problem_angles[-1] - 2 * numpy.pi)

        for (i, rval) in enumerate(rvals):
            def integrand(theta):
                return self.set_covariance(rval * numpy.cos(theta),
                                           rval * numpy.sin(theta))

            iso_set_cov[i] = (integrate.quad(integrand, theta0,
                                             2.0 * numpy.pi + theta0,
                                             limit=QUADLIMIT,
                                             points=problem_angles)[0] /
                              (2 * numpy.pi))

        return interpolate.interp1d(rvals, iso_set_cov, bounds_error=False,
                                    fill_value=0.0)

    def isotropised_set_covariance(self, r):
        """
        Compute the isotropised set covariance of the window at given
        displacements

        This is a wrapper around self._isotropised_set_covariance_interpolator,
        providing a user friendly call signature.

        Parameters
        ----------
        r : array-like
            Array giving the displacements at which to evaluate the isotropised
            set covariance.

        Returns
        -------
        ndarray
            Array of the same shape as `r` containing the isotropised set
            covariance at each displacement.

        """
        return self._isotropised_set_covariance_interpolator()(r)

    @memoize_method
    def _pvdenom_interpolator(self):
        """
        Compute an interpolator for the denominator of the p-function for the
        adapted intensity estimator based on area

        Returns
        -------
        interp1d
            Interpolator that computes the the p-function denominator.

        """
        def integrand(t):
            return 2 * numpy.pi * t * self.isotropised_set_covariance(t)

        rvals = numpy.linspace(0.0, self.longest_diagonal, RSAMPLES)
        dvals = numpy.empty_like(rvals)
        for (i, rval) in enumerate(rvals):
            dvals[i] = integrate.quad(integrand, 0.0, rval, limit=QUADLIMIT)[0]

        return interpolate.interp1d(rvals, dvals, bounds_error=True)

    def pvdenom(self, r):
        """
        Compute the denominator of the p-function for the adapted intensity
        estimator based on area

        This is a wrapper around self._pvdenom_interpolator, providing a user
        friendly call signature.

        Parameters
        ----------
        r : array-like
            Array giving the distances at which to evaluate the p-function
            denominator.

        Returns
        -------
        ndarray
            Array of the same shape as `r` containing the p-function
            denominator at each distance.

        """
        return self._pvdenom_interpolator()(r)

    def p_V(self, point, r):
        """
        Compute the p-function for the adapted intensity estimator based on
        area

        :point: a Point instance giving the location at which to evaluate the
                function
        :r: array-like with radii around 'point' at which to ev' 'aluate the
            p-function
        :returns: the value of the area p-function

        """
        try:
            num = numpy.asarray([self.intersection(point.buffer(rval)).area
                                 for rval in r])
        except TypeError:
            num = self.intersection(point.buffer(r)).area

        return sensibly_divide(num, self.pvdenom(r))

    def p_S(self, point, r):
        """
        Compute the p-function for the adapted intensity estimator based on
        perimeter

        :point: a Point instance giving the location at which to evaluate the
                function
        :r: array-like with radii around 'point' at which to evaluate the
            p-function
        :returns: the value of the perimeter p-function

        """
        try:
            num = numpy.asarray(
                [self.intersection(point.buffer(rval).boundary).length
                 for rval in r])
        except TypeError:
            num = self.intersection(point.buffer(r).boundary).length

        denom = 2.0 * numpy.pi * r * self.isotropised_set_covariance(r)

        return sensibly_divide(num, denom)

    def patch(self, **kwargs):
        """
        Return a matplotlib.patches.Polygon instance for this window

        :kwargs: passed through to the matplotlib.patches.Polygon constructor
        :returns: matplotlib.patches.Polygon instance

        """
        return patches.Polygon(self.boundary, **kwargs)

    def plot(self, axes=None, linewidth=2.0, color='0.5', fill=False,
             **kwargs):
        """
        Plot the window

        The window can be added to an existing plot via the optional 'axes'
        argument.

        :axes: Axes instance to add the window to. If None (default), the
               current Axes instance with equal aspect ratio is used if any, or
               a new one created.
        :linewidth: the linewidth to use for the window boundary. Defaults to
                    2.0.
        :color: a valid matplotlib color specification to use for the window.
                Defaults to '0.5', a moderately heavy gray.
        :fill: if True, plot a filled window. If False (default), only plot the
               boundary.
        :kwargs: additional keyword arguments passed on to the
                 patches.Polygon() constructor. Note in particular the keywords
                 'edgecolor', 'facecolor' and 'label'.
        :returns: the plotted matplotlib.patches.Polygon instance

        """
        if axes is None:
            axes = pyplot.gca(aspect='equal')
            cent = self.centroid
            diag = self.longest_diagonal
            axes.set(xlim=(cent.x - diag, cent.x + diag),
                     ylim=(cent.y - diag, cent.y + diag))

        wpatch = self.patch(linewidth=linewidth, color=color, fill=fill,
                            **kwargs)
        wpatch = axes.add_patch(wpatch)

        return wpatch


class PointPattern(AlmostImmutable):
    """
    Represent a planar point pattern and its associated window, and provide
    methods for analyzing its statistical properties


    Parameters
    ----------
    points : sequence or MultiPoint
        A sequence of coordinate tuples or any other valid MultiPoint
        constructor argument, representing the points in the point pattern
    window : sequence or Polygon
        A sequence of coordinate tuples or any othr valid Polygon constructor
        argument, defining the set within which the point pattern is takes
        values.
    pluspoints : sequence or MultiPoint, optional
        Like `points`, but representing a set of extra points (usually outside
        the window) to use for plus sampling.
    edge_correction : str {'stationary', 'finite', 'isotropic', 'periodic',
                           'plus'}, optional
        String to select the default edge handling to apply in computations:

        ``stationary``
            Translational edge correction used. Intensity estimated by the
            adapted intensity estimator based on area.
        ``finite``
            Translational edge correction used. Intensity estimated by the
            standard intensity estimator.
        ``isotropic``
            Rotational edge correction used. Intensity estimated by the adapted
            intensity estimator based on area.
        ``periodic``:
            No edge correction used, but points are assumed to repeat
            periodically according on a lattice defined by the basis vectors in
            `self.window.lattice` (if defined). Intensity estimated by the
            standard intensity estimator.
        ``plus``
            No edge correction, but plus sampling is used instead. Intensity
            estimated by the standard intensity estimator.

    """

    _edge_config = {
        'stationary': {
            'pmode': 'default',
            'imode': 'area',
        },
        'finite': {
            'pmode': 'default',
            'imode': 'standard',  #corrected,
        },
        'isotropic': {
            'pmode': 'default',
            'imode': 'area',
        },
        'periodic': {
            'pmode': 'periodic',
            'imode': 'standard',  #corrected,
        },
        'plus': {
            'pmode': 'plus',
            'imode': 'standard',  #corrected,
        },
    }

    def __init__(self, points, window, pluspoints=None,
                 edge_correction='stationary'):
        self._points = geometry.MultiPoint(points)
        self.pluspoints = geometry.MultiPoint(pluspoints)

        # Avoid copying the window unless needed
        if not isinstance(window, Window):
            window = Window(window)
        self.window = Window(window)

        self._edge_correction = edge_correction

    def __getitem__(self, key):
        return self._points.__getitem__(key)

    def __iter__(self):
        return self._points.__iter__()

    def __len__(self):
        return self._points.__len__()

    def _inherit_binary_operation(self, other, op):
        """
        Define the general pattern for inheriting a binary operation on the
        points as a binary operation on the PointPattern

        Parameters
        ----------
        other : shapely object
            The binary operation is applied to `self` and `other`. If `other`
            is also a `PointPattern` instance, an exception is raised if they
            are not defined in `Window` instances that compare equal. If other
            is not a `PointPattern` instance, the union of self._points with
            the intersection of other and self.window is used as the points in
            the PointPattern, while the union of self.pluspoints with the
            difference of other and self.window is used as the pluspoints.
        op : string or callable
            Either a string naming the attribute of `self._points` that
            implements the binary operation, or a callable implementing the
            binary operation on two shapely objects.

        Returns
        -------
        PointPattern
            The result of the binary operation applied to the `PointPattern`
            instances.

        """
        spoints = self._points
        spluspoints = self.pluspoints
        try:
            bound_op = getattr(spoints, op)
            bound_op_plus = getattr(spluspoints, op)
        except TypeError:
            def bound_op(opoints):
                return op(spoints, opoints)

            def bound_op_plus(opluspoints):
                return op(spluspoints, opluspoints)

        swindow = self.window
        try:
            owindow = other.window
        except AttributeError:
            # Apparently, other is not a PointPattern
            opoints = other.intersection(swindow)
            opluspoints = other.difference(opoints)
            new_points = bound_op(opoints)
            new_pluspoints = bound_op_plus(opluspoints)
        else:
            if not (swindow == owindow):
                raise ValueError("instances of {} must be defined over "
                                 "instances of {} that compare equal for "
                                 "binary operations to be defined"
                                 .format(self.__class__.__name__,
                                         swindow.__class__.__name__))
            new_points = bound_op(other._points)
            new_pluspoints = bound_op_plus(other.pluspoints)

        return self.__class__(new_points, swindow, pluspoints=new_pluspoints)

    def difference(self, other):
        return self._inherit_binary_operation(other, 'difference')

    def intersection(self, other):
        return self._inherit_binary_operation(other, 'intersection')

    def symmetric_difference(self, other):
        return self._inherit_binary_operation(other, 'symmetric_difference')

    def union(self, other):
        return self._inherit_binary_operation(other, 'union')

    @property
    @memoize_method
    def periodic_extension(self):
        """
        Compute the periodic extension of this point pattern assuming the
        periodic boundary conditions implied by self.window

        The first two levels of periodic copies of the pattern is computed,
        provided the Window instance associated with the pattern is a simple
        plane-filling polygon.

        :returns: MultiPoint instance containing the points in the periodic
                  extension. The points from the pattern itself are not
                  included

        """
        lattice = self.window.lattice
        lattice_r1 = numpy.roll(lattice, 1, axis=0)
        dvecs = numpy.vstack((lattice, lattice + lattice_r1))
        periodic_points = ops.cascaded_union(
            [affinity.translate(self.points(), xoff=dvec[0], yoff=dvec[1])
             for dvec in dvecs])
        return periodic_points

    def points(self, mode='default'):
        """
        Return the points in the pattern

        Parameters
        ----------
        mode : str {'default', 'periodic', plus'}, optional
            String to select points:

            ``default``
                The points constituting the pattern are returned.
            ``periodic``
                The union of the pattern and its periodic extension as defined
                by `self.periodic_extension` is returned.
            ``plus``
                The union of the pattern and the associated plus sampling
                points in `self.pluspoints` is returned.

        Returns
        -------
        MultiPoint
            MultiPoint instance containing the requested points.

        """
        if mode == 'default':
            return self._points
        elif mode == 'periodic':
            return self._points.union(self.periodic_extension)
        elif mode == 'plus':
            return self._points.union(self.pluspoints)
        else:
            raise ValueError("unknown mode: {}".format(mode))

    @staticmethod
    def pairwise_vectors(pp1, pp2=None):
        """
        Return a matrix of vectors between points in a point pattern

        :pp1: PointPattern or MultiPoint instance containing the points to find
              vectors between
        :pp2: if not None, vectors are calculated from points in pp1 to points
              in pp2 instead of between points in pp1
        :returns: numpy array of where slice [i, j, :] contains the vector
                  pointing from pp1[i] to pp1[j], or if pp2 is not None, from
                  pp1[i] to pp2[j]

        """
        ap1 = numpy.asarray(pp1)[:, :2]
        if pp2 is not None:
            ap2 = numpy.asarray(pp2)[:, :2]
        else:
            ap2 = ap1
        return ap2 - ap1[:, numpy.newaxis, :]

    @staticmethod
    def pairwise_distances(pp1, pp2=None):
        """
        Return a matrix of distances between points in a point pattern

        :pp1: PointPattern or MultiPoint instance containing the points to find
              distances between
        :pp2: if not None, distances are calculated from points in pp1 to
              points in pp2 instead of between points in pp1
        :returns: numpy array of where slice [i, j, :] contains the distance
                  from pp1[i] to pp1[j], or if pp2 is not None, from pp1[i] to
                  pp2[j]

        """
        #diff = PointPattern.pairwise_vectors(pp1, pp2=pp2)
        #return numpy.sqrt(numpy.sum(diff * diff, axis=-1))
        ap1 = numpy.asarray(pp1)[:, :2]
        if pp2 is not None:
            ap2 = numpy.asarray(pp2)[:, :2]
        else:
            ap2 = ap1
        return distance.cdist(ap1, ap2)

    def nearest(self, point, mode='standard'):
        """
        Return the point in the pattern closest to the location given by
        'point'

        :point: Point instance giving the location to find the nearest point to
        :mode: string to select the points among which to look for the nearest
               point. See the documentation for PointPattern.points() for
               details.
        :returns: Point instance representing the point in the pattern nearest
                  'point'

        """
        return min(self.points(mode=mode).difference(point),
                   key=lambda p: point.distance(p))

    def nearest_list(self, point, mode='standard'):
        """
        Return the list of points in the pattern, sorted by distance to the
        location given by 'point'

        The list does not include 'point' itself, even if it is part of the
        pattern.

        :point: Point instance giving the location to sort the points by
                distance to.
        :mode: string to select the points to sort. See the documentation for
               PointPattern.points() for details.
        :returns: list of Point instances containing the points in the pattern,
                  sorted by distance to 'point'.

        """
        return sorted(self.points(mode=mode).difference(point),
                      key=lambda p: point.distance(p))

    def intensity(self, mode='standard', r=None):
        """
        Compute an intensity estimate, assuming a stationary point pattern

        :mode: flag to select the kind of estimator to compute. Possible
               values:
            'standard': The standard estimator: the number of points in the
                        pattern divided by the area of the window.
            'area': The adapted estimator based on area.
            'perimeter': The adapted estimator based on perimeter.
            'minus': The standard estimator in a window eroded by the radius r.
            'neighbor': The standard estimator subject to nearest neighbor edge
                        correction.
        :r: array-like, containing distances at which to evaluate the intensity
            estimator, for modes where this is relevant. For modes where
            distance is not relevant, `r` may be omitted.
        :returns: scalar or array-like containing the estimated intensities.

        """
        window = self.window

        if mode == 'standard':
            intensity = len(self) / window.area
        elif mode in ('area', 'perimeter'):
            if mode == 'area':
                pfunc = window.p_V
            else:
                pfunc = window.p_S
            intensity = sum(pfunc(p, r) for p in self)
        elif mode == 'minus':
            try:
                r_enum = enumerate(r)
            except TypeError:
                ew = window.buffer(-r)
                intensity = len(self._points.intersection(ew)) / ew.area
            else:
                intensity = numpy.zeros_like(r)
                for (i, rval) in r_enum:
                    ew = window.buffer(-rval)
                    intensity[i] = len(self._points.intersection(ew)) / ew.area
        elif mode == 'neighbor':
            intensity = 0.0
            for p in self:
                nn_dist = p.distance(self._points.difference(p))
                if nn_dist <= p.distance(window.boundary):
                    intensity += 1.0 / window.buffer(-nn_dist).area
        else:
            raise ValueError("unknown mode: {}".format(mode))

        return intensity

    def squared_intensity(self, mode='standard', r=None):
        """
        Compute an estimate of the squared intensity, assuming a stationary
        point pattern

        The estimate is found by squaring an estimate of the intensity, and
        multiplying with (n - 1) / n, where n is the number of points in the
        pattern, to remove statistical bias due to the squaring.

        :mode: flag to select the kind of estimator to compute. The supported
               modes are listed in the documentation for
               PointPattern.intensity().
            #   In addition, the
            #   following mode is supported:
            #'corrected': The square of the 'standard' intensity estimate,
            #             multiplied by (n - 1) / n to give an unbiased
            #             estimate of the squared intensity.
        :r: array-like, containing distances at which to evaluate the squared
            intensity estimator, for modes where this is relevant. For modes
            where distance is not relevant, `r` may be omitted.
        :returns: scalar or array-like containing the estimated squared
                  intensities.

        """
        n = len(self)

        #if mode == 'corrected':
        #    if n == 0:
        #        return 0.0
        #
        #    lambda_ = self.intensity(mode='standard')
        #    return lambda_ * lambda_ * (n - 1) / n
        #else:
        #    lambda_ = self.intensity(mode=mode, r=r)
        #    return lambda_ * lambda_

        if n == 0:
            return 0.0

        lambda_ = self.intensity(mode=mode, r=r)
        return lambda_ * lambda_ * (n - 1) / n

    @staticmethod
    def rmax(window, edge_correction):
        """
        Return the largest relevant interpoint distance when using a particular
        edge correction in a particular window

        Parameters
        ----------
        window : Window
            Window in which the points take values.
        edge_correction : str {'stationary', 'finite', 'isotropic', 'periodic',
                               'plus'}
            String to select the edge handling to apply in computations. See
            the documentation for `PointPattern` for details.

        Returns
        -------
        scalar
            Largest relevant interpoint distance.

        """
        if edge_correction in ('finite', 'plus', 'isotropic'):
            return window.longest_diagonal

        elif edge_correction == 'periodic':
            return 0.5 * window.longest_diagonal

        elif edge_correction == 'stationary':
            return 2.0 * window.inscribed_circle['radius']

        else:
            raise ValueError("unknown edge correction: {}"
                             .format(edge_correction))

    def _rvals(self, edge_correction):
        """
        Construct an array of r values tailored for the empirical K/L-functions

        The returned array contains a pair of tightly spaced values around each
        vertical step in the K/L-functions, and evenly spaced r values with
        moderate resolution elsewhere.

        Parameters
        ----------
        edge_correction : str {'stationary', 'finite', 'isotropic', 'periodic',
                               'plus'}
            String to select the edge handling to apply in computations. See
            the documentation for `PointPattern` for details.

        Returns
        -------
        array
            Array of r values tailored to the empirical K/L-functions

        """
        rmax = self.rmax(self.window, edge_correction)
        rvals = numpy.linspace(0.0, rmax, RSAMPLES)

        # Get step locations
        rsteps, __ = self._estimator_base(edge_correction=edge_correction)
        micrormax = 1.e-6 * rmax

        # Add r values tightly around each step
        rvals = numpy.sort(numpy.hstack((rvals, rsteps - micrormax,
                                         rsteps + micrormax)))
        return rvals

    @staticmethod
    def pair_weights(window, mp1, mp2, edge_correction):
        """
        Compute the weights that pairs of points in a window contribute in the
        estimation of second-order summary characteristics

        Parameters
        ----------
        window : Window
            Window in which the points take values.
        mp1, mp2 : MultiPoint
            MultiPoint instances containing the points to pair up.
        edge_correction : str {'stationary', 'finite', 'isotropic', 'periodic',
                               'plus'}
            String to select the edge handling to apply in computations. See
            the documentation for `PointPattern` for details.

        Returns
        -------
        array
            Array containing the weight of the pair `(mp1[i], mp2[j])` in
            element `[i, j]`.

        """
        if edge_correction in ('finite', 'stationary'):
            diff = PointPattern.pairwise_vectors(mp1, mp2)
            return 1.0 / window.set_covariance(diff[:, :, 0], diff[:, :, 1])

        elif edge_correction == 'periodic':
            m, n = len(mp1), len(mp2)
            w = numpy.zeros((m, n))

            if n < m:
                mp1, mp2 = mp2, mp1
                wview = w.transpose()
            else:
                wview = w

            mp2_arr = numpy.array(mp2)

            area_inv = 1.0 / window.area
            #origin = geometry.Point((0.0, 0.0))
            #distances = PointPattern.pairwise_distances(mp1, mp2)
            #pi2 = 2.0 * numpy.pi
            for (i, p1) in enumerate(mp1):
                translated_window = affinity.translate(window, xoff=p1.x,
                                                       yoff=p1.y)
                valid_mp2 = mp2.intersection(translated_window)
                vmp2_arr = numpy.asarray(valid_mp2)
                vindex = numpy.any(
                    PointPattern.pairwise_distances(mp2_arr, vmp2_arr) == 0.0,
                    axis=-1)

                wview[i, vindex] = area_inv

                ## Isotropic edge correction (to cancel corner effects that are
                ## still present for large r)
                #for j in numpy.nonzero(vindex)[0]:
                #    r = distances[i, j]
                #    ring = origin.buffer(r).boundary
                #    wview[i, j] *= (pi2 * r /
                #                    ring.intersection(window).length)

            return w

        elif edge_correction == 'plus':
            m, n = len(mp1), len(mp2)
            w = numpy.empty((m, n))
            w.fill(1.0 / window.area)
            return w

        elif edge_correction == 'isotropic':
            m, n = len(mp1), len(mp2)
            w = numpy.zeros((m, n))

            distances = PointPattern.pairwise_distances(mp1, mp2)

            origin = geometry.Point((0.0, 0.0))
            pi2 = 2.0 * numpy.pi
            for (i, p1) in enumerate(mp1):
                for j in range(n):
                    r = distances[i, j]
                    ring = p1.buffer(r).boundary
                    rball = origin.buffer(r)
                    doughnut = window.difference(window.erode_by_this(rball))
                    w[i, j] = pi2 * r / (
                        window.intersection(ring).length * doughnut.area)
            return w

        else:
            raise ValueError("unknown edge correction: {}"
                             .format(edge_correction))

    @memoize_method
    def _estimator_base(self, edge_correction):
        """
        Compute the distances between pairs of points in the pattern, and the
        weights they contribute in the estimation of second-order
        characteristics

        Parameters
        ----------
        edge_correction : str {'stationary', 'finite', 'isotropic', 'periodic',
                               'plus'}
            String to select the edge handling to apply in computations. See
            the documentation for `PointPattern` for details.

        Returns
        -------
        r : array
            Array of containing the pairwise distances in the point pattern,
            sorted from small to large. Only pairs that actually contribute
            with the selected edge correction are included.
        weights : array
            Array containing the weights associated with pairs in the point
            pattern, sorted such that weights[i] gives the weight of the pair
            with distance r[i].

        """
        window = self.window
        rmax = self.rmax(window, edge_correction)
        pmode = self._edge_config[edge_correction]['pmode']

        allpoints = self.points(mode=pmode)
        distances = self.pairwise_distances(self._points, allpoints)
        valid = numpy.logical_and(distances < rmax, distances != 0.0)

        index1, = numpy.nonzero(numpy.any(valid, axis=1))
        index2, = numpy.nonzero(numpy.any(valid, axis=0))
        mp1 = geometry.MultiPoint([self[i] for i in index1])
        mp2 = geometry.MultiPoint([allpoints[i] for i in index2])
        weight_matrix = self.pair_weights(window, mp1, mp2, edge_correction)

        r = distances[valid]
        sort_ind = numpy.argsort(r)
        r = r[sort_ind]

        weights = weight_matrix[valid[index1, :][:, index2]]
        weights = weights[sort_ind]

        return r, weights

    def kfunction(self, r, edge_correction=None):
        """
        Evaluate the empirical K-function of the point pattern

        Parameters
        ----------
        r : array-like
            array of values at which to evaluate the emprical K-function.
        edge_correction : str {'stationary', 'finite', 'isotropic', 'periodic',
                               'plus'}, optional
            String to select the edge handling to apply in computations. See
            the documentation for `PointPattern` for details.
            If not supplied, the edge correction falls back to the default
            value (set at instance initialization).

        Returns
        -------
        array
            Values of the empirical K-function evaulated at `r`.

        """
        if edge_correction is None:
            edge_correction = self._edge_correction

        rmax = self.rmax(self.window, edge_correction)
        rsteps, weights = self._estimator_base(edge_correction=edge_correction)
        rsteps = numpy.hstack((0.0, rsteps, rmax))
        weights = numpy.hstack((0.0, weights, numpy.nan))
        numsteps = numpy.cumsum(weights)

        indices = numpy.searchsorted(rsteps, r, side='right') - 1

        imode = self._edge_config[edge_correction]['imode']
        lambda2 = self.squared_intensity(mode=imode, r=r)
        return sensibly_divide(numsteps[indices], lambda2)

    def lfunction(self, r, edge_correction=None):
        """
        Evaluate the empirical L-function of the point pattern

        Parameters
        ----------
        r : array-like
            array of values at which to evaluate the emprical K-function.
        edge_correction : str {'stationary', 'finite', 'isotropic', 'periodic',
                               'plus'}, optional
            String to select the edge handling to apply in computations. See
            the documentation for `PointPattern` for details.
            If not supplied, the edge correction falls back to the default
            value (set at instance initialization).

        Returns
        -------
        array
            Values of the empirical L-function evaulated at `r`.

        """
        if edge_correction is None:
            edge_correction = self._edge_correction

        return numpy.sqrt(self.kfunction(r, edge_correction=edge_correction) /
                          numpy.pi)

    @memoize_method
    def lstatistic(self, rmin=None, rmax=None, edge_correction=None):
        """
        Compute the L test statistic for CSR

        The test statstic is defined as max(abs(L(r) - r)) for r-values between
        zero and some maximum radius. Here, the radius of the largest inscribed
        circle in self.window is used as maximum r-value.  Note that if
        edge_correction == 'finite', the power of the L test may depend heavily
        on the maximum r-value and the number of points in the pattern, and the
        statistic computed by this function may not be adequate.

        Parameters
        ----------
        rmin : scalar
            The minimum r value to use when computing the statistic. If None,
            the value :math:`0.2 / \sqrt(\lambda)`, is used, where
            :math:`\lambda` is the standard intensity estimate for the process.
        rmax : scalar
            The maximum r value to use when computing the statistic. If None,
            the radius of the largest inscribed circle in the window of the
            point pattern is used.
        edge_correction : str {'stationary', 'finite', 'isotropic', 'periodic',
                               'plus'}, optional
            String to select the edge handling to apply in computations. See
            the documentation for `PointPattern` for details.
            If not supplied, the edge correction falls back to the default
            value (set at instance initialization).

        Returns
        -------
        scalar
            the L test statistic

        """
        if edge_correction is None:
            edge_correction = self._edge_correction

        if rmin is None:
            rmin = 0.2 / numpy.sqrt(self.intensity())

        r_absolute_max = self.rmax(self.window, edge_correction)
        if rmax is None:
            rmax = self.window.inscribed_circle['radius']
        rmax = min(rmax, r_absolute_max)

        # The largest deviation between L(r) and r is bound to be at a vertical
        # step. We go manual instead of using self.lfunction, in order to get
        # it as exactly and cheaply as possible.
        rsteps, weights = self._estimator_base(edge_correction=edge_correction)
        rsteps = numpy.hstack((0.0, rsteps, r_absolute_max))
        valid = numpy.logical_and(rsteps > rmin, rsteps < rmax)
        rsteps = rsteps[valid]
        weights = numpy.hstack((0.0, weights, numpy.nan))
        numsteps = numpy.cumsum(weights)
        numsteps_high = numsteps[valid]
        numsteps_low = numsteps[numpy.roll(valid, -1)]

        # Compute the L-values just before and after each step
        imode = self._edge_config[edge_correction]['imode']
        lambda2 = self.squared_intensity(mode=imode, r=rsteps)
        lvals_high = numpy.sqrt(sensibly_divide(numsteps_high,
                                                numpy.pi * lambda2))
        lvals_low = numpy.sqrt(sensibly_divide(numsteps_low,
                                               numpy.pi * lambda2))

        # Compute the offset and return the maximum
        offset = numpy.hstack((lvals_high - rsteps, lvals_low - rsteps))
        return numpy.nanmax(numpy.abs(offset))

    def pair_corr_function(self, r, bandwidth=None, edge_correction=None):
        """
        Evaluate the empirical pair correlation function of the point pattern

        Parameters
        ----------
        r : array-like
            array of values at which to evaluate the emprical pair correlation
            function.
        bandwidth : scalar
            The bandwidth of the box kernel used to estimate the density of
            points pairs at a given distance. If None, the bandwidth is
            automatically chose as :math:`0.2 / \sqrt(\lambda)`, where
            :math:`\lambda` is the standard intensity estimate for the process.
        edge_correction : str {'stationary', 'finite', 'isotropic', 'periodic',
                               'plus'}, optional
            String to select the edge handling to apply in computations. See
            the documentation for `PointPattern` for details.
            If not supplied, the edge correction falls back to the default
            value (set at instance initialization).

        Returns
        -------
        array
            Values of the empirical pair correlation function evaulated at `r`.

        """
        if edge_correction is None:
            edge_correction = self._edge_correction

        if bandwidth is None:
            bandwidth = 0.2 / numpy.sqrt(self.intensity())

        rpairs, weights = self._estimator_base(edge_correction=edge_correction)

        # Find the contribution from each pair to each element in `r`
        d = numpy.abs(r[numpy.newaxis, ...] - rpairs[..., numpy.newaxis])
        num = numpy.sum((d < bandwidth) * weights[..., numpy.newaxis], axis=0)
        num *= 1.0 / (4.0 * numpy.pi * r * bandwidth)

        imode = self._edge_config[edge_correction]['imode']
        lambda2 = self.squared_intensity(mode=imode, r=r)
        return sensibly_divide(num, lambda2)

    def plot_kfunction(self, axes=None, edge_correction=None, linewidth=2.0,
                       csr=False, csr_kw=None, **kwargs):
        """
        Plot the empirical K-function for the pattern

        Parameters
        ----------
        axes : Axes, optional
            Axes instance to add the K-function to. If None (default), the
            current Axes instance is used if any, or a new one created.
        edge_correction : str {'stationary', 'finite', 'isotropic', 'periodic',
                               'plus'}, optional
            String to select the edge handling to apply in computations. See
            the documentation for `PointPattern` for details.
            If not supplied, the edge correction falls back to the default
            value (set at instance initialization).
        linewidth : scalar, optional
            The width of the line showing the K-function.
        csr : bool, optional
            If True, overlay the curve :math:`K(r) = \pi r^2`, which is the
            theoretical K-function for complete spatial randomness. The style
            of this line may be customized using csr_kw.
        csr_kw : dict, optional
            Keyword arguments to pass to `axes.plot` when plotting the CSR
            curve.
        **kwargs : dict, optional
            Additional keyword arguments to pass to `axes.plot`. Note in
            particular the keywords 'linestyle', 'color' and 'label'.

        Returns
        -------
        list
            List of handles to the Line2D instances added to the plot, in the
            following order: empirical K-function, CSR curve (optional).

        """
        if axes is None:
            axes = pyplot.gca()

        if edge_correction is None:
            edge_correction = self._edge_correction

        rvals = self._rvals(edge_correction)
        kvals = self.kfunction(rvals, edge_correction=edge_correction)

        lines = axes.plot(rvals, kvals, linewidth=linewidth, **kwargs)

        if csr:
            if csr_kw is None:
                csr_kw = {}

            kcsr = numpy.pi * rvals * rvals
            lines += axes.plot(rvals, kcsr, linestyle='dashed', **csr_kw)

        return lines

    def plot_lfunction(self, axes=None, edge_correction=None, linewidth=2.0,
                       csr=False, csr_kw=None, **kwargs):
        """
        Plot the empirical L-function for the pattern

        Parameters
        ----------
        axes : Axes, optional
            Axes instance to add the L-function to. If None (default), the
            current Axes instance is used if any, or a new one created.
        edge_correction : str {'stationary', 'finite', 'isotropic', 'periodic',
                               'plus'}, optional
            String to select the edge handling to apply in computations. See
            the documentation for `PointPattern` for details.
            If not supplied, the edge correction falls back to the default
            value (set at instance initialization).
        linewidth : scalar, optional
            The width of the line showing the L-function.
        csr : bool, optional
            If True, overlay the curve :math:`L(r) = r`, which is the
            theoretical L-function for complete spatial randomness. The style
            of this line may be customized using csr_kw.
        csr_kw : dict, optional
            Keyword arguments to pass to `axes.plot` when plotting the CSR
            curve.
        **kwargs : dict, optional
            Additional keyword arguments to pass to `axes.plot`. Note in
            particular the keywords 'linestyle', 'color' and 'label'.

        Returns
        -------
        list
            List of handles to the Line2D instances added to the plot, in the
            following order: empirical L-function, CSR curve (optional).

        """
        if axes is None:
            axes = pyplot.gca()

        if edge_correction is None:
            edge_correction = self._edge_correction

        rvals = self._rvals(edge_correction)
        lvals = self.lfunction(rvals, edge_correction=edge_correction)

        lines = axes.plot(rvals, lvals, linewidth=linewidth, **kwargs)

        if csr:
            if csr_kw is None:
                csr_kw = {}

            lines += axes.plot(rvals, rvals, linestyle='dashed', **csr_kw)

        return lines

    def plot_pair_corr_function(self, axes=None, bandwidth=None,
                                edge_correction=None, linewidth=2.0, csr=False,
                                csr_kw=None, **kwargs):
        """
        Plot the empirical pair correlation function for the pattern

        Parameters
        ----------
        axes : Axes, optional
            Axes instance to add the K-function to. If None (default), the
            current Axes instance is used if any, or a new one created.
        bandwidth : scalar
            The bandwidth of the box kernel used to estimate the density of
            points pairs at a given distance. See the documentation for
            `PointPattern.pair_corr_function` for details.
        edge_correction : str {'stationary', 'finite', 'isotropic', 'periodic',
                               'plus'}, optional
            String to select the edge handling to apply in computations. See
            the documentation for `PointPattern` for details.
            If not supplied, the edge correction falls back to the default
            value (set at instance initialization).
        linewidth : scalar, optional
            The width of the line showing the K-function.
        csr : bool, optional
            If True, overlay the curve :math:`g(r) = 1`, which is the
            theoretical pair correlation function for complete spatial
            randomness. The style of this line may be customized using csr_kw.
        csr_kw : dict, optional
            Keyword arguments to pass to `axes.plot` when plotting the CSR
            curve.
        **kwargs : dict, optional
            Additional keyword arguments to pass to `axes.plot`. Note in
            particular the keywords 'linestyle', 'color' and 'label'.

        Returns
        -------
        list
            List of handles to the Line2D instances added to the plot, in the
            following order: empirical pair correlation function, CSR curve
            (optional).

        """
        if axes is None:
            axes = pyplot.gca()

        if edge_correction is None:
            edge_correction = self._edge_correction

        rmax = self.rmax(self.window, edge_correction)
        rvals = numpy.linspace(0.0, rmax, RSAMPLES)
        gvals = self.pair_corr_function(rvals, bandwidth=bandwidth,
                                        edge_correction=edge_correction)

        lines = axes.plot(rvals, gvals, linewidth=linewidth, **kwargs)

        if csr:
            if csr_kw is None:
                csr_kw = {}

            gcsr = numpy.ones_like(rvals)
            lines += axes.plot(rvals, gcsr, linestyle='dashed', **csr_kw)

        return lines

    def plot_pattern(self, axes=None, marker='o', periodic=False, plus=False,
                     window=False, periodic_kw=None, plus_kw=None,
                     window_kw=None, **kwargs):
        """
        Plot point pattern

        The point pattern can be added to an existing plot via the optional
        'axes' argument.

        :axes: Axes instance to add the point pattern to. If None (default),
               the current Axes instance with equal aspect ratio is used if
               any, or a new one created.
        :marker: a valid matplotlib marker specification. Defaults to 'o'
        :periodic: if True, add the periodic extension of the pattern to the
                   plot.
        :plus: if True, add plus sampling points to the plot.
        :window: if True, the window boundaries are added to the plot.
        :periodic_kw: dict of keyword arguments to pass to the axes.plot()
                      method used to plot the periodic extension. Default: None
                      (empty dict)
        :plus_kw: dict of keyword arguments to pass to the axes.plot()
                  method used to plot the plus sampling points. Default: None
                  (empty dict)
        :window_kw: dict of keyword arguments to pass to the Window.plot()
                    method. Default: None (empty dict)
        :kwargs: additional keyword arguments passed on to axes.plot() method
                 used to plot the point pattern. Note especially the keywords
                 'color', 's' (marker size) and 'label'.
        :returns: list of the plotted objects: a Line2D instance with the point
                  pattern, and optionally another Line2D instance for the plus
                  sampling points, and a matplotlib.patches.Polygon instance
                  for the window.

        """
        if axes is None:
            axes = pyplot.gca(aspect='equal')
            cent = self.window.centroid
            diag = self.window.longest_diagonal
            axes.set(xlim=(cent.x - diag, cent.x + diag),
                     ylim=(cent.y - diag, cent.y + diag))

        pp = numpy.asarray(self._points)
        h = axes.plot(pp[:, 0], pp[:, 1], linestyle='None', marker=marker,
                      **kwargs)

        if periodic:
            if periodic_kw is None:
                periodic_kw = {}
            pp = numpy.asarray(self.periodic_extension)
            h += axes.plot(pp[:, 0], pp[:, 1], linestyle='None', marker=marker,
                           **periodic_kw)

        if plus:
            if plus_kw is None:
                plus_kw = {}
            pp = numpy.asarray(self.pluspoints)
            h += axes.plot(pp[:, 0], pp[:, 1], linestyle='None', marker=marker,
                           **plus_kw)

        if window:
            if window_kw is None:
                window_kw = {}
            wpatch = self.window.plot(axes=axes, **window_kw)
            h.append(wpatch)

        return h


class PointPatternCollection(AlmostImmutable):
    """
    Represent a collection of planar point patterns defined in the same window,
    and provide methods to compute statistics over them.

    Parameters
    ----------
    patterns : sequence
        List of PointPattern instnaces to include in the collection.
    edge_correction : str {'stationary', 'finite', 'isotropic', 'periodic',
                           'plus'}, optional
        String to select the default edge handling to apply in computations.
        See the documentation for `PointPattern` for details.

    """

    def __init__(self, patterns, edge_correction='stationary'):
        self.patterns = list(patterns)
        self._edge_correction = edge_correction

    @classmethod
    def from_simulation(cls, nsims, window, intensity, process='binomial',
                        edge_correction='stationary'):
        """
        Create a PointPatternCollection instance by simulating a number of
        point patterns in the same window

        Parameters
        ----------
        nsims : integer
            The number of point patterns to generate.
        window : Window
            Window instance to simulate the process within.
        intensity : scalar
            The intensity (density of points) of the process.
        process : str {'binomial', 'poisson'}, optional
            String to select the kind of process to simulate.
        edge_correction : str {'stationary', 'finite', 'isotropic', 'periodic',
                               'plus'}, optional
            String to select the default edge handling to apply in
            computations. See the documentation for `PointPattern` for details.

        Returns
        -------
        PointPatternCollection
            Collection of the simulated processes

        """
        xmin, ymin, xmax, ymax = window.bounds
        nmean = intensity * window.area
        process = process.lower()

        if process == 'poisson':
            nlist = numpy.random.poisson(nmean, nsims)
        elif process == 'binomial':
            nlist = numpy.empty((nsims,))
            nlist.fill(nmean)
        else:
            raise ValueError("unknown point process: {}".format(process))

        patterns = []
        for n in nlist:
            # We know we need to draw at least n points, so do this right away
            draw = numpy.column_stack(
                (numpy.random.uniform(low=xmin, high=xmax, size=(n,)),
                 numpy.random.uniform(low=ymin, high=ymax, size=(n,))))
            points = geometry.MultiPoint(draw).intersection(window)

            # Iterate until we have enough
            while len(points) < n:
                newpoint = geometry.Point(
                    (numpy.random.uniform(low=xmin, high=xmax),
                     numpy.random.uniform(low=ymin, high=ymax)))
                if window.contains(newpoint):
                    points = points.union(newpoint)

            pp = PointPattern(points, window, edge_correction=edge_correction)
            patterns.append(pp)

        return cls(patterns, edge_correction=edge_correction)

    def __getitem__(self, key):
        return self.patterns.__getitem__(key)

    def __iter__(self):
        return self.patterns.__iter__()

    def __len__(self):
        return self.patterns.__len__()

    ## Fun to consider:
    #def __getattr__(self, name):
    #    try:
    #        return AlmostImmutable.__getattr__(self, name)
    #    except AttributeError as e:
    #        if name[-1] == 's':
    #            try:
    #                ppattr = getattr(PointPattern, name[:-1])
    #            except AttributeError:
    #                pass
    #            else:
    #
    #                def aggregate(edge_correction=None):
    #                    if edge_correction is None:
    #                        edge_correction = self._edge_correction
    #                    return pandas.Series(
    #                        [ppattr(pp, edge_correction=edge_correction)
    #                         for pp in self.patterns])
    #                return aggregate
    #        raise e

    @property
    @memoize_method
    def npoints(self):
        """
        The total number of points in the whole collection

        """
        return sum(len(pp) for pp in self.patterns)

    @property
    @memoize_method
    def total_area(self):
        """
        The total area of the windows for all the patterns in collection

        """
        return sum(pp.window.area for pp in self.patterns)

    @property
    @memoize_method
    def nweights(self):
        """
        List of the fractions of the total number of points in the collection
        coming from each of the patterns

        """
        npoints = self.npoints
        return [len(pp) / npoints for pp in self.patterns]

    @property
    @memoize_method
    def aweights(self):
        """
        List of the fraction of the total window area in the collection coming
        from to the window of each of the patterns

        """
        total_area = self.total_area
        return [pp.window.area / total_area for pp in self.patterns]

    @memoize_method
    def rmax(self, edge_correction):
        """
        The maximum r-value where the K-functions of all patterns in the
        collection are defined

        """
        return min(pp.rmax(pp.window, edge_correction) for pp in self.patterns)

    def aggregate_intensity(self, mode='standard', r=None):
        """
        Compute the aggregate of the intensity estimates of all patterns in the
        collection

        :mode: flag to select the kind of estimator to compute. For details,
               see PointPattern.intensity().
        :r: array-like, containing distances at which to evaluate the aggregate
            intensity estimator, for modes where this is relevant. For modes
            where distance is not relevant, `r` may be omitted.
        :returns: scalar or array-like containing the estimated aggregate
                  intensities.

        """
        implemented_modes = ('standard',)
        if mode not in implemented_modes:
            raise NotImplementedError("aggregate intensity only implemented "
                                      "for the following modes: {}"
                                      .format(implemented_modes))

        aweights = self.aweights
        intensities = [pp.intensity(mode=mode, r=r) for pp in self.patterns]

        return sum(aw * intensity
                   for (aw, intensity) in zip(aweights, intensities))

    def aggregate_squared_intensity(self, mode='standard', r=None):
        """
        Compute the aggregate of the squared intensity estimates of all
        patterns in the collection

        The estimate is found by squaring an estimate of the aggregate
        intensity, and multiplying with (n - 1) / n, where n is the number of
        points in the pattern, to remove statistical bias due to the squaring.

        :mode: flag to select the kind of estimator to compute. If any of the
               values listed in the documentation for PointPattern.intensity is
               given, the square of this estimate is returned.
            #   In addition, the
            #   following mode is supported:
            #'corrected': The square of the 'standard' aggregate intensity
            #             estimate, multiplied by (n - 1) / n to give an
            #             unbiased estimate of the squared aggregate intensity.
        :r: array-like, containing distances at which to evaluate the
            aggregate squared intensity estimator, for modes where this is
            relevant. For modes where distance is not relevant, `r` may be
            omitted.
        :returns: scalar or array-like containing the estimated aggregate
                  squared intensities.

        """
        n = self.npoints

        #if mode == 'corrected':
        #    if n == 0:
        #        return 0.0
        #
        #    lambda_ = self.aggregate_intensity(mode='standard')
        #    return lambda_ * lambda_ * (n - 1) / n
        #else:
        #    lambda_ = self.aggregate_intensity(mode=mode, r=r)
        #    return lambda_ * lambda_

        if n == 0:
            return 0.0

        lambda_ = self.aggregate_intensity(mode=mode, r=r)
        return lambda_ * lambda_ * (n - 1) / n

    def aggregate_kfunction(self, r, edge_correction=None):
        """
        Compute the aggregate of the empirical K-function over all patterns in
        the collection

        Parameters
        ----------
        r : array-like
            Array of values at which to evaluate the emprical aggregate
            K-function.
        edge_correction : str {'stationary', 'finite', 'isotropic', 'periodic',
                               'plus'}, optional
            String to select the edge handling to apply in computations. See
            the documentation for `PointPattern` for details.
            If not supplied, the edge correction falls back to the default
            value (set at instance initialization).

        Returns
        -------
        array
            Values of the empirical aggregate K-function evaluated at `r`.

        """
        if edge_correction is None:
            edge_correction = self._edge_correction

        nweights = self.nweights
        kvalues = [pp.kfunction(r, edge_correction=edge_correction)
                   for pp in self.patterns]

        return sum(nw * kv for (nw, kv) in zip(nweights, kvalues))

    def aggregate_lfunction(self, r, edge_correction=None):
        """
        Compute the aggregate of the empirical L-function over all patterns in
        the collection

        Parameters
        ----------
        r : array-like
            Array of values at which to evaluate the emprical aggregate
            L-function.
        edge_correction : str {'stationary', 'finite', 'isotropic', 'periodic',
                               'plus'}, optional
            String to select the edge handling to apply in computations. See
            the documentation for `PointPattern` for details.
            If not supplied, the edge correction falls back to the default
            value (set at instance initialization).

        Returns
        -------
        array
            Values of the empirical aggregate L-function evaluated at `r`.

        """
        if edge_correction is None:
            edge_correction = self._edge_correction

        return numpy.sqrt(self.aggregate_kfunction(
            r, edge_correction=edge_correction) / numpy.pi)

    def _pp_attr_r_frame(self, attr, r, edge_correction=None, **kwargs):
        """
        Compute a DataFrame containing values of some PointPattern attribute
        which is a function of a distance.

        Parameters
        ----------
        attr : string
            Name of `PointPattern` attribute to use.
        r : array-like
            Array of values at which to evaluate the `PointPattern` attribute.
        edge_correction : str {'stationary', 'finite', 'isotropic', 'periodic',
                               'plus'}, optional
            String to select the edge handling to apply in computations. See
            the documentation for `PointPattern` for details.
            If not supplied, the edge correction falls back to the default
            value (set at instance initialization).
        **kwargs : dict, optional
            Other arguments to pass to the `PointPattern` attribute.

        Returns
        -------
        DataFrame
            DataFrame where each row contains values of the
            `PointPattern` attribute from one pattern, evaluated at `r`.

        """
        if edge_correction is None:
            edge_correction = self._edge_correction

        return pandas.DataFrame(
            [getattr(pp, attr)(r, edge_correction=edge_correction, **kwargs)
             for pp in self.patterns])

    def _pp_attr_r_critical(self, attr, alpha, r, edge_correction=None,
                            **kwargs):
        """
        Compute critical values of some PointPattern attribute

        Parameters
        ----------
        attr : string
            name of `pointpattern` attribute to use.
        alpha : scalar between 0.0 and 1.0
            Percentile defining the critical values.
        r : array-like
            Array of values at which to evaluate the critical values.
        edge_correction : str {'stationary', 'finite', 'isotropic', 'periodic',
                               'plus'}, optional
            String to select the edge handling to apply in computations. See
            the documentation for `PointPattern` for details.
            If not supplied, the edge correction falls back to the default
            value (set at instance initialization).
        **kwargs : dict, optional
            Other arguments to pass to the `PointPattern` attribute.

        Returns
        -------
        array
            Critical values of the `PointPattern` attribute evaluated at `r`.

        """
        if edge_correction is None:
            edge_correction = self._edge_correction

        attr_frame = self._pp_attr_r_frame(attr, r,
                                           edge_correction=edge_correction,
                                           **kwargs)
        return attr_frame.quantile(q=alpha, axis=0)

    def _pp_attr_r_mean(self, attr, r, edge_correction=None, **kwargs):
        """
        Compute the mean of some PointPattern attribute

        Parameters
        ----------
        attr : string
            name of `pointpattern` attribute to use.
        r : array-like
            Array of values at which to evaluate the mean values.
        edge_correction : str {'stationary', 'finite', 'isotropic', 'periodic',
                               'plus'}, optional
            String to select the edge handling to apply in computations. See
            the documentation for `PointPattern` for details.
            If not supplied, the edge correction falls back to the default
            value (set at instance initialization).

        Returns
        -------
        array
            Mean of the `PointPattern` attribute evaluated at `r`.

        """
        if edge_correction is None:
            edge_correction = self._edge_correction

        attr_frame = self._pp_attr_r_frame(attr, r,
                                           edge_correction=edge_correction,
                                           **kwargs)
        return attr_frame.mean(axis=0, skipna=True)

    def kframe(self, r, edge_correction=None):
        """
        Compute a DataFrame containing values of the empirical K-functions of
        the patterns

        Parameters
        ----------
        r : array-like
            Array of values at which to evaluate the emprical K-functions.
        edge_correction : str {'stationary', 'finite', 'isotropic', 'periodic',
                               'plus'}, optional
            String to select the edge handling to apply in computations. See
            the documentation for `PointPattern` for details.
            If not supplied, the edge correction falls back to the default
            value (set at instance initialization).

        Returns
        -------
        DataFrame
            DataFrame where each row contains values of the empirical
            K-function from one pattern, evaluated at `r`.

        """
        return self._pp_attr_r_frame('kfunction', r,
                                     edge_correction=edge_correction)

    def kcritical(self, alpha, r, edge_correction=None):
        """
        Compute critical values of the empirical K-functions of the patterns

        Parameters
        ----------
        alpha : scalar between 0.0 and 1.0
            Percentile defining the critical values.
        r : array-like
            Array of values at which to evaluate the critical values of the
            empirical K-functions.
        edge_correction : str {'stationary', 'finite', 'isotropic', 'periodic',
                               'plus'}, optional
            String to select the edge handling to apply in computations. See
            the documentation for `PointPattern` for details.
            If not supplied, the edge correction falls back to the default
            value (set at instance initialization).

        Returns
        -------
        array
            Critical values of the empirical K-functions evaluated at `r`.

        """
        return self._pp_attr_r_critical('kfunction', alpha, r,
                                        edge_correction=edge_correction)

    def kmean(self, r, edge_correction=None):
        """
        Compute the mean of the empirical K-functions of the patterns

        Parameters
        ----------
        r : array-like
            Array of values at which to evaluate the mean values of the
            empirical K-functions.
        edge_correction : str {'stationary', 'finite', 'isotropic', 'periodic',
                               'plus'}, optional
            String to select the edge handling to apply in computations. See
            the documentation for `PointPattern` for details.
            If not supplied, the edge correction falls back to the default
            value (set at instance initialization).

        Returns
        -------
        array
            Mean of the empirical K-functions evaluated at `r`.

        """
        return self._pp_attr_r_mean('kfunction', r,
                                    edge_correction=edge_correction)

    def lframe(self, r, edge_correction=None):
        """
        Compute a DataFrame containing values of the empirical L-functions of
        the patterns


        Parameters
        ----------
        r : array-like
            Array of values at which to evaluate the emprical L-functions.
        edge_correction : str {'stationary', 'finite', 'isotropic', 'periodic',
                               'plus'}, optional
            String to select the edge handling to apply in computations. See
            the documentation for `PointPattern` for details.
            If not supplied, the edge correction falls back to the default
            value (set at instance initialization).

        Returns
        -------
        DataFrame
            DataFrame where each row contains values of the empirical
            L-function from one pattern, evaluated at `r`.

        """
        return self._pp_attr_r_frame('lfunction', r,
                                     edge_correction=edge_correction)

    def lcritical(self, alpha, r, edge_correction=None):
        """
        Compute critical values of the empirical L-functions of the patterns

        Parameters
        ----------
        alpha : scalar between 0.0 and 1.0
            Percentile defining the critical values.
        r : array-like
            Array of values at which to evaluate the critical values of the
            empirical L-functions.
        edge_correction : str {'stationary', 'finite', 'isotropic', 'periodic',
                               'plus'}, optional
            String to select the edge handling to apply in computations. See
            the documentation for `PointPattern` for details.
            If not supplied, the edge correction falls back to the default
            value (set at instance initialization).

        Returns
        -------
        array
            Critical values of the empirical L-functions evaluated at `r`.

        """
        return self._pp_attr_r_critical('lfunction', alpha, r,
                                        edge_correction=edge_correction)

    def lmean(self, r, edge_correction=None):
        """
        Compute the mean of the empirical L-functions of the patterns

        Parameters
        ----------
        r : array-like
            Array of values at which to evaluate the mean values of the
            empirical L-functions.
        edge_correction : str {'stationary', 'finite', 'isotropic', 'periodic',
                               'plus'}, optional
            String to select the edge handling to apply in computations. See
            the documentation for `PointPattern` for details.
            If not supplied, the edge correction falls back to the default
            value (set at instance initialization).

        Returns
        -------
        array
            Mean of the empirical L-functions evaluated at `r`.

        """
        return self._pp_attr_r_mean('lfunction', r,
                                    edge_correction=edge_correction)

    def pair_corr_frame(self, r, bandwidth=None, edge_correction=None):
        """
        Compute a DataFrame containing values of the empirical pair correlation
        functions of the patterns

        Parameters
        ----------
        r : array-like
            Array of values at which to evaluate the functions.
        bandwidth : scalar
            The bandwidth of the box kernel used to estimate the density of
            points pairs at a given distance. See the documentation for
            `PointPattern.pair_corr_function` for details.
        edge_correction : str {'stationary', 'finite', 'isotropic', 'periodic',
                               'plus'}, optional
            String to select the edge handling to apply in computations. See
            the documentation for `PointPattern` for details.
            If not supplied, the edge correction falls back to the default
            value (set at instance initialization).

        Returns
        -------
        DataFrame
            DataFrame where each row contains values of the empirical
            pair correlation function from one pattern, evaluated at `r`.

        """
        return self._pp_attr_r_frame('pair_corr_function', r,
                                     bandwidth=bandwidth,
                                     edge_correction=edge_correction)

    def pair_corr_critical(self, alpha, r, bandwidth=None,
                           edge_correction=None):
        """
        Compute critical values of the empirical pair correlation functions of
        the patterns

        Parameters
        ----------
        alpha : scalar between 0.0 and 1.0
            Percentile defining the critical values.
        r : array-like
            Array of values at which to evaluate the critical values of the
            functions.
        bandwidth : scalar
            The bandwidth of the box kernel used to estimate the density of
            points pairs at a given distance. See the documentation for
            `PointPattern.pair_corr_function` for details.
        edge_correction : str {'stationary', 'finite', 'isotropic', 'periodic',
                               'plus'}, optional
            String to select the edge handling to apply in computations. See
            the documentation for `PointPattern` for details.
            If not supplied, the edge correction falls back to the default
            value (set at instance initialization).

        Returns
        -------
        array
            Critical values of the pair correlation functions evaluated at `r`.

        """
        return self._pp_attr_r_critical('pair_corr_function', alpha, r,
                                        bandwidth=bandwidth,
                                        edge_correction=edge_correction)

    def pair_corr_mean(self, r, bandwidth=None, edge_correction=None):
        """
        Compute the mean of the pair correlation functions of the patterns

        Parameters
        ----------
        r : array-like
            Array of values at which to evaluate the mean values of the
            functions.
        bandwidth : scalar
            The bandwidth of the box kernel used to estimate the density of
            points pairs at a given distance. See the documentation for
            `PointPattern.pair_corr_function` for details.
        edge_correction : str {'stationary', 'finite', 'isotropic', 'periodic',
                               'plus'}, optional
            String to select the edge handling to apply in computations. See
            the documentation for `PointPattern` for details.
            If not supplied, the edge correction falls back to the default
            value (set at instance initialization).

        Returns
        -------
        array
            Mean of the empirical pair correlation functions evaluated at `r`.

        """
        return self._pp_attr_r_mean('pair_corr_function', r,
                                    bandwidth=bandwidth,
                                    edge_correction=edge_correction)

    @memoize_method
    def _pp_attr_series(self, attr, edge_correction=None, **kwargs):
        """
        Compute a Series containing values of some scalar PointPattern
        attribute.

        Parameters
        ----------
        attr : string
            Name of `PointPattern` attribute to use.
        edge_correction : str {'stationary', 'finite', 'isotropic', 'periodic',
                               'plus'}, optional
            String to select the edge handling to apply in computations. See
            the documentation for `PointPattern` for details.
            If not supplied, the edge correction falls back to the default
            value (set at instance initialization).
        **kwargs : dict, optional
            Other arguments to pass to the `PointPattern` attribute.

        Returns
        -------
        Series
            Series containing values of the `PointPattern` attribute from each
            pattern.

        """
        if edge_correction is None:
            edge_correction = self._edge_correction

        return pandas.Series(
            [getattr(pp, attr)(edge_correction=edge_correction, **kwargs)
             for pp in self.patterns])

    def _pp_attr_test(self, attr, pattern, edge_correction=None, **kwargs):
        """
        Perform a statistical test on a PointPattern, based on the distribution
        of a PointPattern attribute over the patterns in this collection.

        Parameters
        ----------
        attr : string
            Name of `PointPattern` attribute to use.
        pattern : PointPattern
            PointPattern to perform the test on.
        edge_correction : str {'stationary', 'finite', 'isotropic', 'periodic',
                               'plus'}, optional
            String to select the edge handling to apply in computations. See
            the documentation for `PointPattern` for details.
            If not supplied, the edge correction falls back to the default
            value (set at instance initialization).
        **kwargs : dict, optional
            Other arguments to pass to the `PointPattern` attribute.

        Returns
        -------
        scalar
            The p-value computed using the the selected attribute as the test
            statistic for `pattern`, and its distribution over this collection
            as the null distribution.

        """
        if edge_correction is None:
            edge_correction = self._edge_correction

        tsdist = self._pp_attr_series(attr, edge_correction=edge_correction,
                                      **kwargs).dropna()
        teststat = getattr(pattern, attr)(edge_correction=edge_correction,
                                          **kwargs)
        return 1.0 - 0.01 * percentileofscore(tsdist, teststat, kind='mean')

    def lstatistics(self, rmin=None, rmax=None, edge_correction=None):
        """
        Compute the L test statistic for CSR for each pattern in the collection

        See `PointPattern.lstatistic` for details about the L test statistic.

        Parameters
        ----------
        rmin : scalar
            The minimum r value to use when computing the statistics. See the
            documentation for `PointPattern.lstatistic`for details.
        rmax : scalar
            The maximum r value to use when computing the statistics. See the
            documentation for `PointPattern.lstatistic`for details.
        edge_correction : str {'stationary', 'finite', 'isotropic', 'periodic',
                               'plus'}, optional
            String to select the edge handling to apply in computations. See
            the documentation for `PointPattern` for details.
            If not supplied, the edge correction falls back to the default
            value (set at instance initialization).

        Returns
        -------
        Series
            Series containing the L test statistic for each pattern in the
            collection.

        """
        return self._pp_attr_series('lstatistic', rmin=rmin, rmax=rmax,
                                    edge_correction=edge_correction)

    def ltest(self, pattern, rmin=None, rmax=None, edge_correction=None):
        """
        Perform an L test for CSR on a PointPattern, based on the distribution
        of L test statictics from the patterns in this collection.

        Parameters
        ----------
        pattern : PointPattern
            PointPattern to perform the L test on.
        rmin : scalar
            The minimum r value to use when computing the statistics. See the
            documentation for `PointPattern.lstatistic`for details.
        rmax : scalar
            The maximum r value to use when computing the statistics. See the
            documentation for `PointPattern.lstatistic`for details.
        edge_correction : str {'stationary', 'finite', 'isotropic', 'periodic',
                               'plus'}, optional
            String to select the edge handling to apply in computations. See
            the documentation for `PointPattern` for details.
            If not supplied, the edge correction falls back to the default
            value (set at instance initialization).

        Returns
        -------
        scalar
            The p-value of the L test statistic for `pattern`.

        """
        return self._pp_attr_test('lstatistic', pattern, rmin=rmin, rmax=rmax,
                                  edge_correction=edge_correction)

    @memoize_method
    def histogram(self, attribute, edge_correction=None, **kwargs):
        """
        Compute the histogram of a statistic of the patterns in the collection

        Parameters
        ----------
        attribute : {'lstatistic'}
            Statistic for which to compute the histogram. The valid names
            reflect the `PointPattern` attribute name for the corresponding
            statistic.
        edge_correction : str {'stationary', 'finite', 'isotropic', 'periodic',
                               'plus'}, optional
            String to select the edge handling to apply in computations. See
            the documentation for `PointPattern` for details.
            If not supplied, the edge correction falls back to the default
            value (set at instance initialization).
        **kwargs : dict, optional
            Keyword arguments passed to the `numpy.histogram` function.

        Returns
        -------
        hist: array
            The values of the histogram. See `numpy.histogram` for details.
        bin_edges: array of dtype float
            The bin edges: an array of length `len(hist) + 1` giving the edges
            of the histogram bins.

        """
        if edge_correction is None:
            edge_correction = self._edge_correction

        try:
            vals = getattr(self, attribute + 's')(
                edge_correction=edge_correction)
        except AttributeError:
            vals = numpy.array(
                [getattr(pp, attribute)(edge_correction=edge_correction)
                 for pp in self.patterns])
        return numpy.histogram(vals, **kwargs)

    def plot_kenvelope(self, axes=None, edge_correction=None, low=0.025,
                       high=0.975, alpha=0.25, **kwargs):
        """
        Plot an envelope of empirical K-function values

        Parameters
        ----------
        axes : Axes, optional
            Axes instance to add the envelope to. If None (default), the
            current Axes instance is used if any, or a new one created.
        edge_correction : str {'stationary', 'finite', 'isotropic', 'periodic',
                               'plus'}, optional
            String to select the edge handling to apply in computations. See
            the documentation for `PointPattern` for details.
            If not supplied, the edge correction falls back to the default
            value (set at instance initialization).
        low : scalar between 0.0 and `high`, optional
            Percentile defining the lower edge of the envelope.
        high : scalar between `low` and 1.0, optional
            Percentile defining the higher edge of the envelope.
        alpha : scalar between 0.0 and 1.0, optional
            The opacity of the envelope fill.
        **kwargs : dict, optional
            Additional keyword arguments pass to `axes.fill_between`. Note in
            particular the keywords 'edgecolor', 'facecolor' and 'label'.

        Returns
        -------
        PolyCollection
            The PolyCollection instance filling the envelope.

        """
        if axes is None:
            axes = pyplot.gca()

        if edge_correction is None:
            edge_correction = self._edge_correction

        rvals = numpy.linspace(0.0, self.rmax(edge_correction), RSAMPLES)
        kvals_low = self.kcritical(low, rvals, edge_correction=edge_correction)
        kvals_high = self.kcritical(high, rvals,
                                    edge_correction=edge_correction)

        h = axes.fill_between(rvals, kvals_low, kvals_high, alpha=alpha,
                              **kwargs)
        return h

    def plot_kmean(self, axes=None, edge_correction=None, **kwargs):
        """
        Plot the mean of the empirical K-function values

        Parameters
        ----------
        axes : Axes, optional
            Axes instance to add the mean to. If None (default), the current
            Axes instance is used if any, or a new one created.
        edge_correction : str {'stationary', 'finite', 'isotropic', 'periodic',
                               'plus'}, optional
            String to select the edge handling to apply in computations. See
            the documentation for `PointPattern` for details.
            If not supplied, the edge correction falls back to the default
            value (set at instance initialization).
        **kwargs : dict, optional
            Additional keyword arguments pass to `axes.plot`. Note in
            particular the keywords 'linestyle', 'color', and 'label'.

        Returns
        -------
        list
            List containing the Line2D of the plotted mean.

        """
        if axes is None:
            axes = pyplot.gca()

        if edge_correction is None:
            edge_correction = self._edge_correction

        rvals = numpy.linspace(0.0, self.rmax(edge_correction), RSAMPLES)
        kmean = self.kmean(rvals, edge_correction=edge_correction)

        h = axes.plot(rvals, kmean, **kwargs)
        return h

    def plot_lenvelope(self, axes=None, edge_correction=None, low=0.025,
                       high=0.975, alpha=0.25, **kwargs):
        """
        Plot an envelope of empirical L-function values

        Parameters
        ----------
        axes : Axes, optional
            Axes instance to add the envelope to. If None (default), the
            current Axes instance is used if any, or a new one created.
        edge_correction : str {'stationary', 'finite', 'isotropic', 'periodic',
                               'plus'}, optional
            String to select the edge handling to apply in computations. See
            the documentation for `PointPattern` for details.
            If not supplied, the edge correction falls back to the default
            value (set at instance initialization).
        low : scalar between 0.0 and `high`, optional
            Percentile defining the lower edge of the envelope.
        high : scalar between `low` and 1.0, optional
            Percentile defining the higher edge of the envelope.
        alpha : scalar between 0.0 and 1.0, optional
            The opacity of the envelope fill.
        **kwargs : dict, optional
            Additional keyword arguments pass to `axes.fill_between`. Note in
            particular the keywords 'edgecolor', 'facecolor' and 'label'.

        Returns
        -------
        PolyCollection
            The PolyCollection instance filling the envelope.

        """
        if axes is None:
            axes = pyplot.gca()

        if edge_correction is None:
            edge_correction = self._edge_correction

        rvals = numpy.linspace(0.0, self.rmax(edge_correction), RSAMPLES)
        lvals_low = self.lcritical(low, rvals, edge_correction=edge_correction)
        lvals_high = self.lcritical(high, rvals,
                                    edge_correction=edge_correction)

        h = axes.fill_between(rvals, lvals_low, lvals_high, alpha=alpha,
                              **kwargs)
        return h

    def plot_lmean(self, axes=None, edge_correction=None, **kwargs):
        """
        Plot an the mean of the empirical L-function values

        Parameters
        ----------
        axes : Axes, optional
            Axes instance to add the mean to. If None (default), the current
            Axes instance is used if any, or a new one created.
        edge_correction : str {'stationary', 'finite', 'isotropic', 'periodic',
                               'plus'}, optional
            String to select the edge handling to apply in computations. See
            the documentation for `PointPattern` for details.
            If not supplied, the edge correction falls back to the default
            value (set at instance initialization).
        **kwargs : dict, optional
            Additional keyword arguments pass to `axes.plot`. Note in
            particular the keywords 'linestyle', 'color', and 'label'.

        Returns
        -------
        list
            List containing the Line2D of the plotted mean.

        """
        if axes is None:
            axes = pyplot.gca()

        if edge_correction is None:
            edge_correction = self._edge_correction

        rvals = numpy.linspace(0.0, self.rmax(edge_correction), RSAMPLES)
        lmean = self.lmean(rvals, edge_correction=edge_correction)

        h = axes.plot(rvals, lmean, **kwargs)
        return h

    def plot_pair_corr_envelope(self, axes=None, bandwidth=None,
                                edge_correction=None, low=0.025, high=0.975,
                                alpha=0.25, **kwargs):
        """
        Plot an envelope of empirical pair correlation function values

        Parameters
        ----------
        axes : Axes, optional
            Axes instance to add the envelope to. If None (default), the
            current Axes instance is used if any, or a new one created.
        bandwidth : scalar
            The bandwidth of the box kernel used to estimate the density of
            points pairs at a given distance. See the documentation for
            `PointPattern.pair_corr_function` for details.
        edge_correction : str {'stationary', 'finite', 'isotropic', 'periodic',
                               'plus'}, optional
            String to select the edge handling to apply in computations. See
            the documentation for `PointPattern` for details.
            If not supplied, the edge correction falls back to the default
            value (set at instance initialization).
        low : scalar between 0.0 and `high`, optional
            Percentile defining the lower edge of the envelope.
        high : scalar between `low` and 1.0, optional
            Percentile defining the higher edge of the envelope.
        alpha : scalar between 0.0 and 1.0, optional
            The opacity of the envelope fill.
        **kwargs : dict, optional
            Additional keyword arguments pass to `axes.fill_between`. Note in
            particular the keywords 'edgecolor', 'facecolor' and 'label'.

        Returns
        -------
        PolyCollection
            The PolyCollection instance filling the envelope.

        """
        if axes is None:
            axes = pyplot.gca()

        if edge_correction is None:
            edge_correction = self._edge_correction

        rvals = numpy.linspace(0.0, self.rmax(edge_correction), RSAMPLES)
        gvals_low = self.pair_corr_critical(low, rvals, bandwidth=bandwidth,
                                            edge_correction=edge_correction)
        gvals_high = self.pair_corr_critical(high, rvals, bandwidth=bandwidth,
                                             edge_correction=edge_correction)

        h = axes.fill_between(rvals, gvals_low, gvals_high, alpha=alpha,
                              **kwargs)
        return h

    def plot_pair_corr_mean(self, axes=None, bandwidth=None,
                            edge_correction=None, **kwargs):
        """
        Plot an the mean of the empirical pair correlation function values

        Parameters
        ----------
        axes : Axes, optional
            Axes instance to add the mean to. If None (default), the current
            Axes instance is used if any, or a new one created.
        bandwidth : scalar
            The bandwidth of the box kernel used to estimate the density of
            points pairs at a given distance. See the documentation for
            `PointPattern.pair_corr_function` for details.
        edge_correction : str {'stationary', 'finite', 'isotropic', 'periodic',
                               'plus'}, optional
            String to select the edge handling to apply in computations. See
            the documentation for `PointPattern` for details.
            If not supplied, the edge correction falls back to the default
            value (set at instance initialization).
        **kwargs : dict, optional
            Additional keyword arguments pass to `axes.plot`. Note in
            particular the keywords 'linestyle', 'color', and 'label'.

        Returns
        -------
        list
            List containing the Line2D of the plotted mean.

        """
        if axes is None:
            axes = pyplot.gca()

        if edge_correction is None:
            edge_correction = self._edge_correction

        rvals = numpy.linspace(0.0, self.rmax(edge_correction), RSAMPLES)
        gmean = self.pair_corr_mean(rvals, bandwidth=bandwidth,
                                    edge_correction=edge_correction)

        h = axes.plot(rvals, gmean, **kwargs)
        return h

    def plot_aggregate_kfunction(self, axes=None, edge_correction=None,
                                 linewidth=2.0, csr=False, csr_kw=None,
                                 **kwargs):
        """
        Plot the aggregate of the empirical K-function over all patterns in the
        collection

        Parameters
        ----------
        axes : Axes, optional
            Axes instance to add the aggregate K-function to. If None
            (default), the current Axes instance is used if any, or a new one
            created.
        edge_correction : str {'stationary', 'finite', 'isotropic', 'periodic',
                               'plus'}, optional
            String to select the edge handling to apply in computations. See
            the documentation for `PointPattern` for details.
            If not supplied, the edge correction falls back to the default
            value (set at instance initialization).
        linewidth : scalar, optional
            The width of the line showing the K-function.
        csr : bool, optional
            If True, overlay the curve :math:`K(r) = \pi r^2`, which is the
            theoretical K-function for complete spatial randomness. The style
            of this line may be customized using csr_kw.
        csr_kw : dict, optional
            Keyword arguments to pass to `axes.plot` when plotting the CSR
            curve.
        **kwargs : dict, optional
            Additional keyword arguments pass to `axes.plot`. Note in
            particular the keywords 'linestyle', 'color', and 'label'.

        Returns
        -------
        list
            List of handles to the Line2D instances added to the plot, in the
            following order: aggregate K-function, CSR curve (optional).

        """
        if axes is None:
            axes = pyplot.gca()

        if edge_correction is None:
            edge_correction = self._edge_correction

        rmax = self.window.rmax(edge_correction)
        rvals = numpy.linspace(0.0, rmax, RSAMPLES)
        kvals = self.aggregate_kfunction(
            rvals, edge_correction=edge_correction)

        lines = axes.plot(rvals, kvals, linewidth=linewidth, **kwargs)

        if csr:
            if csr_kw is None:
                csr_kw = {}

            kcsr = numpy.pi * rvals * rvals
            lines += axes.plot(rvals, kcsr, linestyle='dashed', **csr_kw)

        return lines

    def plot_aggregate_lfunction(self, axes=None, edge_correction=None,
                                 linewidth=2.0, csr=False, csr_kw=None,
                                 **kwargs):
        """
        Plot the aggregate of the empirical L-function over all patterns in the
        collection

        Parameters
        ----------
        axes : Axes, optional
            Axes instance to add the aggregate L-function to. If None
            (default), the current Axes instance is used if any, or a new one
            created.
        edge_correction : str {'stationary', 'finite', 'isotropic', 'periodic',
                               'plus'}, optional
            String to select the edge handling to apply in computations. See
            the documentation for `PointPattern` for details.
            If not supplied, the edge correction falls back to the default
            value (set at instance initialization).
        linewidth : scalar, optional
            The width of the line showing the L-function.
        csr : bool, optional
            If True, overlay the curve :math:`L(r) = r`, which is the
            theoretical L-function for complete spatial randomness. The style
            of this line may be customized using csr_kw.
        csr_kw : dict, optional
            Keyword arguments to pass to `axes.plot` when plotting the CSR
            curve.
        **kwargs : dict, optional
            Additional keyword arguments pass to `axes.plot`. Note in
            particular the keywords 'linestyle', 'color', and 'label'.

        Returns
        -------
        list
            List of handles to the Line2D instances added to the plot, in the
            following order: aggregate L-function, CSR curve (optional).

        """
        if axes is None:
            axes = pyplot.gca()

        if edge_correction is None:
            edge_correction = self._edge_correction

        rmax = self.window.rmax(edge_correction)
        rvals = numpy.linspace(0.0, rmax, RSAMPLES)
        lvals = self.aggregate_lfunction(
            rvals, edge_correction=edge_correction)

        lines = axes.plot(rvals, lvals, linewidth=linewidth, **kwargs)

        if csr:
            if csr_kw is None:
                csr_kw = {}

            lines += axes.plot(rvals, rvals, linestyle='dashed', **csr_kw)

        return lines

    def plot_histogram(self, attribute, axes=None, edge_correction=None,
                       histtype='stepfilled', **kwargs):
        """
        Plot the histogram of a statistic of the patterns in the collection

        Parameters
        ----------
        attribute : {'lstatistic'}
            Statistic for which to plot the histogram. See the documentation
            for `PointPatternCollection.histogram` for details.
        axes : Axes, optional
            Axes instance to add the histogram to. If None (default), the
            current Axes instance is used if any, or a new one created.
        edge_correction : str {'stationary', 'finite', 'isotropic', 'periodic',
                               'plus'}, optional
            String to select the edge handling to apply in computations. See
            the documentation for `PointPattern` for details.
            If not supplied, the edge correction falls back to the default
            value (set at instance initialization).
        histtype : {'bar', 'step', 'stepfilled'}, optional
            The type of histogram to draw. See the documentation for
            `pyplot.hist` for details. Note that 'barstacked' is not a relevant
            option in this case, since a `PointPatternCollection` only provides
            a single set of data.
        **kwargs : dict, optional
            Keyword arguments passed to the `pyplot.hist` function.

        Returns
        -------
        list
            List of matplotlib patches used to create the histogram.

        """
        if axes is None:
            axes = pyplot.gca()

        if edge_correction is None:
            edge_correction = self._edge_correction

        try:
            vals = getattr(self, attribute + 's')(
                edge_correction=edge_correction)
        except AttributeError:
            vals = numpy.array(
                [getattr(pp, attribute)(edge_correction=edge_correction)
                 for pp in self.patterns])
        return axes.hist(vals, histtype=histtype, **kwargs)[2]
