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
from scipy.spatial import distance, Voronoi
from scipy.stats import percentileofscore
from shapely import geometry, affinity, ops, speedups
from matplotlib import pyplot, patches
from collections import Sequence

from .utils import AlmostImmutable, sensibly_divide, project_vectors
from .memoize.memoize import memoize_method

try:
    __ = basestring
except NameError:
    basestring = str

if speedups.available:
    speedups.enable()

_PI = numpy.pi
_2PI = 2.0 * _PI
_PI_4 = _PI / 4.0

RSAMPLES = 49
QUADLIMIT = 100
ORIGIN = geometry.Point((0.0, 0.0))


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

    @memoize_method
    def lattice(self):
        """
        Compute lattice vectors of a Bravais lattice having the window as unit
        cell

        The lattice vectors are stored as an n-by-2 array, with n the number of
        window edges, such that each row contains the coordinates of a lattice
        vector crossing a window edge

        If the window is not a simple plane-filling polygon (parallellogram or
        hexagon with reflection symmetry through its center), a ValueError
        is raised.

        Returns
        -------
        ndarray
            Array of lattice vectors.

        """
        vertices = (numpy.asarray(self.boundary)[:-1] - self.centroid)
        l = vertices.shape[0]
        vrotated = numpy.roll(vertices, l // 2, axis=0)
        if not (l in (4, 6) and numpy.allclose(vertices, -vrotated)):
            raise ValueError("window must be a simple plane-filling polygon "
                             "(a parallellogram, or a hexagon with reflection "
                             "symmetry through its center) to compute lattice "
                             "vectors.")
        lattice = vertices + numpy.roll(vertices, 1, axis=0)

        # Sort by angle, starting with the one closest to the x axis
        angles = numpy.arctan2(lattice[:, 1], lattice[:, 0])
        asort = numpy.argsort(angles)
        start_index = numpy.argmin(numpy.abs(angles[asort]))
        asort = numpy.roll(asort, -start_index)
        return lattice[asort]

    @memoize_method
    def inscribed_circle(self):
        """
        Compute the center and radius of the largest circle that can be
        inscribed in the polygon

        ..note:: The largest inscribed circle is found using a standard
        optimization routine. There is in principle no guarantee that it
        will converge to the global optimum that corresponds to the largest
        inscribed circle possible.

        Returns
        -------
        Series
            The x and y coordinates of the inscribed circle center, and the
            radius of the inscribed circle, with the index ('x', 'y', 'r').

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

        return dict(x=x, y=y, r=r)

    @memoize_method
    def longest_diagonal(self):
        """
        Compute the length of the longest diagonal across the polygon

        Returns
        -------
        scalar
            Length of the longest diagonal.

        """
        bpoints = list(geometry.MultiPoint(self.boundary.coords[:-1]))
        dmax = 0.0
        while bpoints:
            p1 = bpoints.pop()
            for p2 in bpoints:
                d = p1.distance(p2)
                if d > dmax:
                    dmax = d

        return dmax

    @memoize_method
    def voronoi(self):
        """
        Compute the central Voronoi unit cell of the lattice defined by the
        window

        Returns
        -------
        Window
            New window instance representing the lattice Voronoi unit cell,
            centered at the origin (not at the centroid of this Window
            instance).

        """
        lattice = self.lattice()
        lattice_r1 = numpy.roll(lattice, 1, axis=0)

        lattice_points = numpy.vstack(((0.0, 0.0), lattice,
                                       lattice + lattice_r1))
        voronoi = Voronoi(lattice_points)
        window = voronoi.vertices[voronoi.regions[voronoi.point_region[0]]]
        return type(self)(window)

    @memoize_method
    def centered(self):
        """
        Compute a translation of the window such that the centroid coincides
        with the origin

        Returns
        -------
        Window
            Centered window.

        """
        cent = self.centroid
        return affinity.translate(self, xoff=-cent.x, yoff=-cent.y)

    @memoize_method
    def diagonal_cut(self):
        """
        Compute the window obtained byt cutting this window in half along
        a diagonal

        This operation can only be performed on windows with an even number of
        vertices and reflection symmetry through the centroid. This ensures
        that all diagonals between opposite vertices cut the window into two
        halves.

        Returns
        -------
        Window
            Diagonally cut window.

        """
        boundary = numpy.asarray(self.boundary)[:-1]
        vertices = boundary - self.centroid
        l = vertices.shape[0]
        l_2 = l // 2
        vrotated = numpy.roll(vertices, l_2, axis=0)
        if not (l % 2 == 0 and numpy.allclose(vertices, -vrotated)):
            raise ValueError("window must have an even number of vertices and "
                             "reflection symmetry through its centroid to "
                             "compute diagonal cut.")

        # We want to begin in the lower right quadrant
        angles = numpy.arctan2(vertices[:, 1], vertices[:, 0])
        asort = numpy.argsort(angles)
        start_index = numpy.argmin(numpy.abs(angles[asort] + _PI_4))
        asort = numpy.roll(asort, -start_index)
        new_boundary = boundary[asort[:l_2 + 1]]
        return type(self)(new_boundary)

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
        for p in numpy.asarray(sbpoints):
            plist.append(affinity.translate(other, xoff=p[0], yoff=p[1]))
        for p in numpy.asarray(obpoints):
            plist.append(affinity.translate(self, xoff=p[0], yoff=p[1]))
        return ops.cascaded_union(plist)

    def erode_by_this(self, other):
        """
        Erode another polygon by this polygon

        :other: polygon to erode
        :returns: eroded polygon

        NB! Don't know if this algorithm is correct in all cases

        """
        eroded = type(self)(other)
        sbpoints = geometry.MultiPoint(self.boundary)[:-1]
        for p in numpy.asarray(sbpoints):
            eroded = eroded.intersection(affinity.translate(other, xoff=-p[0],
                                                            yoff=-p[1]))
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
        ld = self.longest_diagonal()
        rssqrt = int(numpy.sqrt(RSAMPLES))
        xoffs = numpy.linspace(-ld, ld, 4 * (rssqrt + 1) - 1)
        yoffs = numpy.linspace(-ld, ld, 4 * (rssqrt + 1) - 1)
        scarray = numpy.zeros((xoffs.size, yoffs.size))
        for (i, xoff) in enumerate(xoffs):
            for (j, yoff) in enumerate(yoffs):
                scarray[i, j] = self.translated_intersection(xoff, yoff).area
        #return interpolate.RegularGridInterpolator((xoffs, yoffs), scarray,
        #                                           bounds_error=False,
        #                                           fill_value=0.0)
        return interpolate.RectBivariateSpline(xoffs, yoffs, scarray,
                                               kx=3, ky=3)

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
        #xi = numpy.concatenate((x[..., numpy.newaxis],
        #                        y[..., numpy.newaxis]), axis=-1)
        #return self._set_covariance_interpolator()(xi)
        return self._set_covariance_interpolator()(x, y, grid=False)

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
        rvals = numpy.linspace(0.0, self.longest_diagonal(),
                               2 * (RSAMPLES + 1) - 1)
        iso_set_cov = numpy.zeros_like(rvals)

        # Identify potentially problematic angles and a safe starting- and
        # ending angle for the quadrature integration
        xy = numpy.asarray(self.boundary)[:-1]
        problem_angles = numpy.sort(numpy.arctan2(xy[:, 1], xy[:, 0]))
        theta0 = 0.5 * (problem_angles[0] + problem_angles[-1] - _2PI)

        for (i, rval) in enumerate(rvals):
            def integrand(theta):
                return self.set_covariance(rval * numpy.cos(theta),
                                           rval * numpy.sin(theta))

            iso_set_cov[i] = (integrate.quad(integrand, theta0,
                                             _2PI + theta0,
                                             limit=QUADLIMIT,
                                             points=problem_angles)[0] / _2PI)

        return interpolate.interp1d(rvals, iso_set_cov, kind='cubic',
                                    bounds_error=False, fill_value=0.0)

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
    def _ball_difference_area_interpolator(self):
        """
        Compute a ball difference area interpolator for the window

        Returns
        -------
        interp1d
            Interpolator that computes the the ball difference area for the
            window.

        """
        rvals = numpy.linspace(0.0, .5 * self.longest_diagonal(), RSAMPLES)
        ball_diff_area = numpy.zeros_like(rvals)
        centroid = self.centroid
        for (i, r) in enumerate(rvals):
            disc = centroid.buffer(r)
            ball_diff_area[i] = self.difference(disc).area
        return interpolate.interp1d(rvals, ball_diff_area, kind='cubic',
                                    bounds_error=False, fill_value=0.0)

    def ball_difference_area(self, r):
        """
        Compute the area of the set difference of the window and a ball of
        a given radius centered on the window centroid

        This function provides a speedup of this computation for multiple
        values of the radius, by relying on an interpolator.

        Parameters
        ----------
        r : array-like
            Array giving the radii of the balls to subtract from the window.

        Returns
        ndarray
            Array of the same shape as `r` containing for each value in `r` the
            area of the set difference of the window and b(c, r), where c is
            the centroid of the window.

        """
        return self._ball_difference_area_interpolator()(r)

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
            return _2PI * t * self.isotropised_set_covariance(t)

        rvals = numpy.linspace(0.0, self.longest_diagonal(), RSAMPLES)
        dvals = numpy.empty_like(rvals)
        for (i, rval) in enumerate(rvals):
            dvals[i] = integrate.quad(integrand, 0.0, rval,
                                      limit=QUADLIMIT,
                                      )[0]

        return interpolate.interp1d(rvals, dvals, kind='cubic',
                                    bounds_error=True)

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
        r = numpy.asarray(r)
        num = numpy.empty_like(r)
        r_ravel = r.ravel()
        num_ravel = num.ravel()
        for (i, rval) in enumerate(r_ravel):
            num_ravel[i] = self.intersection(point.buffer(rval)).area

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
        r = numpy.asarray(r)
        num = numpy.empty_like(r)
        r_ravel = r.ravel()
        num_ravel = num.ravel()
        for (i, rval) in enumerate(r_ravel):
            num_ravel[i] = self.intersection(
                point.buffer(rval).boundary).length

        denom = _2PI * r * self.isotropised_set_covariance(r)

        return sensibly_divide(num, denom)

    def patch(self, **kwargs):
        """
        Return a matplotlib.patches.Polygon instance for this window

        :kwargs: passed through to the matplotlib.patches.Polygon constructor
        :returns: matplotlib.patches.Polygon instance

        """
        return patches.Polygon(self.boundary, **kwargs)

    def plot(self, axes=None, linewidth=2.0, fill=False, **kwargs):
        """
        Plot the window

        The window can be added to an existing plot via the optional 'axes'
        argument.

        :axes: Axes instance to add the window to. If None (default), the
               current Axes instance with equal aspect ratio is used if any, or
               a new one created.
        :linewidth: the linewidth to use for the window boundary. Defaults to
                    2.0.
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
            diag = self.longest_diagonal()
            axes.set(xlim=(cent.x - diag, cent.x + diag),
                     ylim=(cent.y - diag, cent.y + diag))

        wpatch = self.patch(linewidth=linewidth, fill=fill, **kwargs)
        wpatch = axes.add_patch(wpatch)

        return wpatch


class PointPattern(AlmostImmutable, Sequence):
    """
    Represent a planar point pattern and its associated window, and provide
    methods for analyzing its statistical properties

    Parameters
    ----------
    points : sequence or MultiPoint
        A sequence of coordinate tuples or any other valid MultiPoint
        constructor argument, representing the points in the point pattern.
    window : sequence or Polygon or Window
        A sequence of coordinate tuples or any other valid Window constructor
        argument, defining the set within which the point pattern is takes
        values. A ValueError is raised if the window does not contain all
        points in `points`. The static method `wrap_into` from this class can
        be used to wrap points into the window before initalization, if the
        window is a simple plane-filling polygon (thus providing periodic
        boundary conditions by which the points can be wrapped).
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
            'imode': 'standard',  # corrected,
        },
        'isotropic': {
            'pmode': 'default',
            'imode': 'perimeter',
        },
        'periodic': {
            'pmode': 'periodic',
            'imode': 'standard',  # corrected,
        },
        'plus': {
            'pmode': 'plus',
            'imode': 'standard',  # corrected,
        },
    }

    def __init__(self, points, window, pluspoints=None,
                 edge_correction='stationary'):
        # Avoid copying the window unless needed
        if not isinstance(window, Window):
            window = Window(window)
        self.window = window

        points = geometry.MultiPoint(points)
        if len(set(map(tuple, numpy.asarray(points)))) != len(points):
            raise ValueError("{} instances do not support point patterns "
                             "with multiple exactly equal points"
                             .format(type(self)))

        if not window.contains(points):
            raise ValueError("Not all points in 'points' are contained inside "
                             "'window'.")
        self._points = points

        self.pluspoints = geometry.MultiPoint(pluspoints)

        self.edge_correction = edge_correction

    # Implement abstract methods
    def __getitem__(self, index, *args, **kwargs):
        return self._points.__getitem__(index, *args, **kwargs)

    def __len__(self, *args, **kwargs):
        return self._points.__len__(*args, **kwargs)

    # Override certain possibly very slow mixins
    def __iter__(self, *args, **kwargs):
        return self._points.__iter__(*args, **kwargs)

    def __reversed__(self, *args, **kwargs):
        return self._points.__reversed__(*args, **kwargs)

    def index(self, *args, **kwargs):
        return self._points.index(*args, **kwargs)

    @staticmethod
    def wrap_into(window, points):
        """
        Wrap a set of points into a plane-filling window

        Parameters
        ----------
        window : Window
            A sequence of coordinate tuples or any other valid Window
            constructor argument, defining the window to wrap the points into.
            A `ValueError` is raised if the window not a simple plane-filling
            polygon.
        points : MultiPoint
            A sequence of coordinate tuples or any other valid MultiPoint
            constructor argument, representing the points to wrap.

        Returns
        -------
        MultiPoint
            New `MultiPoint` instance containing the wrapped representation of
            all points in `points`.

        """
        if not isinstance(points, geometry.MultiPoint):
            points = geometry.MultiPoint(points)
        if not isinstance(window, Window):
            window = Window(window)

        if window.contains(points):
            return points

        lattice = window.lattice()
        basis = lattice[:2]

        # Wrap points directly into the concentric rhomboidal window
        parray = numpy.asarray(points)
        pcoeffs = project_vectors(parray, basis)
        ccoeff = project_vectors(window.centroid, basis)
        pcoeffs = numpy.mod(pcoeffs + 0.5 - ccoeff, 1.0) - 0.5 + ccoeff
        parray = pcoeffs.dot(basis)
        points = geometry.MultiPoint(parray)

        # If window is hexagonal, there may be some residual wrapping to do
        if len(lattice) == 6 and not window.contains(points):
            vectors = numpy.vstack((basis, -basis))
            new_points = []
            for p in points:
                if window.contains(p):
                    new_points.append(p)
                    continue
                for v in vectors:
                    wp = affinity.translate(p, xoff=v[0], yoff=v[1])
                    if window.contains(wp):
                        new_points.append(wp)
                        break
            points = geometry.MultiPoint(new_points)
        return points

    def _inherit_binary_operation(self, other, op):
        """
        Define the general pattern for inheriting a binary operation on the
        points as a binary operation on the PointPattern

        Parameters
        ----------
        other : shapely object
            The binary operation is applied to `self` and `other`. If `other`
            is also a `PointPattern` instance, an exception is raised if they
            are not defined in `Window` instances that compare equal. If
            `other` is not a `PointPattern` instance, the binary operation is
            applied to `self._points` and `other`. The result of this operation
            is returned directly, unless it is a `geometry.MultiPoint` or
            `geometry.Point` instance, in which case it is used to initialize
            a new `PointPattern` instance in the same window as `self`. If
            applying the binary operation to `self.pluspoints` and `other` also
            returns a `geometry.MultiPoint` or `geometry.Point` instance, this
            is used as the `pluspoints` of the new `PointPattern`.
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
        if (isinstance(op, basestring) and
                hasattr(spoints, op) and
                hasattr(spluspoints, op)):
            bound_op = getattr(spoints, op)
            bound_op_plus = getattr(spluspoints, op)
        else:
            def bound_op(ogeom):
                return op(spoints, ogeom)

            def bound_op_plus(opluspoints):
                return op(spluspoints, opluspoints)

        swindow = self.window
        if isinstance(other, type(self)) or isinstance(self, type(other)):
            owindow = other.window
            if not (swindow == owindow):
                raise ValueError("instances of {} must be defined over "
                                 "instances of {} that compare equal for "
                                 "binary operations to be defined"
                                 .format(self.__class__.__name__,
                                         swindow.__class__.__name__))
            new_points = bound_op(other._points)
            new_pluspoints = bound_op_plus(other.pluspoints)
            return type(self)(new_points, swindow, pluspoints=new_pluspoints,
                              edge_correction=self.edge_correction)

        # Apparently, other is not a PointPattern. Do the easiest thing.
        new_geom = bound_op(other)
        if isinstance(new_geom, geometry.Point):
            new_geom = geometry.MultiPoint((new_geom,))
        if isinstance(new_geom, geometry.MultiPoint):
            new_pluspoints = None
            potential_pluspoints = bound_op_plus(other)
            if isinstance(potential_pluspoints, geometry.Point):
                potential_pluspoints = geometry.MultiPoint((new_pluspoints,))
            if isinstance(potential_pluspoints, geometry.MultiPoint):
                new_pluspoints = potential_pluspoints
            return type(self)(
                new_geom, swindow, pluspoints=new_pluspoints,
                edge_correction=self.edge_correction)
        return new_geom

    def difference(self, other):
        return self._inherit_binary_operation(other, 'difference')

    def intersection(self, other):
        return self._inherit_binary_operation(other, 'intersection')

    def symmetric_difference(self, other):
        return self._inherit_binary_operation(other, 'symmetric_difference')

    def union(self, other):
        return self._inherit_binary_operation(other, 'union')

    def periodic_extension(self, periodic_levels):
        """
        Compute the periodic extension of this point pattern

        The extension is made by assuming that periodic boundary conditions
        hold across the boundaries of the window associated with the pattern.

        Returns
        -------
        periodic_levels : integer
            The number of levels of periodic extensions to compute. A level
            roughly consists of all the lattice displacements that can be
            written as a sum of an equal number of lattice unit vectors.
        MultiPoint
            MultiPoint instance containing the points comprising the periodic
            extension. Note that the points from the pattern itself are not
            included.

        """
        lattice = self.window.lattice()
        lattice_r1 = numpy.roll(lattice, 1, axis=0)
        dvec_list = []
        for i in range(periodic_levels + 1):
            for l in range(i):
                k = i - l
                dvec_list.append(k * lattice + l * lattice_r1)
        dvecs = numpy.vstack(dvec_list)
        periodic_points = ops.cascaded_union(
            [affinity.translate(self.points(), xoff=dvec[0], yoff=dvec[1])
             for dvec in dvecs])
        return periodic_points

    def points(self, mode='default', periodic_levels=2, project_points=False):
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
        periodic_levels : integer, optional
            The number of periodic levels to compute if `mode == 'periodic'`.
            See `PointPattern.periodic_extension` for explanation.
        project_points : bool, optional
            If True, the points will be projected into the unit square by
            oblique projection onto the edges of the window of the point
            pattern. The periodic extension points or plus sampling points will
            of course take values outside the unit square, but will be subject
            to the same transformation. If the window is not rhomboidal, an
            error will be raised.

        Returns
        -------
        MultiPoint
            MultiPoint instance containing the requested points.

        """
        if mode == 'default':
            points = self._points
        elif mode == 'periodic':
            points = self._points.union(
                self.periodic_extension(periodic_levels))
        elif mode == 'plus':
            points = self._points.union(self.pluspoints)
        else:
            raise ValueError("unknown mode: {}".format(mode))

        if project_points:
            basis_vectors = self.window.lattice()
            if len(basis_vectors) != 4:
                raise ValueError("projection is only possible for point "
                                 "patterns in rhomboidal windows.")
            basis_vectors = basis_vectors[:2]

            # Find the lower left corner (with respect to the basis vectors)
            # of the window
            boundary = numpy.asarray(self.window.boundary)[:-1]
            boundary_coeffs = project_vectors(boundary, basis_vectors)
            anchor_coeffs = min(boundary_coeffs, key=numpy.sum)

            # Subtract anchor and project
            parray = numpy.array(points) - anchor_coeffs.dot(basis_vectors)
            point_coeffs = project_vectors(parray, basis_vectors)
            points = geometry.MultiPoint(point_coeffs)

        return points

    @staticmethod
    def range_tree_build(points):
        """
        Construct a range tree from a set of points

        Parameters
        ----------
        points : sequence
            Sequence of coordinate tuples instances to build the range tree
            from. ..note:: shapely Point instances are not supported.

        Returns
        -------
        tuple
            Root node of the range tree. The nodes are tuples in the
            following format:
            [median_point, left_child, right_child, associated_binary_tree].
            The associated binary tree at each node points to the root node of
            a binary tree with nodes in the following format:
            [median_point_r, left_child, right_child]. Here, `median_point` is
            a regular coordinate tuple, while `median_point_r` is a reversed
            coordinate tuple.

        """
        def binary_node_stuff(points, sort_index):
            # Binary tree node format: [point, left, right]
            mid = len(sort_index) // 2
            p = points[sort_index[mid]]
            si_l, si_r = sort_index[:mid], sort_index[mid:]
            return [p, None, None], si_l, si_r

        def build_binary_tree(points, sort_index):
            root_stuff = binary_node_stuff(points, sort_index)

            stack = []
            if len(sort_index) > 1:
                stack.append(root_stuff)
            while stack:
                current, si_l, si_r = stack.pop()
                left_stuff = binary_node_stuff(points, si_l)
                current[1] = left_stuff[0]
                if len(si_l) > 1:
                    stack.append(left_stuff)
                right_stuff = binary_node_stuff(points, si_r)
                current[2] = right_stuff[0]
                if len(si_r) > 1:
                    stack.append(right_stuff)
            return root_stuff[0]

        def range_node_stuff(points, xsort_index, points_r, ysort_index):
            # Range tree node format: [point, left, right,
            #                          associated_binary_tree)
            b = build_binary_tree(points_r, ysort_index)

            mid = len(xsort_index) // 2
            p = points[xsort_index[mid]]
            xsi_l, xsi_r = xsort_index[:mid], xsort_index[mid:]
            ysi_l = [yi for yi in ysort_index if yi in xsi_l]
            ysi_r = [yi for yi in ysort_index if yi in xsi_r]
            return [p, None, None, b], xsi_l, xsi_r, ysi_l, ysi_r

        def build_range_tree(points, xsort_index, points_r, ysort_index):
            root_stuff = range_node_stuff(points, xsort_index,
                                          points_r, ysort_index)

            stack = []
            if len(xsort_index) > 1:
                stack.append(root_stuff)
            while stack:
                current, xsi_l, xsi_r, ysi_l, ysi_r = stack.pop()
                left_stuff = range_node_stuff(points, xsi_l, points_r, ysi_l)
                current[1] = left_stuff[0]
                if len(xsi_l) > 1:
                    stack.append(left_stuff)
                right_stuff = range_node_stuff(points, xsi_r, points_r, ysi_r)
                current[2] = right_stuff[0]
                if len(xsi_r) > 1:
                    stack.append(right_stuff)

            return root_stuff[0]

        indices = range(len(points))
        points_r = [p[::-1] for p in points]
        xsort_index = sorted(indices, key=lambda i: points[i])
        ysort_index = sorted(indices, key=lambda i: points_r[i])
        return build_range_tree(points, xsort_index, points_r, ysort_index)

    @staticmethod
    def range_tree_query(tree, xmin, xmax, ymin, ymax):
        """
        Return the points stored in a range tree that lie inside a rectangular
        region

        Parameters
        ----------
        root : tuple
            Root node of the range tree, as returned from
            `PointPattern.range_tree_static`.
        xmin, xmax, ymin, ymax : scalars
            Limits of the range in which to query the range tree for points.
            Limits are inclusive in both ends.

        Returns
        -------
        list
            List of coordinate tuples for all points from the tree inside the
            given range.

        """
        xmin, xmax = (xmin, -numpy.inf), (xmax, numpy.inf)
        ymin, ymax = (ymin, -numpy.inf), (ymax, numpy.inf)

        def isleaf(node):
            return (node[1] is None) and (node[2] is None)

        def query(root, min_, max_, report, points):
            # Find split node.
            split = root
            while not isleaf(split):
                x = split[0]
                if x > max_:
                    split = split[1]
                elif x <= min_:
                    split = split[2]
                else:
                    break
            else:
                # The split node is a leaf node. Report if relevant and finish.
                if min_ <= split[0] <= max_:
                    report(split, points)
                return
            # The split node is a non-leaf node: traverse subtrees and report
            # relevant nodes.
            # First, take the left subtree.
            node = split[1]
            while not isleaf(node):
                if node[0] > min_:
                    # The whole right subtree is relevant. Report it.
                    report(node[2], points)
                    node = node[1]
                else:
                    node = node[2]
            # We end on a leaf node. Report if relevant.
            if min_ <= node[0] <= max_:
                report(node, points)
            # Then take the right subtree.
            node = split[2]
            while not isleaf(node):
                if node[0] <= max_:
                    # The whole left subtree is relevant. Report it.
                    report(node[1], points)
                    node = node[2]
                else:
                    node = node[1]
            # We end on a leaf node. Report if relevant.
            if min_ <= node[0] <= max_:
                report(node, points)

        def report_subtree_r(node, points):
            stack = [node]
            while stack:
                node = stack.pop()
                if isleaf(node):
                    points.append(node[0][::-1])
                else:
                    stack.extend(node[1:3])

        def report_yquery(node, points):
            return query(node[3], ymin, ymax, report_subtree_r, points)

        points = []
        query(tree, xmin, xmax, report_yquery, points)

        return points

    #@memoize_method
    def range_tree(self, project_points=True):
        """
        Construct a range tree from the points in the pattern

        Only the actual points in the pattern are added to the range tree --
        plus sampling points or points from the periodic extension is never
        used.

        Parameters
        ----------
        project_points : bool, optional
            Passed to `PointPattern.points`.

        Returns
        -------
        Root node of the range tree. For details about the type and format, see
        `PointPattern.range_tree_static`.

        """
        points = self.points(project_points=project_points)
        return self.range_tree_build([tuple(p)
                                      for p in numpy.asarray(points)])

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
        ap1 = numpy.array(pp1)[:, :2]
        if pp2 is not None:
            ap2 = numpy.array(pp2)[:, :2]
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
        ap1 = numpy.array(pp1)[:, :2]
        if pp2 is not None:
            ap2 = numpy.array(pp2)[:, :2]
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

    def rmax(self, edge_correction=None):
        """
        Return the largest relevant interpoint distance for a given edge
        correction in the window of this point pattern

        Parameters
        ----------
        edge_correction : str {'stationary', 'finite', 'isotropic', 'periodic',
                               'plus'}
            String to select the edge handling to apply in computations. See
            the documentation for `PointPattern` for details.  If None, the
            edge correction falls back to the default value (set at instance
            initialization).

        Returns
        -------
        scalar
            Largest relevant interpoint distance.

        """
        if edge_correction is None:
            edge_correction = self.edge_correction

        if edge_correction in ('finite', 'plus', 'isotropic'):
            return self.window.longest_diagonal()

        elif edge_correction == 'periodic':
            return 0.5 * self.window.voronoi().longest_diagonal()

        elif edge_correction == 'stationary':
            return 2.0 * self.window.inscribed_circle()['r']

        else:
            raise ValueError("unknown edge correction: {}"
                             .format(edge_correction))

    def rvals(self, edge_correction=None):
        """
        Construct an array of r values tailored for the empirical K/L-functions

        The returned array contains a pair of tightly spaced values around each
        vertical step in the K/L-functions, and evenly spaced r values with
        moderate resolution elsewhere.

        Parameters
        ----------
        edge_correction : str {'stationary', 'finite', 'isotropic', 'periodic',
                               'plus'}, optional
            String to select the edge handling to apply in computations. See
            the documentation for `PointPattern` for details.  If None, the
            edge correction falls back to the default value (set at instance
            initialization).

        Returns
        -------
        array
            Array of r values tailored to the empirical K/L-functions

        """
        if edge_correction is None:
            edge_correction = self.edge_correction
        rmax = self.rmax(edge_correction=edge_correction)
        rvals = numpy.linspace(0.0, rmax, RSAMPLES)

        # Get step locations
        rsteps, __ = self._estimator_base(edge_correction)
        micrormax = 1.e-6 * rmax
        rstep_values = numpy.repeat(rsteps, 2)
        rstep_values[0::2] -= micrormax
        rstep_values[1::2] += micrormax

        # Add r values tightly around each step
        rstep_indices = numpy.searchsorted(rvals, rstep_values)
        rvals = numpy.insert(rvals, rstep_indices, rstep_values)
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

            voronoi = window.voronoi()
            area_inv = 1.0 / voronoi.area
            centroid = voronoi.centroid
            #distances = PointPattern.pairwise_distances(mp1, mp2)
            pdisps = numpy.asarray(mp1) - numpy.asarray(centroid)
            for (i, pd) in enumerate(pdisps):
                translated_window = affinity.translate(voronoi,
                                                       xoff=pd[0], yoff=pd[1])
                valid_mp2 = mp2.intersection(translated_window)
                vmp2_arr = numpy.atleast_2d(valid_mp2)
                vindex = numpy.any(distance.cdist(mp2_arr, vmp2_arr) == 0.0,
                                   axis=-1)
                #vindex = numpy.any(
                #    numpy.all(PointPattern.pairwise_vectors(
                #        mp2_arr, vmp2_arr) == 0.0, axis=-1), axis=-1)

                wview[i, vindex] = area_inv

                ## Isotropic edge correction (to cancel corner effects that are
                ## still present for large r)
                #for j in numpy.nonzero(vindex)[0]:
                #    r = distances[i, j]
                #    ring = centroid.buffer(r).boundary
                #    wview[i, j] *= (_2PI * r /
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

            for (i, p1) in enumerate(mp1):
                for j in range(n):
                    r = distances[i, j]
                    ring = p1.buffer(r).boundary
                    rball = ORIGIN.buffer(r)
                    doughnut = window.difference(window.erode_by_this(rball))
                    w[i, j] = _2PI * r / (
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
        rmax = self.rmax(edge_correction=edge_correction)
        pmode = self._edge_config[edge_correction]['pmode']

        allpoints = self.points(mode=pmode)
        distances = self.pairwise_distances(self._points, allpoints)
        valid = numpy.logical_and(distances < rmax, distances != 0.0)

        index1, = numpy.nonzero(numpy.any(valid, axis=1))
        index2, = numpy.nonzero(numpy.any(valid, axis=0))
        mp1 = geometry.MultiPoint([self[i] for i in index1])
        mp2 = geometry.MultiPoint([allpoints[i] for i in index2])
        weight_matrix = self.pair_weights(self.window, mp1, mp2,
                                          edge_correction)

        r = distances[valid]
        sort_ind = numpy.argsort(r)
        r = r[sort_ind]

        weights = weight_matrix[valid[index1, :][:, index2]]
        weights = weights[sort_ind]

        return r, weights

    def _cumulative_base(self, edge_correction):
        """
        Compute the cumulative weight of the points in the pattern
        Parameters
        ----------
        edge_correction : str {'stationary', 'finite', 'isotropic', 'periodic',
                               'plus'}, optional
            String to select the edge handling to apply in computations. See
            the documentation for `PointPattern` for details.

        Returns
        -------
        rsteps : ndarray
            Array containing the r values between `rmin`and `rmax`  at which
            the cumulative characteristics make jumps.
        cweights : ndarray
             Array of the same shape as `rsteps`, containing the value of the
             cumulated weights just after each step.

        """
        rmax = self.rmax(edge_correction=edge_correction)
        rsteps, weights = self._estimator_base(edge_correction)
        rsteps = numpy.hstack((0.0, rsteps, rmax))
        weights = numpy.hstack((0.0, weights, numpy.nan))
        cweights = numpy.cumsum(weights)
        return rsteps, cweights

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
            the documentation for `PointPattern` for details.  If None, the
            edge correction falls back to the default value (set at instance
            initialization).

        Returns
        -------
        array
            Values of the empirical K-function evaulated at `r`.

        """
        if edge_correction is None:
            edge_correction = self.edge_correction

        rsteps, cweights = self._cumulative_base(
            edge_correction=edge_correction)

        indices = numpy.searchsorted(rsteps, r, side='right') - 1

        imode = self._edge_config[edge_correction]['imode']
        lambda2 = self.squared_intensity(mode=imode, r=r)
        return sensibly_divide(cweights[indices], lambda2)

    def lfunction(self, r, edge_correction=None):
        """
        Evaluate the empirical L-function of the point pattern

        Parameters
        ----------
        r : array-like
            array of values at which to evaluate the emprical L-function.
        edge_correction : str {'stationary', 'finite', 'isotropic', 'periodic',
                               'plus'}, optional
            String to select the edge handling to apply in computations. See
            the documentation for `PointPattern` for details.  If None, the
            edge correction falls back to the default value (set at instance
            initialization).

        Returns
        -------
        array
            Values of the empirical L-function evaulated at `r`.

        """
        if edge_correction is None:
            edge_correction = self.edge_correction

        return numpy.sqrt(self.kfunction(r, edge_correction=edge_correction) /
                          _PI)

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
            points pairs at a given distance. If None, the bandwidth is set to
            :math:`0.2 / \sqrt(\lambda)`, where :math:`\lambda` is the standard
            intensity estimate for the process.
        edge_correction : str {'stationary', 'finite', 'isotropic', 'periodic',
                               'plus'}, optional
            String to select the edge handling to apply in computations. See
            the documentation for `PointPattern` for details.  If None, the
            edge correction falls back to the default value (set at instance
            initialization).

        Returns
        -------
        array
            Values of the empirical pair correlation function evaulated at `r`.

        """
        if edge_correction is None:
            edge_correction = self.edge_correction

        if bandwidth is None:
            bandwidth = 0.2 / numpy.sqrt(self.intensity())

        rpairs, weights = self._estimator_base(edge_correction)

        # Find the contribution from each pair to each element in `r`
        d = numpy.abs(r[numpy.newaxis, ...] - rpairs[..., numpy.newaxis])
        w = numpy.sum((d < bandwidth) * weights[..., numpy.newaxis], axis=0)
        w *= 1.0 / (2.0 * _2PI * r * bandwidth)

        imode = self._edge_config[edge_correction]['imode']
        lambda2 = self.squared_intensity(mode=imode, r=r)
        return sensibly_divide(w, lambda2)

    def kfunction_std(self, r, edge_correction=None):
        """
        Compute the theoretical standard deviation of the empirical k-function
        of a point pattern like this one, under the CSR hypothesis.

        The ``theoretical'' standard deviation is really an empirically
        validated formula, and should be a very good fit to the true standard
        deviation within the interval given by
        `PointPattern.kstatistic_interval`. It is currently only implemented
        for periodic boundary conditions -- an array of ones is returned for
        other edge corrections.

        Parameters
        ----------
        r : array-like
            array of values at which to evaluate the emprical K-function
            standard deviation.
        edge_correction : str {'stationary', 'finite', 'isotropic', 'periodic',
                               'plus'}, optional
            String to select the edge handling to apply in computations. See
            the documentation for `PointPattern` for details.  If None, the
            edge correction falls back to the default value (set at instance
            initialization).

        Returns
        -------
        array
            Values of the standard deviation of the empirical K-function,
            evaulated at `r`.

        """
        if edge_correction is None:
            edge_correction = self.edge_correction

        r = numpy.asarray(r)
        if edge_correction == 'periodic':
            imode = self._edge_config[edge_correction]['imode']
            squared_intensity = self.squared_intensity(r=r, mode=imode)
            voronoi = self.window.voronoi()
            area = voronoi.area
            npnp_1 = area * area * squared_intensity
            kstd = r * numpy.sqrt(2.0 * _PI * voronoi.ball_difference_area(r) /
                                  npnp_1)
                                  #(npnp_1 + 0.5 + numpy.sqrt(npnp_1 + 0.25)))
        else:
            kstd = numpy.ones_like(r)

        return kstd

    def kfunction_std_inv(self, r, edge_correction=None):
        """
        Compute the inverse of the theoretical standard deviation of the
        empirical k-function of a point pattern like this one, under the CSR
        hypothesis.

        Parameters
        ----------
        r, edge_correction
            See `PointPattern.kfunction_std`.

        Returns
        -------
        array
            Values of the inverse of the standard deviation of the empirical
            K-function, evaulated at `r`.

        """
        return 1.0 / self.kfunction_std(r, edge_correction=edge_correction)

    def lfunction_std(self, r, edge_correction=None):
        """
        Compute the theoretical standard deviation of the empirical L-function
        of a point pattern like this one, under the CSR hypothesis.

        The ``theoretical'' standard deviation is really an empirically
        validated formula, and should be a very good fit to the true standard
        deviation within the interval given by
        `PointPattern.lstatistic_interval`. It is currently only implemented
        for periodic boundary conditions -- an array of ones is returned for
        other edge corrections.

        Parameters
        ----------
        r : array-like
            array of values at which to evaluate the emprical L-function
            standard deviation.
        edge_correction : str {'stationary', 'finite', 'isotropic', 'periodic',
                               'plus'}, optional
            String to select the edge handling to apply in computations. See
            the documentation for `PointPattern` for details.  If None, the
            edge correction falls back to the default value (set at instance
            initialization).

        Returns
        -------
        array
            Values of the standard deviation of the empirical L-function,
            evaulated at `r`.

        """
        if edge_correction is None:
            edge_correction = self.edge_correction

        r = numpy.asarray(r)
        if edge_correction == 'periodic':
            lstd = (self.kfunction_std(r, edge_correction=edge_correction) /
                    (2.0 * _PI * r))
        else:
            lstd = numpy.ones_likes(r)
        return lstd

    def lfunction_std_inv(self, r, edge_correction=None):
        """
        Compute the inverse of the theoretical standard deviation of the
        empirical L-function of a point pattern like this one, under the CSR
        hypothesis.

        Parameters
        ----------
        r, edge_correction
            See `PointPattern.lfunction_std`.

        Returns
        -------
        array
            Values of the inverse of the standard deviation of the empirical
            L-function, evaulated at `r`.

        """
        return 1.0 / self.lfunction_std(r, edge_correction=edge_correction)

    def kstatistic(self, rmin=None, rmax=None, weight_function=None,
                   edge_correction=None):
        """
        Compute the K test statistic for CSR

        The test statstic is defined as max(abs(K(r) - pi * r ** 2)) for
        r-values between some minimum and maximum radii.

        Parameters
        ----------
        rmin : scalar
            The minimum r value to consider when computing the statistic. If
            None, the value is set to 0.0.
        rmin : scalar
            The maximum r value to consider when computing the statistic. If
            None, the value is set by the upper limit from
            `PointPattern.lstatistic_interval`.
        weight_function : callable, optional
            If not None, the offset `K(r) - pi * r ** 2` is weighted by
            `weight_function(r)`. The function should accept one array-like
            argument of r values. A typical example of a relevant weight
            function is `pp.kfunction_std_inv(r)`, where `pp` is the
            `PointPattern` instance for which the K test statistic is computed.
            This weight will compensate for the variation of the variance of
            K(r) for different r.
        edge_correction : str {'stationary', 'finite', 'isotropic', 'periodic',
                               'plus'}, optional
            String to select the edge handling to apply in computations. See
            the documentation for `PointPattern` for details.  If None, the
            edge correction falls back to the default value (set at instance
            initialization).

        Returns
        -------
        scalar
            The K test statistic.

        """
        if edge_correction is None:
            edge_correction = self.edge_correction

        interval = self.kstatistic_interval(edge_correction=edge_correction)
        if rmin is None:
            rmin = interval[0]
        if rmax is None:
            rmax = interval[1]
        if rmin > rmax:
            raise ValueError("'rmin' is smaller than 'rmax'.")

        # The largest deviation between K(r) and r is bound to be at a vertical
        # step. We go manual instead of using self.kfunction, in order to get
        # it as exactly and cheaply as possible.
        rsteps, cweights = self._cumulative_base(
            edge_correction=edge_correction)
        left = numpy.searchsorted(rsteps, rmin, side='right')
        right = numpy.searchsorted(rsteps, rmax, side='left')
        # Include endpoints, and extract cweights for the in-between intervals
        rsteps = numpy.hstack((rmin, rsteps[left:right], rmax))
        cweights = cweights[left - 1:right]

        # Compute the K-values just before and after each step
        imode = self._edge_config[edge_correction]['imode']
        lambda2 = numpy.ones_like(rsteps)
        lambda2[:] = self.squared_intensity(mode=imode, r=rsteps)
        kvals_low = sensibly_divide(cweights, lambda2[:-1])
        kvals_high = sensibly_divide(cweights, lambda2[1:])

        # Compute the offset
        pi_rsteps_sq = _PI * rsteps * rsteps
        offset = numpy.hstack((kvals_low - pi_rsteps_sq[:-1],
                               kvals_high - pi_rsteps_sq[1:]))

        # Weight the offsets by the weight function
        if weight_function is not None:
            weight = weight_function(rsteps)
            weight = numpy.hstack((weight[:-1], weight[1:]))
            offset *= weight
        return numpy.nanmax(numpy.abs(offset))

    def lstatistic(self, rmin=None, rmax=None, weight_function=None,
                   edge_correction=None):
        """
        Compute the L test statistic for CSR

        The test statstic is defined as max(abs(L(r) - r)) for r-values between
        some minimum and maximum radii. Note that if edge_correction ==
        'finite', the power of the L test may depend heavily on the maximum
        r-value and the number of points in the pattern, and the statistic
        computed by this function may not be adequate.

        Parameters
        ----------
        rmin : scalar
            The minimum r value to consider when computing the statistic. If
            None, the value is set by `PointPattern.lstatistic_interval`.
        rmin : scalar
            The maximum r value to consider when computing the statistic. If
            None, the value is set by `PointPattern.lstatistic_interval`.
        weight_function : callable, optional
            If not None, the offset `L(r) - r` is weighted by
            `weight_function(r)`. The function should accept one array-like
            argument of r values. A typical example of a relevant weight
            function is `pp.lfunction_std_inv(r)`, where `pp` is the
            `PointPattern` instance for which the L test statistic is computed.
            This weight will compensate for the variation of the variance of
            L(r) for different r.
        edge_correction : str {'stationary', 'finite', 'isotropic', 'periodic',
                               'plus'}, optional
            String to select the edge handling to apply in computations. See
            the documentation for `PointPattern` for details.  If None, the
            edge correction falls back to the default value (set at instance
            initialization).

        Returns
        -------
        scalar
            The L test statistic.

        """
        if edge_correction is None:
            edge_correction = self.edge_correction

        interval = self.lstatistic_interval(edge_correction=edge_correction)
        if rmin is None:
            rmin = interval[0]
        if rmax is None:
            rmax = interval[1]

        # The largest deviation between L(r) and r is bound to be at a vertical
        # step. We go manual instead of using self.lfunction, in order to get
        # it as exactly and cheaply as possible.
        rsteps, cweights = self._cumulative_base(
            edge_correction=edge_correction)
        valid = numpy.nonzero((rsteps > rmin) & (rsteps < rmax))
        # Include endpoints, and extract cweights for the in-between intervals
        rsteps = numpy.hstack((rmin, rsteps[valid], rmax))
        cweights = numpy.hstack((cweights[valid[0][0] - 1], cweights[valid]))

        # Compute the L-values just before and after each step
        imode = self._edge_config[edge_correction]['imode']
        lambda2 = numpy.ones_like(rsteps)
        lambda2[:] = self.squared_intensity(mode=imode, r=rsteps)
        lvals_low = numpy.sqrt(sensibly_divide(cweights, _PI * lambda2[:-1]))
        lvals_high = numpy.sqrt(sensibly_divide(cweights, _PI * lambda2[1:]))

        # Compute the offset
        offset = numpy.hstack((lvals_high - rsteps[:-1],
                               lvals_low - rsteps[1:]))

        # Weight the offsets by the theoretical standard deviation at the
        # corresponding r values.
        if weight_function is not None:
            weight = weight_function(rsteps)
            weight = numpy.hstack((weight[:-1], weight[1:]))
            offset *= weight
        return numpy.nanmax(numpy.abs(offset))

    @memoize_method
    def ksstatistic(self, variation='fasano'):
        """
        Compute the 2D Kolmogorov-Smirnov test statistic for CSR

        Parameters
        ----------
        variation : {'fasano', 'peacock'}
            Flag to select which definition of the 2D extension of the test
            statistic to use. See Lopes, R., Reid, I., & Hobson, P. (2007). The
            two-dimensional Kolmogorov-Smirnov test. Proceedings of Science.
            Retrieved from http://bura.brunel.ac.uk/handle/2438/1166.

        Returns
        -------
        scalar
            The value of the KS test statistic.

        """
        if variation == 'fasano':
            def piter(points):
                for p in numpy.asarray(points):
                    yield p[0], p[1], True
        elif variation == 'peacock':
            def piter(points):
                parray = numpy.asarray(points)
                for (i, p) in enumerate(parray):
                    for (j, q) in enumerate(parray):
                        yield p[0], q[1], i == j
        else:
            raise ValueError("Unknown 'variation': {}".format(variation))

        tree = self.range_tree(project_points=True)
        points = self.points(project_points=True)
        n = len(points)
        ks = 0.0
        for x, y, ispoint in piter(points):
            for (xmin, xmax) in ((0.0, x), (x, 1.0)):
                for (ymin, ymax) in ((0.0, y), (y, 1.0)):
                    np = len(self.range_tree_query(tree, xmin, xmax,
                                                   ymin, ymax))

                    #rect = geometry.Polygon(((xmin, ymin), (xmax, ymin),
                    #                         (xmax, ymax), (xmin, ymax)))
                    #ps = rect.intersection(points)
                    #if isinstance(ps, geometry.Point):
                    #    np = 1
                    #else:
                    #    np = len(ps)

                    new_ks = numpy.abs(n * (xmax - xmin) * (ymax - ymin) - np)
                    ks = max(ks, new_ks)

                    # If x, y corresponds to an actual point location, the EDF
                    # has a jump here, and we should also check the other
                    # possible value.
                    if ispoint:
                        new_ks = numpy.abs(n * (xmax - xmin) * (ymax - ymin) -
                                           (np - 1))
                        ks = max(ks, new_ks)
        return ks / numpy.sqrt(n)

    def kstatistic_interval(self, edge_correction=None):
        """
        Compute the an appropriate interval over which to evaluate the K test
        statistic for this pattern

        The interval is defined as [rmin, rmax], where rmax is the the same as
        for `PointPattern.lstatistic_interval`, and rmin is a third of the rmin
        from `PointPattern.lstatistic_interval`.

        Parameters
        ----------
        edge_correction : str {'stationary', 'finite', 'isotropic', 'periodic',
                               'plus'}, optional
            String to select the edge handling to apply in computations. See
            the documentation for `PointPattern` for details.  If None, the
            edge correction falls back to the default value (set at instance
            initialization).

        Returns
        -------
        rmin : scalar
            The minimum end of the K test statistic interval
        rmax : scalar
            The maximum end of the K test statistic interval

        """
        if edge_correction is None:
            edge_correction = self.edge_correction

        rmin, rmax = self.lstatistic_interval(edge_correction=edge_correction)
        rmin /= 3.0
        return rmin, rmax

    def lstatistic_interval(self, edge_correction=None):
        """
        Compute the an appropriate interval over which to evaluate the L test
        statistic for this pattern

        The interval is defined as [rmin, rmax], where rmax is the minimum of
        the following two alternatives:
        - the radius of the largest inscribed circle in the window of the point
          pattern, as computed by `Window.inscribed_circle` (if using periodic
          edge correction, the radius of the largest inscribed circle in the
          Voronoi unit cell of the periodic lattice is used instead),
        - the maximum relevant interpoint distance in the point pattern, as
          computed by `PointPattern.rmax`.
        The value of rmin is set to `1.8 / (intensity * sqrt(area))`, where
        `area` is the area of the window of the point pattern, and `intensity`
        is the standard intensity estimate of the point pattern.

        Parameters
        ----------
        edge_correction : str {'stationary', 'finite', 'isotropic', 'periodic',
                               'plus'}, optional
            String to select the edge handling to apply in computations. See
            the documentation for `PointPattern` for details.  If None, the
            edge correction falls back to the default value (set at instance
            initialization).

        Returns
        -------
        rmin : scalar
            The minimum end of the L test statistic interval
        rmax : scalar
            The maximum end of the L test statistic interval

        """
        if edge_correction is None:
            edge_correction = self.edge_correction

        intensity = self.intensity()
        window = self.window
        rmax_absolute = self.rmax(edge_correction=edge_correction)
        if edge_correction == 'periodic':
            rmax_standard = self.window.voronoi().inscribed_circle()['r']
        else:
            rmax_standard = self.window.inscribed_circle()['r']
        rmax = min(rmax_standard, rmax_absolute)
        rmin = 1.8 / (intensity * numpy.sqrt(window.area))
        return rmin, rmax

    @memoize_method
    def _simulate(self, nsims, process, edge_correction):
        """
        Simulate a number of point processes in the same window, and of the
        same intensity, as this pattern

        This part of `PointPattern.simulate` is factored out to optimize
        memoization.

        """
        return PointPatternCollection.from_simulation(
            nsims, self.window, self.intensity(), process=process,
            edge_correction=edge_correction)

    def simulate(self, nsims=100, process='binomial', edge_correction=None):
        """
        Simulate a number of point processes in the same window, and of the
        same intensity, as this pattern

        Parameters
        ----------
        nsims : int, optional
            The number of point patterns to generate.
        process : str {'binomial', 'poisson'}, optional
            String to select the kind of process to simulate.
        edge_correction : str {'stationary', 'finite', 'isotropic', 'periodic',
                               'plus'}, optional
            String to select the default edge handling for the simulated
            patterns. See the documentation for `PointPattern` for details.  If
            None, the default edge correction for this pattern is used.

        Returns
        -------
        PointPatternCollection
            Collection of the simulated patterns.

        """
        if edge_correction is None:
            edge_correction = self.edge_correction
        return self._simulate(nsims, process, edge_correction)

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
            the documentation for `PointPattern` for details.  If None, the
            edge correction falls back to the default value (set at instance
            initialization).
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
            edge_correction = self.edge_correction

        rvals = self.rvals(edge_correction=edge_correction)
        kvals = self.kfunction(rvals, edge_correction=edge_correction)

        lines = axes.plot(rvals, kvals, linewidth=linewidth, **kwargs)

        if csr:
            if csr_kw is None:
                csr_kw = {}

            kcsr = _PI * rvals * rvals
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
            the documentation for `PointPattern` for details.  If None, the
            edge correction falls back to the default value (set at instance
            initialization).
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
            edge_correction = self.edge_correction

        rvals = self.rvals(edge_correction=edge_correction)
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
            the documentation for `PointPattern` for details.  If None, the
            edge correction falls back to the default value (set at instance
            initialization).
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
            edge_correction = self.edge_correction

        rmax = self.rmax(edge_correction=edge_correction)
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

    def plot_pattern(self, axes=None, marker='o', periodic_levels=0,
                     plus=False, window=False, periodic_kw=None, plus_kw=None,
                     window_kw=None, **kwargs):
        """
        Plot point pattern

        The point pattern can be added to an existing plot via the optional
        'axes' argument.

        :axes: Axes instance to add the point pattern to. If None (default),
               the current Axes instance with equal aspect ratio is used if
               any, or a new one created.
        :marker: a valid matplotlib marker specification. Defaults to 'o'
        periodic_levels : integer, optional
            Add this many levels of periodic extensions of the point pattern to
            the plot. See `PointPattern.periodic_extension` for further
            explanation.
        :plus: if True, add plus sampling points to the plot.
        :window: if True, the window boundaries are added to the plot.
        :periodic_kw: dict of keyword arguments to pass to the axes.scatter()
                      method used to plot the periodic extension. Default: None
                      (empty dict)
        :plus_kw: dict of keyword arguments to pass to the axes.scatter()
                  method used to plot the plus sampling points. Default: None
                  (empty dict)
        :window_kw: dict of keyword arguments to pass to the Window.plot()
                    method. Default: None (empty dict)
        :kwargs: additional keyword arguments passed on to axes.scatter()
                 method used to plot the point pattern. Note especially the
                 keywords 'c' (colors), 's' (marker sizes) and 'label'.
        :returns: list of the artists added to the plot:
                  a matplotlib.collections.PathCollection instance for the
                  point pattern, and optionally another
                  matplotlib.collections.PathCollection instance for each of
                  the periodic extension and the plus sampling points, and
                  finally a a matplotlib.patches.Polygon instance for the
                  window.

        """
        if axes is None:
            axes = pyplot.gca(aspect='equal')
            cent = self.window.centroid
            diag = self.window.longest_diagonal()
            axes.set(xlim=(cent.x - diag, cent.x + diag),
                     ylim=(cent.y - diag, cent.y + diag))

        pp = numpy.asarray(self._points)
        h = [axes.scatter(pp[:, 0], pp[:, 1], marker=marker, **kwargs)]

        if periodic_levels > 0:
            if periodic_kw is None:
                periodic_kw = {}
            pp = numpy.asarray(self.periodic_extension(periodic_levels))
            h.append(axes.scatter(pp[:, 0], pp[:, 1], marker=marker,
                                  **periodic_kw))

        if plus:
            if plus_kw is None:
                plus_kw = {}
            pp = numpy.asarray(self.pluspoints)
            h.append(axes.scatter(pp[:, 0], pp[:, 1], marker=marker,
                                  **plus_kw))

        if window:
            if window_kw is None:
                window_kw = {}
            wpatch = self.window.plot(axes=axes, **window_kw)
            h.append(wpatch)

        return h


class PointPatternCollection(AlmostImmutable, Sequence):
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
        self.edge_correction = edge_correction

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
        area_factor = (xmax - xmin) * (ymax - ymin) / window.area
        nmean = intensity * window.area

        if process == 'poisson':
            nlist = numpy.random.poisson(nmean, nsims)
        elif process == 'binomial':
            nlist = numpy.empty((nsims,), dtype=numpy.int_)
            nlist.fill(round(nmean))
        else:
            raise ValueError("unknown point process: {}".format(process))

        patterns = []
        for n in nlist:
            points = []
            left = n
            while left > 0:
                ndraw = int(area_factor * left)
                draw = numpy.column_stack(
                    (numpy.random.uniform(low=xmin, high=xmax, size=ndraw),
                     numpy.random.uniform(low=ymin, high=ymax, size=ndraw)))
                new_points = geometry.MultiPoint(draw).intersection(window)
                if isinstance(new_points, geometry.Point):
                    points.append(new_points)
                    left -= 1
                else:
                    points.extend(new_points)
                    left -= len(new_points)

            pp = PointPattern(points[:n], window,
                              edge_correction=edge_correction)
            patterns.append(pp)

        return cls(patterns, edge_correction=edge_correction)

    # Implement abstract methods
    def __getitem__(self, index, *args, **kwargs):
        return self.patterns.__getitem__(index, *args, **kwargs)

    def __len__(self, *args, **kwargs):
        return self.patterns.__len__(*args, **kwargs)

    # Override certain possibly very slow mixins
    def __iter__(self, *args, **kwargs):
        return self.patterns.__iter__(*args, **kwargs)

    def __reversed__(self, *args, **kwargs):
        return self.patterns.__reversed__(*args, **kwargs)

    def index(self, *args, **kwargs):
        return self.patterns.index(*args, **kwargs)

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
    #                        edge_correction = self.edge_correction
    #                    return pandas.Series(
    #                        [ppattr(pp, edge_correction=edge_correction)
    #                         for pp in self.patterns])
    #                return aggregate
    #        raise e

    @property
    def npoints(self):
        """
        The total number of points in the whole collection

        """
        return sum(len(pp) for pp in self.patterns)

    def nweights(self):
        """
        List of the fractions of the total number of points in the collection
        coming from each of the patterns

        """
        npoints = self.npoints
        return [len(pp) / npoints for pp in self.patterns]

    def aweights(self):
        """
        List of the fraction of the total window area in the collection coming
        from to the window of each of the patterns

        """
        total_area = sum(pp.window.area for pp in self.patterns)
        return [pp.window.area / total_area for pp in self.patterns]

    def rmax(self, edge_correction=None):
        """
        Compute the maximum r-value where the K-functions of all patterns in
        the collection are defined

        edge_correction : str {'stationary', 'finite', 'isotropic', 'periodic',
                               'plus'}, optional
            String to select the edge handling to apply in computations. See
            the documentation for `PointPattern` for details.  If None, the
            edge correction falls back to the default value (set at instance
            initialization).

        Returns
        -------
        scalar
            Maximum valid r-value.

        """
        if edge_correction is None:
            edge_correction = self.edge_correction
        return min(pp.rmax(edge_correction=edge_correction)
                   for pp in self.patterns)

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

        intensities = [pp.intensity(mode=mode, r=r) for pp in self.patterns]

        return sum(aw * intensity
                   for (aw, intensity) in zip(self.aweights(), intensities))

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
            the documentation for `PointPattern` for details.  If None, the
            edge correction falls back to the default value (set at instance
            initialization).

        Returns
        -------
        array
            Values of the empirical aggregate K-function evaluated at `r`.

        """
        if edge_correction is None:
            edge_correction = self.edge_correction

        kvalues = [pp.kfunction(r, edge_correction=edge_correction)
                   for pp in self.patterns]

        return sum(nw * kv for (nw, kv) in zip(self.nweights(), kvalues))

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
            the documentation for `PointPattern` for details.  If None, the
            edge correction falls back to the default value (set at instance
            initialization).

        Returns
        -------
        array
            Values of the empirical aggregate L-function evaluated at `r`.

        """
        if edge_correction is None:
            edge_correction = self.edge_correction

        return numpy.sqrt(self.aggregate_kfunction(
            r, edge_correction=edge_correction) / _PI)

    def _pp_attr_r_frame(self, attr, r, edge_correction, **kwargs):
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
            the documentation for `PointPattern` for details.  If None, the
            edge correction falls back to the default value (set at instance
            initialization).
        **kwargs : dict, optional
            Other arguments to pass to the `PointPattern` attribute.

        Returns
        -------
        DataFrame
            DataFrame where each row contains values of the
            `PointPattern` attribute from one pattern, evaluated at `r`.

        """
        if edge_correction is None:
            edge_correction = self.edge_correction

        return pandas.DataFrame(
            [getattr(pp, attr)(r, edge_correction=edge_correction, **kwargs)
             for pp in self.patterns])

    def _pp_attr_r_critical(self, attr, alpha, r, edge_correction, **kwargs):
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
            the documentation for `PointPattern` for details.  If None, the
            edge correction falls back to the default value (set at instance
            initialization).
        **kwargs : dict, optional
            Other arguments to pass to the `PointPattern` attribute.

        Returns
        -------
        array
            Critical values of the `PointPattern` attribute evaluated at `r`.

        """
        if edge_correction is None:
            edge_correction = self.edge_correction

        attr_frame = self._pp_attr_r_frame(attr, r,
                                           edge_correction=edge_correction,
                                           **kwargs)
        return attr_frame.quantile(q=alpha, axis=0)

    def _pp_attr_r_mean(self, attr, r, edge_correction, **kwargs):
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
            the documentation for `PointPattern` for details.  If None, the
            edge correction falls back to the default value (set at instance
            initialization).

        Returns
        -------
        array
            Mean of the `PointPattern` attribute evaluated at `r`.

        """
        if edge_correction is None:
            edge_correction = self.edge_correction

        attr_frame = self._pp_attr_r_frame(attr, r,
                                           edge_correction=edge_correction,
                                           **kwargs)
        return attr_frame.mean(axis=0, skipna=True)

    def _pp_attr_r_var(self, attr, r, edge_correction, **kwargs):
        """
        Compute the variance of some PointPattern attribute

        Parameters
        ----------
        attr : string
            name of `pointpattern` attribute to use.
        r : array-like
            Array of values at which to evaluate the variance.
        edge_correction : str {'stationary', 'finite', 'isotropic', 'periodic',
                               'plus'}, optional
            String to select the edge handling to apply in computations. See
            the documentation for `PointPattern` for details.  If None, the
            edge correction falls back to the default value (set at instance
            initialization).

        Returns
        -------
        array
            Variance of the `PointPattern` attribute evaluated at `r`.

        """
        if edge_correction is None:
            edge_correction = self.edge_correction

        attr_frame = self._pp_attr_r_frame(attr, r,
                                           edge_correction=edge_correction,
                                           **kwargs)
        return attr_frame.var(axis=0, skipna=True)

    def _pp_attr_r_std(self, attr, r, edge_correction, **kwargs):
        """
        Compute the standard deviation of some PointPattern attribute

        Parameters
        ----------
        attr : string
            name of `pointpattern` attribute to use.
        r : array-like
            Array of values at which to evaluate the standard deviation.
        edge_correction : str {'stationary', 'finite', 'isotropic', 'periodic',
                               'plus'}, optional
            String to select the edge handling to apply in computations. See
            the documentation for `PointPattern` for details.  If None, the
            edge correction falls back to the default value (set at instance
            initialization).

        Returns
        -------
        array
            Standard deviation of the `PointPattern` attribute evaluated at
            `r`.

        """
        if edge_correction is None:
            edge_correction = self.edge_correction

        attr_frame = self._pp_attr_r_frame(attr, r,
                                           edge_correction=edge_correction,
                                           **kwargs)
        return attr_frame.std(axis=0, skipna=True)

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
            the documentation for `PointPattern` for details.  If None, the
            edge correction falls back to the default value (set at instance
            initialization).

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
            the documentation for `PointPattern` for details.  If None, the
            edge correction falls back to the default value (set at instance
            initialization).

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
            the documentation for `PointPattern` for details.  If None, the
            edge correction falls back to the default value (set at instance
            initialization).

        Returns
        -------
        array
            Mean of the empirical K-functions evaluated at `r`.

        """
        return self._pp_attr_r_mean('kfunction', r,
                                    edge_correction=edge_correction)

    def kvar(self, r, edge_correction=None):
        """
        Compute the variance of the empirical K-functions of the patterns

        Parameters
        ----------
        r : array-like
            Array of values at which to evaluate the variance of the empirical
            K-functions.
        edge_correction : str {'stationary', 'finite', 'isotropic', 'periodic',
                               'plus'}, optional
            String to select the edge handling to apply in computations. See
            the documentation for `PointPattern` for details.  If None, the
            edge correction falls back to the default value (set at instance
            initialization).

        Returns
        -------
        array
            Variance of the empirical K-functions evaluated at `r`.

        """
        return self._pp_attr_r_var('kfunction', r,
                                   edge_correction=edge_correction)

    def kstd(self, r, edge_correction=None):
        """
        Compute the standard devation of the empirical K-functions of the
        patterns

        Parameters
        ----------
        r : array-like
            Array of values at which to evaluate the standard deviation of the
            empirical K-functions.
        edge_correction : str {'stationary', 'finite', 'isotropic', 'periodic',
                               'plus'}, optional
            String to select the edge handling to apply in computations. See
            the documentation for `PointPattern` for details.  If None, the
            edge correction falls back to the default value (set at instance
            initialization).

        Returns
        -------
        array
            Standard deviation of the empirical K-functions evaluated at `r`.

        """
        return self._pp_attr_r_std('kfunction', r,
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
            the documentation for `PointPattern` for details.  If None, the
            edge correction falls back to the default value (set at instance
            initialization).

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
            the documentation for `PointPattern` for details.  If None, the
            edge correction falls back to the default value (set at instance
            initialization).

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
            the documentation for `PointPattern` for details.  If None, the
            edge correction falls back to the default value (set at instance
            initialization).

        Returns
        -------
        array
            Mean of the empirical L-functions evaluated at `r`.

        """
        return self._pp_attr_r_mean('lfunction', r,
                                    edge_correction=edge_correction)

    def lvar(self, r, edge_correction=None):
        """
        Compute the variance of the empirical L-functions of the patterns

        Parameters
        ----------
        r : array-like
            Array of values at which to evaluate the mean values of the
            empirical L-functions.
        edge_correction : str {'stationary', 'finite', 'isotropic', 'periodic',
                               'plus'}, optional
            String to select the edge handling to apply in computations. See
            the documentation for `PointPattern` for details.  If None, the
            edge correction falls back to the default value (set at instance
            initialization).

        Returns
        -------
        array
            Variance of the empirical L-functions evaluated at `r`.

        """
        return self._pp_attr_r_var('lfunction', r,
                                   edge_correction=edge_correction)

    def lstd(self, r, edge_correction=None):
        """
        Compute the standard deviation of the empirical L-functions of the
        patterns

        Parameters
        ----------
        r : array-like
            Array of values at which to evaluate the mean values of the
            empirical L-functions.
        edge_correction : str {'stationary', 'finite', 'isotropic', 'periodic',
                               'plus'}, optional
            String to select the edge handling to apply in computations. See
            the documentation for `PointPattern` for details.  If None, the
            edge correction falls back to the default value (set at instance
            initialization).

        Returns
        -------
        array
            Standard deviation of the empirical L-functions evaluated at `r`.

        """
        return self._pp_attr_r_std('lfunction', r,
                                   edge_correction=edge_correction)

    def pair_corr_frame(self, r, edge_correction=None, **kwargs):
        """
        Compute a DataFrame containing values of the empirical pair correlation
        functions of the patterns

        Parameters
        ----------
        r : array-like
            Array of values at which to evaluate the functions.
        edge_correction : str {'stationary', 'finite', 'isotropic', 'periodic',
                               'plus'}, optional
            String to select the edge handling to apply in computations. See
            the documentation for `PointPattern` for details.  If None, the
            edge correction falls back to the default value (set at instance
            initialization).
        **kwargs : dict, optional
            Additional keyword arguments are passed to
            `PointPattern.pair_corr_function`.

        Returns
        -------
        DataFrame
            DataFrame where each row contains values of the empirical
            pair correlation function from one pattern, evaluated at `r`.

        """
        return self._pp_attr_r_frame('pair_corr_function', r,
                                     edge_correction=edge_correction, **kwargs)

    def pair_corr_critical(self, alpha, r, edge_correction=None, **kwargs):
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
        edge_correction : str {'stationary', 'finite', 'isotropic', 'periodic',
                               'plus'}, optional
            String to select the edge handling to apply in computations. See
            the documentation for `PointPattern` for details.  If None, the
            edge correction falls back to the default value (set at instance
            initialization).
        **kwargs : dict, optional
            Additional keyword arguments are passed to
            `PointPattern.pair_corr_function`.

        Returns
        -------
        array
            Critical values of the pair correlation functions evaluated at `r`.

        """
        return self._pp_attr_r_critical('pair_corr_function', alpha, r,
                                        edge_correction=edge_correction,
                                        **kwargs)

    def pair_corr_mean(self, r, edge_correction=None, **kwargs):
        """
        Compute the mean of the pair correlation functions of the patterns

        Parameters
        ----------
        r : array-like
            Array of values at which to evaluate the mean values of the
            functions.
        edge_correction : str {'stationary', 'finite', 'isotropic', 'periodic',
                               'plus'}, optional
            String to select the edge handling to apply in computations. See
            the documentation for `PointPattern` for details.  If None, the
            edge correction falls back to the default value (set at instance
            initialization).
        **kwargs : dict, optional
            Additional keyword arguments are passed to
            `PointPattern.pair_corr_function`.

        Returns
        -------
        array
            Mean of the empirical pair correlation functions evaluated at `r`.

        """
        return self._pp_attr_r_mean('pair_corr_function', r,
                                    edge_correction=edge_correction, **kwargs)

    def _pp_attr_series(self, attr, **kwargs):
        """
        Compute a Series containing values of some scalar PointPattern
        attribute.

        Parameters
        ----------
        attr : string
            Name of `PointPattern` attribute to use.
        **kwargs : dict, optional
            Other arguments to pass to the `PointPattern` attribute.

        Returns
        -------
        Series
            Series containing values of the `PointPattern` attribute from each
            pattern.

        """
        return pandas.Series(
            [getattr(pp, attr)(**kwargs) for pp in self.patterns])

    def _pp_attr_test(self, attr, pattern, **kwargs):
        """
        Perform a statistical test on a PointPattern, based on the distribution
        of a PointPattern attribute over the patterns in this collection.

        Parameters
        ----------
        attr : string
            Name of `PointPattern` attribute to use.
        pattern : PointPattern
            PointPattern to perform the test on.
        **kwargs : dict, optional
            Other arguments to pass to the `PointPattern` attribute.

        Returns
        -------
        scalar
            The p-value computed using the the selected attribute as the test
            statistic for `pattern`, and its distribution over this collection
            as the null distribution.

        """
        tsdist = self._pp_attr_series(attr, **kwargs).dropna()
        teststat = getattr(pattern, attr)(**kwargs)
        return 1.0 - 0.01 * percentileofscore(tsdist, teststat, kind='mean')

    def lstatistics(self, edge_correction=None, **kwargs):
        """
        Compute the L test statistic for CSR for each pattern in the collection

        See `PointPattern.lstatistic` for details about the L test statistic.

        Parameters
        ----------
        edge_correction : str {'stationary', 'finite', 'isotropic', 'periodic',
                               'plus'}, optional
            String to select the edge handling to apply in computations. See
            the documentation for `PointPattern` for details.  If None, the
            edge correction falls back to the default value (set at instance
            initialization).
        **kwargs : dict, optional
            Additional keyword arguments are passed to
            `PointPattern.lstatstic`.


        Returns
        -------
        Series
            Series containing the L test statistic for each pattern in the
            collection.

        """
        if edge_correction is None:
            edge_correction = self.edge_correction
        return self._pp_attr_series('lstatistic',
                                    edge_correction=edge_correction, **kwargs)

    def kstatistics(self, edge_correction=None, **kwargs):
        """
        Compute the K test statistic for CSR for each pattern in the collection

        See `PointPattern.kstatistic` for details about the K test statistic.

        Parameters
        ----------
        edge_correction : str {'stationary', 'finite', 'isotropic', 'periodic',
                               'plus'}, optional
            String to select the edge handling to apply in computations. See
            the documentation for `PointPattern` for details.  If None, the
            edge correction falls back to the default value (set at instance
            initialization).
        **kwargs : dict, optional
            Additional keyword arguments are passed to
            `PointPattern.kstatstic`.


        Returns
        -------
        Series
            Series containing the K test statistic for each pattern in the
            collection.

        """
        if edge_correction is None:
            edge_correction = self.edge_correction
        return self._pp_attr_series('kstatistic',
                                    edge_correction=edge_correction, **kwargs)

    def ksstatistics(self, **kwargs):
        """
        Compute the Kolmogorov-Smirnov test statistic for CSR for each pattern
        in the collection

        See `PointPattern.ksstatistic` for details about the Kolmogorov-Smirnov
        test statistic.

        Parameters
        ----------
        edge_correction : str {'stationary', 'finite', 'isotropic', 'periodic',
                               'plus'}, optional
            String to select the edge handling to apply in computations. See
            the documentation for `PointPattern` for details.  If None, the
            edge correction falls back to the default value (set at instance
            initialization).
        **kwargs : dict, optional
            Additional keyword arguments are passed to
            `PointPattern.ksstatstic`.


        Returns
        -------
        Series
            Series containing the Kolmogorov-Smirnov test statistic for each
            pattern in the collection.

        """
        return self._pp_attr_series('ksstatistic', **kwargs)

    def ltest(self, pattern, edge_correction=None, **kwargs):
        """
        Perform an L test for CSR on a PointPattern, based on the distribution
        of L test statictics from the patterns in this collection.

        Parameters
        ----------
        pattern : PointPattern
            PointPattern to perform the test on.
        edge_correction : str {'stationary', 'finite', 'isotropic', 'periodic',
                               'plus'}, optional
            String to select the edge handling to apply in computations. See
            the documentation for `PointPattern` for details.  If None, the
            edge correction falls back to the default value (set at instance
            initialization).
        **kwargs : dict, optional
            Additional keyword arguments are passed to
            `PointPattern.lstatstic`.

        Returns
        -------
        scalar
            The p-value of the L test statistic for `pattern`.

        """
        if edge_correction is None:
            edge_correction = self.edge_correction
        return self._pp_attr_test('lstatistic', pattern,
                                  edge_correction=edge_correction, **kwargs)

    def ktest(self, pattern, edge_correction=None, **kwargs):
        """
        Perform a K test for CSR on a PointPattern, based on the distribution
        of K test statictics from the patterns in this collection.

        Parameters
        ----------
        pattern : PointPattern
            PointPattern to perform the test on.
        edge_correction : str {'stationary', 'finite', 'isotropic', 'periodic',
                               'plus'}, optional
            String to select the edge handling to apply in computations. See
            the documentation for `PointPattern` for details.  If None, the
            edge correction falls back to the default value (set at instance
            initialization).
        **kwargs : dict, optional
            Additional keyword arguments are passed to
            `PointPattern.kstatstic`.

        Returns
        -------
        scalar
            The p-value of the K test statistic for `pattern`.

        """
        if edge_correction is None:
            edge_correction = self.edge_correction
        return self._pp_attr_test('kstatistic', pattern,
                                  edge_correction=edge_correction, **kwargs)

    def kstest(self, pattern, **kwargs):
        """
        Perform a Kolmogorov-Smirnov test for CSR on a PointPattern, based on
        the distribution of Kolmogorov-Smirnov test statictics from the
        patterns in this collection.

        Parameters
        ----------
        pattern : PointPattern
            PointPattern to perform the test on.
        **kwargs : dict, optional
            Additional keyword arguments are passed to
            `PointPattern.ksstatstic`.

        Returns
        -------
        scalar
            The p-value of the Komogorov-Smirnov test statistic for `pattern`.

        """
        return self._pp_attr_test('ksstatistic', pattern, **kwargs)

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
            the documentation for `PointPattern` for details.  If None, the
            edge correction falls back to the default value (set at instance
            initialization).
        **kwargs : dict, optional
            Additional keyword arguments are passed to `numpy.histogram`.

        Returns
        -------
        See `numpy.histogram`.

        """
        if edge_correction is None:
            edge_correction = self.edge_correction

        plural_attr = attribute + 's'
        if hasattr(self, plural_attr):
            vals = getattr(self, plural_attr)(edge_correction=edge_correction)
        else:
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
            the documentation for `PointPattern` for details.  If None, the
            edge correction falls back to the default value (set at instance
            initialization).
        low : scalar between 0.0 and `high`, optional
            Quantile defining the lower edge of the envelope.
        high : scalar between `low` and 1.0, optional
            Quantile defining the higher edge of the envelope.
        alpha : scalar between 0.0 and 1.0, optional
            The opacity of the envelope fill.
        **kwargs : dict, optional
            Additional keyword arguments are passed to `axes.fill_between`.
            Note in particular the keywords 'edgecolor', 'facecolor' and
            'label'.

        Returns
        -------
        PolyCollection
            The PolyCollection instance filling the envelope.

        """
        if axes is None:
            axes = pyplot.gca()

        if edge_correction is None:
            edge_correction = self.edge_correction

        rvals = numpy.linspace(0.0, self.rmax(edge_correction=edge_correction),
                               RSAMPLES)
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
            the documentation for `PointPattern` for details.  If None, the
            edge correction falls back to the default value (set at instance
            initialization).
        **kwargs : dict, optional
            Additional keyword arguments are passed to `axes.plot`. Note in
            particular the keywords 'linestyle', 'color', and 'label'.

        Returns
        -------
        list
            List containing the Line2D of the plotted mean.

        """
        if axes is None:
            axes = pyplot.gca()

        if edge_correction is None:
            edge_correction = self.edge_correction

        rvals = numpy.linspace(0.0, self.rmax(edge_correction=edge_correction),
                               RSAMPLES)
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
            the documentation for `PointPattern` for details.  If None, the
            edge correction falls back to the default value (set at instance
            initialization).
        low : scalar between 0.0 and `high`, optional
            Quantile defining the lower edge of the envelope.
        high : scalar between `low` and 1.0, optional
            Quantile defining the higher edge of the envelope.
        alpha : scalar between 0.0 and 1.0, optional
            The opacity of the envelope fill.
        **kwargs : dict, optional
            Additional keyword arguments are passed to `axes.fill_between`.
            Note in particular the keywords 'edgecolor', 'facecolor' and
            'label'.

        Returns
        -------
        PolyCollection
            The PolyCollection instance filling the envelope.

        """
        if axes is None:
            axes = pyplot.gca()

        if edge_correction is None:
            edge_correction = self.edge_correction

        rvals = numpy.linspace(0.0, self.rmax(edge_correction=edge_correction),
                               RSAMPLES)
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
            the documentation for `PointPattern` for details.  If None, the
            edge correction falls back to the default value (set at instance
            initialization).
        **kwargs : dict, optional
            Additional keyword arguments are passed to `axes.plot`. Note in
            particular the keywords 'linestyle', 'color', and 'label'.

        Returns
        -------
        list
            List containing the Line2D of the plotted mean.

        """
        if axes is None:
            axes = pyplot.gca()

        if edge_correction is None:
            edge_correction = self.edge_correction

        rvals = numpy.linspace(0.0, self.rmax(edge_correction=edge_correction),
                               RSAMPLES)
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
            the documentation for `PointPattern` for details.  If None, the
            edge correction falls back to the default value (set at instance
            initialization).
        low : scalar between 0.0 and `high`, optional
            Quantile defining the lower edge of the envelope.
        high : scalar between `low` and 1.0, optional
            Quantile defining the higher edge of the envelope.
        alpha : scalar between 0.0 and 1.0, optional
            The opacity of the envelope fill.
        **kwargs : dict, optional
            Additional keyword arguments are passed to `axes.fill_between`.
            Note in particular the keywords 'edgecolor', 'facecolor' and
            'label'.

        Returns
        -------
        PolyCollection
            The PolyCollection instance filling the envelope.

        """
        if axes is None:
            axes = pyplot.gca()

        if edge_correction is None:
            edge_correction = self.edge_correction

        rvals = numpy.linspace(0.0, self.rmax(edge_correction=edge_correction),
                               RSAMPLES)
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
            the documentation for `PointPattern` for details.  If None, the
            edge correction falls back to the default value (set at instance
            initialization).
        **kwargs : dict, optional
            Additional keyword arguments are passed to `axes.plot`. Note in
            particular the keywords 'linestyle', 'color', and 'label'.

        Returns
        -------
        list
            List containing the Line2D of the plotted mean.

        """
        if axes is None:
            axes = pyplot.gca()

        if edge_correction is None:
            edge_correction = self.edge_correction

        rvals = numpy.linspace(0.0, self.rmax(edge_correction=edge_correction),
                               RSAMPLES)
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
            the documentation for `PointPattern` for details.  If None, the
            edge correction falls back to the default value (set at instance
            initialization).
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
            Additional keyword arguments are passed to `axes.plot`. Note in
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
            edge_correction = self.edge_correction

        rvals = numpy.linspace(0.0, self.rmax(edge_correction=edge_correction),
                               RSAMPLES)
        kvals = self.aggregate_kfunction(
            rvals, edge_correction=edge_correction)

        lines = axes.plot(rvals, kvals, linewidth=linewidth, **kwargs)

        if csr:
            if csr_kw is None:
                csr_kw = {}

            kcsr = _PI * rvals * rvals
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
            the documentation for `PointPattern` for details.  If None, the
            edge correction falls back to the default value (set at instance
            initialization).
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
            Additional keyword arguments are passed to `axes.plot`. Note in
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
            edge_correction = self.edge_correction

        rvals = numpy.linspace(0.0, self.rmax(edge_correction=edge_correction),
                               RSAMPLES)
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
            the documentation for `PointPattern` for details.  If None, the
            edge correction falls back to the default value (set at instance
            initialization).
        histtype : {'bar', 'step', 'stepfilled'}, optional
            The type of histogram to draw. See the documentation for
            `pyplot.hist` for details. Note that 'barstacked' is not a relevant
            option in this case, since a `PointPatternCollection` only provides
            a single set of data.
        **kwargs : dict, optional
            Additional keyword arguments are passed to `pyplot.hist`.

        Returns
        -------
        list
            List of matplotlib patches used to create the histogram.

        """
        if axes is None:
            axes = pyplot.gca()

        if edge_correction is None:
            edge_correction = self.edge_correction

        plural_attr = attribute + 's'
        if hasattr(self, plural_attr):
            vals = getattr(self, plural_attr)(edge_correction=edge_correction)
        else:
            vals = numpy.array(
                [getattr(pp, attribute)(edge_correction=edge_correction)
                 for pp in self.patterns])
        return axes.hist(vals, histtype=histtype, **kwargs)[2]
