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
        memcache = memoize_method.CACHE_NAME
        red = list(geometry.Polygon.__reduce__(self))
        red[2] = {'state': red[2],
                  memcache: getattr(self, memcache)}
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

    def rmax(self, edge_correction):
        """
        Compute the maximum relevant interpoint distance for a particular edge
        correction

        :edge_correction: flag to select the handling of edges. Possible
                          values:
            'finite': Translational edge correction used.
            'periodic': No edge correction used. Care is taken to avoid
                        counting the same pair multiple times, under the
                        assumption that points are repeated periodically in the
                        pattern according to self.lattice.
            'plus': No edge correction used.
            'stationary': Translational edge correction used.
            'isotropic': Rotational edge correction used.
        :returns: maximum relevant interpoint distance

        """
        if edge_correction in ('finite', 'plus', 'isotropic'):
            return self.longest_diagonal

        elif edge_correction == 'periodic':
            return 0.5 * self.longest_diagonal

        elif edge_correction == 'stationary':
            return 2.0 * self.inscribed_circle['radius']

        else:
            raise ValueError("unknown edge correction: {}"
                             .format(edge_correction))

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

    def weights(self, mp1, mp2, edge_correction):
        """
        Compute the weights that a set of point pairs in the window contribute
        in the estimation of second-order summary characteristics

        :mp1, mp2: MultiPoint instances containing points to pair up
        :edge_correction: flag to select the handling of edges. Possible
                          values:
            'finite': Translational edge correction used.
            'periodic': No edge correction used, but 0.0 returned if a shorter
                        distance between the points exists under, periodic
                        boundary conditions.
            'plus': No edge correction used.
            'stationary': Translational edge correction used.
            'isotropic': Rotational edge correction used.
        :returns: numpy array containing the weight of mp1[i] and mp2[j] in
                  element [i, j]

        """
        if edge_correction in ('finite', 'stationary'):
            diff = PointPattern.pairwise_vectors(mp1, mp2)
            return 1.0 / self.set_covariance(diff[:, :, 0], diff[:, :, 1])

        elif edge_correction == 'periodic':

            m, n = len(mp1), len(mp2)
            w = numpy.zeros((m, n))

            if n < m:
                mp1, mp2 = mp2, mp1
                wview = w.transpose()
            else:
                wview = w

            mp2_arr = numpy.array(mp2)

            z = (0.0, 0.0)
            zero = numpy.array(z)
            #origin = geometry.Point(z)
            #distances = PointPattern.pairwise_distances(mp1, mp2)
            for (i, p1) in enumerate(mp1):
                translated_window = affinity.translate(self, xoff=p1.x,
                                                       yoff=p1.y)
                valid_mp2 = mp2.intersection(translated_window)
                vmp2_arr = numpy.asarray(valid_mp2)
                vindex = numpy.any(numpy.all(
                    PointPattern.pairwise_vectors(mp2_arr, vmp2_arr) == zero,
                    axis=-1), axis=-1)

                wview[i, vindex] = 1.0 / self.area

                ## Isotropic edge correction (to cancel corner effects that are
                ## still present for large r)
                #for j in numpy.nonzero(vindex)[0]:
                #    r = distances[i, j]
                #    ring = origin.buffer(r).boundary
                #    wview[i, j] *= (2.0 * numpy.pi * r /
                #                    ring.intersection(self).length)

            return w

        elif edge_correction == 'plus':
            m, n = len(mp1), len(mp2)
            w = numpy.empty((m, n))
            w.fill(1.0 / self.area)
            return w

        elif edge_correction == 'isotropic':
            m, n = len(mp1), len(mp2)
            w = numpy.zeros((m, n))

            distances = PointPattern.pairwise_distances(mp1, mp2)

            origin = geometry.Point((0.0, 0.0))
            for (i, p1) in enumerate(mp1):
                for j in range(n):
                    r = distances[i, j]
                    ring = p1.buffer(r).boundary
                    rball = origin.buffer(r)
                    doughnut = self.difference(self.erode_by_this(rball))
                    w[i, j] = 2.0 * numpy.pi * r / (
                        self.intersection(ring).length * doughnut.area)
            return w

        else:
            raise ValueError("unknown edge correction: {}"
                             .format(edge_correction))

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


class PointPattern(geometry.MultiPoint):
    """
    Represent a planar point pattern and its associated window, and provide
    methods for analyzing its statistical properties

    """

    def __init__(self, points=None, window=None, pluspoints=None, **kwargs):
        """
        Initialize the PointPattern instance

        :points: a MultiPoint instance, or a valid argument to the MultiPoint
                 constructor (e.g. a list of coordinate tuples), representing
                 the points in the point pattern
        :window: a Polygon instance, or a valid argument to the Polygon
                 constructor (e.g. a list of coordinate tuples), giving the
                 set within which the point pattern is defined
        :pluspoints: an optional MultiPoint instance or valid argument to the
                     MultiPoint constructor, representing a set of extra points
                     (usually outside the window) to use for plus sampling
        :kwargs: additional keyword arguments to the MultiPoint constructor

        """
        geometry.MultiPoint.__init__(self, points, **kwargs)

        # Avoid copying the window unless needed
        if not isinstance(window, Window):
            window = Window(window)

        self.window = Window(window)
        self.pluspoints = geometry.MultiPoint(pluspoints)

    def __reduce__(self):
        memcache = memoize_method.CACHE_NAME
        red = list(geometry.MultiPoint.__reduce__(self))
        red[2] = {'state': red[2],
                  'window': self.window,
                  'pluspoints': self.pluspoints,
                  memcache: getattr(self, memcache)}
        return tuple(red)

    def __setstate__(self, state):
        geometry.MultiPoint.__setstate__(self, state.pop('state'))
        for key in state:
            setattr(self, key, state[key])

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

    @memoize_method
    def points(self, mode='standard'):
        """
        Return all the relevant points related to the pattern, under some
        definition of "relevant".

        :mode: string to select the relevant points. Possible values:
            'standard': the points in the pattern are returned
            'periodic': the union of the pattern and its periodic extension as
                        defined by self.periodic_extension is returned
            'plus': the union of the pattern and the plus sampling points is
                    returned
        :returns: multipoint instance containing all the points

        """
        if mode == 'standard':
            return geometry.MultiPoint(self)
        elif mode == 'periodic':
            return self.periodic_extension.union(self)
        elif mode == 'plus':
            return self.pluspoints.union(self)
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
                intensity = len(self.intersection(ew)) / ew.area
            else:
                intensity = numpy.zeros_like(r)
                for (i, rval) in r_enum:
                    ew = window.buffer(-rval)
                    intensity[i] = len(self.intersection(ew)) / ew.area
        elif mode == 'neighbor':
            intensity = 0.0
            for p in self:
                nn_dist = p.distance(self.difference(p))
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

    def _edge_config(self, edge_correction):
        """
        Select the set of points to include and the intensity estimator mode
        to use in computations involving a particular edge correction

        :edge_correction: flag to select the handling of edges.
        :returns:
            - mode flag to self.points giving points to include with this edge
              correction
            - mode flag to self.squared_intensity giving the intensity
              estimator to use with this edge correction

        """
        if edge_correction == 'finite':
            pmode = 'standard'
            #imode = 'corrected'
            imode = 'standard'
        elif edge_correction == 'periodic':
            pmode = 'periodic'
            #imode = 'corrected'
            imode = 'standard'
        elif edge_correction == 'plus':
            pmode = 'plus'
            #imode = 'corrected'
            imode = 'standard'
        elif edge_correction in ('stationary', 'isotropic'):
            pmode = 'standard'
            imode = 'area'
        else:
            raise ValueError("unknown edge correction: {}"
                             .format(edge_correction))

        return pmode, imode

    @memoize_method
    def _kappasteps(self, edge_correction='stationary'):
        """
        Compute the steps in the numerator of empirical K-function of the point
        pattern

        :edge_correction: flag to select the handling of edges. See the
                          documentation for PointPattern.kfunc() for details.
        :returns:
            - r: array of r-values where the numerator of the empirical
              K-function steps up
            - kappa: array with values of the numerator of the empirical
              K-function.  Between r[i] and r[i + 1], the numerator has the
              value kappa[i].

        """
        window = self.window
        rmax = window.rmax(edge_correction)
        pmode, __ = self._edge_config(edge_correction)

        allpoints = self.points(mode=pmode)
        distances = self.pairwise_distances(self, allpoints)
        valid = numpy.logical_and(distances < rmax, distances != 0.0)

        index1, = numpy.nonzero(numpy.any(valid, axis=1))
        index2, = numpy.nonzero(numpy.any(valid, axis=0))
        mp1 = geometry.MultiPoint([self[i] for i in index1])
        mp2 = geometry.MultiPoint([allpoints[i] for i in index2])
        weight_matrix = window.weights(mp1, mp2, edge_correction)

        r = distances[valid]
        sort_ind = numpy.argsort(r)
        r = numpy.hstack((0.0, r[sort_ind], rmax))

        weights = weight_matrix[valid[index1, :][:, index2]]
        weights = numpy.hstack((0.0, weights[sort_ind]))
        kappa = numpy.append(numpy.cumsum(weights), numpy.nan)

        return r, kappa

    def kfunction(self, r, edge_correction='stationary'):
        """
        Compute the empirical K-function of the point pattern

        :r: array of r-values at which to evaluate the emprical K-function.
        :edge_correction: flag to select the handling of edges. Possible
                          values:
            'finite': Translational edge correction used. Intensity estimated
                      by the standard intensity estimator.
            'periodic': No edge correction used. Care is taken to avoid
                        counting the same pair multiple times, under the
                        assumption that self.pluspoints is a periodic extension
                        of self.points as defined by self.window.lattice.
                        Intensity estimated by the standard intensity
                        estimator.
            'plus': No edge correction used. Intensity estimated by the
                    standard intensity estimator.
            'stationary': Translational edge correction used. Intensity
                          estimated by the adapted intensity estimator based on
                          area.
            'isotropic': Rotational edge correction used. Intensity estimated
                         by the adapted intensity estimator based on area.
        :returns: array with the values of the empirical K-function
                  corresponding to the distances in `r`.

        """
        __, imode = self._edge_config(edge_correction)
        rsteps, kappa = self._kappasteps(edge_correction=edge_correction)
        indices = numpy.searchsorted(rsteps, r, side='right') - 1
        lambda2 = self.squared_intensity(mode=imode, r=r)
        return sensibly_divide(kappa[indices], lambda2)

    def lfunction(self, r, edge_correction='stationary'):
        """
        Compute the empirical L-function of the point pattern

        :r: array of r-values at which to evaluate the emprical L-function.
        :edge_correction: flag to select the handling of edges. See the
                          documentation for PointPattern.kfunction() for
                          details.
        :returns: array with the values of the empirical L-function
                  corresponding to the distances in `r`.

        """
        return numpy.sqrt(self.kfunction(r, edge_correction=edge_correction) /
                          numpy.pi)

    def plot_kfunction(self, axes=None, edge_correction='stationary',
                       linewidth=2.0, csr=False, csr_kw=None, **kwargs):
        """
        Plot the empirical K-function for the pattern

        The K-function can be added to an existing plot via the optional 'axes'
        argument.

        :axes: Axes instance to add the K-function to. If None (default), the
               current Axes instance is used if any, or a new one created.
        :edge_correction: flag to select the edge_correction to use in the
                          K-function. See the documentation for
                          PointPattern.kfunction() for details.
        :linewidth: number giving the width of the K-function line. Defaults to
                    2.0
        :csr: if True, overlay the curve K(r) = pi r^2, which is the
              theoretical K-function for complete spatial randomness. Defaults
              to a dashed line, but this and all other style parameters may be
              changed using csr_kw.
        :csr_kw: dict of keyword arguments to pass to the axes.plot() method
                 used to plot the CSR curve. Default: None (empty dict)
        :kwargs: additional keyword arguments passed on to the axes.plot()
                 method used to plot angles. Note in particular the keywords
                 'linestyle', 'color' and 'label'.
        :returns: list of handles to the Line2D instances added to the plot, in
                  the following order: empirical K-function, CSR curve
                  (optional).

        """
        if axes is None:
            axes = pyplot.gca()

        rmax = self.window.rmax(edge_correction)
        rvals = numpy.linspace(0.0, rmax, 3 * RSAMPLES)
        kvals = self.kfunction(rvals, edge_correction=edge_correction)

        lines = axes.plot(rvals, kvals, linewidth=linewidth, **kwargs)

        if csr:
            if csr_kw is None:
                csr_kw = {}

            kcsr = numpy.pi * rvals * rvals
            lines += axes.plot(rvals, kcsr, linestyle='dashed', **csr_kw)

        return lines

    def plot_lfunction(self, axes=None, edge_correction='stationary',
                       linewidth=2.0, csr=False, csr_kw=None, **kwargs):
        """
        Plot the empirical L-function for the pattern

        The L-function can be added to an existing plot via the optional 'axes'
        argument.

        :axes: Axes instance to add the L-function to. If None (default), the
               current Axes instance is used if any, or a new one created.
        :edge_correction: flag to select the edge_correction to use in the
                          K-function. See the documentation for
                          PointPattern.kfunction() for details.
        :linewidth: number giving the width of the L-function line. Defaults to
                    2.0
        :csr: if True, overlay the curve L(r) = r, which is the theoretical
              L-function for complete spatial randomness. Defaults to a dashed
              line, but this and all other style parameters may be changed
              using csr_kw.
        :csr_kw: dict of keyword arguments to pass to the axes.plot() method
                 used to plot the CSR curve. Default: None (empty dict)
        :kwargs: additional keyword arguments passed on to the axes.plot()
                 method used to plot angles. Note in particular the keywords
                 'linestyle', 'color' and 'label'.
        :returns: list of handles to the Line2D instances added to the plot, in
                  the following order: empirical L-function, CSR curve
                  (optional).

        """
        if axes is None:
            axes = pyplot.gca()

        rmax = self.window.rmax(edge_correction)
        rvals = numpy.linspace(0.0, rmax, 3 * RSAMPLES)
        lvals = self.lfunction(rvals, edge_correction=edge_correction)

        lines = axes.plot(rvals, lvals, linewidth=linewidth, **kwargs)

        if csr:
            if csr_kw is None:
                csr_kw = {}

            lines += axes.plot(rvals, rvals, linestyle='dashed', **csr_kw)

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

        pp = numpy.asarray(self)
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

    """

    def __init__(self, patterns):
        """
        Initialize the PointPatternCollection object.

        :patterns: list of PointPattern objects to include in the collection.

        """
        self.patterns = list(patterns)

    @classmethod
    def from_simulation(cls, window, intensity, process='binomial',
                        nsims=1000):
        """
        Create a PointPatternCollection instance by simulating a number of
        point patterns in the same window

        :window: a Window instance to simulate the process within
        :intensity: the intensity (density per unit area) of the process
        :process: string specifying the process to simulate. Possible values:
                  'binomial', 'poisson'
        :nsims: the number of patterns to generate from the process
        :returns: a PointPatternCollection instance containing the simulated
                  processes

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

            pp = PointPattern(points, window)
            patterns.append(pp)

        return cls(patterns)

    def __getitem__(self, key):
        return self.patterns.__getitem__(key)

    def __iter__(self):
        return self.patterns.__iter__()

    def __len__(self):
        return self.patterns.__len__()

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
        return min(pp.window.rmax(edge_correction) for pp in self.patterns)

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

    def aggregate_kfunction(self, r, edge_correction='stationary'):
        """
        Compute the aggregate of the empirical K-function over all patterns in
        the collection

        :r: array of r-values at which to evaluate the emprical aggregate
            K-function.
        :edge_correction: flag to select the handling of edges. See
                          PointPattern.kfunction() for details.
        :returns: array with the values of the empirical aggregate K-function
                  corresponding to the distances in `r`.

        """
        nweights = self.nweights
        kvalues = [pp.kfunction(r, edge_correction=edge_correction)
                   for pp in self.patterns]

        return sum(nw * kv for (nw, kv) in zip(nweights, kvalues))

    def aggregate_lfunction(self, r, edge_correction='stationary'):
        """
        Compute the aggregate of the empirical L-function over all patterns in
        the collection

        :r: array of r-values at which to evaluate the emprical aggregate
            L-function.
        :edge_correction: flag to select the handling of edges. See
                          PointPattern.kfunction() for details.
        :returns: array with the values of the empirical aggregate L-function
                  corresponding to the distances in `r`.

        """
        return numpy.sqrt(self.aggregate_kfunction(
            r, edge_correction=edge_correction) / numpy.pi)

    def kframe(self, r, edge_correction='stationary'):
        """
        Compute a DataFrame containing values of the empirical K-functions of
        the patterns

        :r: array of r-values at which to evaluate the empirical K-functions.
        :edge_correction: flag to select the handling of edges. See the
                          documentation for PointPattern.kfunction() for
                          details.
        :returns: DataFrame where each row contains values of the empirical
                  K-function of one pattern, evaluated at the distances in `r`.

        """
        return pandas.DataFrame(
            [pp.kfunction(r, edge_correction=edge_correction)
             for pp in self.patterns])

    def kcritical(self, alpha, r, edge_correction='stationary'):
        """
        Compute critical values of the empirical K-functions of the patterns

        :alpha: percentile defining the critical values
        :r: array of r-values at which to evaluate the critical values of the
            empirical K-functions.
        :edge_correction: flag to select the handling of edges. See the
                          documentation for PointPattern.kfunction() for
                          details.
        :returns: array with the critical values of the empirical K-functions
                  at the distances in `r`.

        """
        kvals = self.kframe(r, edge_correction=edge_correction)
        return kvals.quantile(q=alpha, axis=0)

    def kmean(self, r, edge_correction='stationary'):
        """
        Compute the mean of the empirical K-functions of the patterns

        :r: array of r-values at which to evaluate the critical values of the
            empirical K-functions.
        :edge_correction: flag to select the handling of edges. See the
                          documentation for PointPattern.kfunction() for
                          details.
        :returns: array with the mean of the empirical K-functions at the
                  distances in `r`.

        """
        kvals = self.kframe(r, edge_correction=edge_correction)
        return kvals.mean(axis=0, skipna=True)

    def lframe(self, r, edge_correction='stationary'):
        """
        Compute a DataFrame containing values of the empirical L-functions of
        the patterns

        :r: array of r-values at which to evaluate the empirical L-functions.
        :edge_correction: flag to select the handling of edges. See the
                          documentation for PointPattern.kfunction() for
                          details.
        :returns: DataFrame where each row contains values of the empirical
                  L-function of one pattern, evaluated at the distances in `r`.

        """
        return pandas.DataFrame(
            [pp.lfunction(r, edge_correction=edge_correction)
             for pp in self.patterns])

    def lcritical(self, alpha, r, edge_correction='stationary'):
        """
        Compute critical values of the empirical L-functions of the patterns

        :alpha: percentile defining the critical values
        :r: array of r-values at which to evaluate the critical values of the
            empirical L-functions.
        :edge_correction: flag to select the handling of edges. See the
                          documentation for PointPattern.kfunction() for
                          details.
        :returns: array with the critical values of the empirical L-functions
                  at the distances in `r`.

        """
        lvals = self.lframe(r, edge_correction=edge_correction)
        return lvals.quantile(q=alpha, axis=0)

    def lmean(self, r, edge_correction='stationary'):
        """
        Compute the mean of the empirical L-functions of the patterns

        :r: array of r-values at which to evaluate the critical values of the
            empirical L-functions.
        :edge_correction: flag to select the handling of edges. See the
                          documentation for PointPattern.kfunction() for
                          details.
        :returns: array with the mean of the empirical L-functions at the
                  distances in `r`.

        """
        lvals = self.lframe(r, edge_correction=edge_correction)
        return lvals.mean(axis=0, skipna=True)

    def plot_kenvelope(self, axes=None, edge_correction='stationary',
                       low=0.025, high=0.975, alpha=0.25, **kwargs):
        """
        Plot an envelope of empirical K-function values

        The envelope can be added to an existing plot via the optional 'axes'
        argument.

        :axes: Axes instance to add the envelope to. If None (default), the
               current Axes instance is used if any, or a new one created.
        :edge_correction: flag to select the handling of edges. See the
                          documentation for PointPattern.kfunction() for
                          details.
        :low: percentile defining the lower edge of the envelope
        :high: percentile defining the higher edge of the envelope
        :alpha: the opacity of the envelope
        :kwargs: additional keyword arguments passed on to the
                 axes.fill_between() method. Note in particular the keywords
                 'edgecolor', 'facecolor' and 'label'.
        :returns: the PolyCollection instance filling the envelope.

        """
        if axes is None:
            axes = pyplot.gca()

        rvals = numpy.linspace(0.0, self.rmax(edge_correction), 3 * RSAMPLES)
        kvals_low = self.kcritical(low, rvals, edge_correction=edge_correction)
        kvals_high = self.kcritical(high, rvals,
                                    edge_correction=edge_correction)

        h = axes.fill_between(rvals, kvals_low, kvals_high, alpha=alpha,
                              **kwargs)
        return h

    def plot_kmean(self, axes=None, edge_correction='stationary', **kwargs):
        """
        Plot an the mean of the empirical K-function values

        The mean can be added to an existing plot via the optional 'axes'
        argument.

        :axes: Axes instance to add the mean to. If None (default), the current
               Axes instance is used if any, or a new one created.
        :edge_correction: flag to select the handling of edges. See the
                          documentation for PointPattern.kfunction() for
                          details.
        :kwargs: additional keyword arguments passed on to the axes.plot()
                 method. Note in particular the keywords 'linewidth',
                 'linestyle', 'color', and 'label'.
        :returns: list containing the Line2D of the plotted mean.

        """
        if axes is None:
            axes = pyplot.gca()

        rvals = numpy.linspace(0.0, self.rmax(edge_correction), 3 * RSAMPLES)
        kmean = self.kmean(rvals, edge_correction=edge_correction)

        h = axes.plot(rvals, kmean, **kwargs)
        return h

    def plot_lenvelope(self, axes=None, edge_correction='stationary',
                       low=0.025, high=0.975, alpha=0.25, **kwargs):
        """
        Plot an envelope of empirical L-function values

        The envelope can be added to an existing plot via the optional 'axes'
        argument.

        :axes: Axes instance to add the envelope to. If None (default), the
               current Axes instance is used if any, or a new one created.
        :edge_correction: flag to select the handling of edges. See the
                          documentation for PointPattern.kfunction() for
                          details.
        :low: percentile defining the lower edge of the envelope
        :high: percentile defining the higher edge of the envelope
        :alpha: the opacity of the envelope
        :kwargs: additional keyword arguments passed on to the
                 axes.fill_between() method. Note in particular the keywords
                 'edgecolor', 'facecolor' and 'label'.
        :returns: the PolyCollection instance filling the envelope.

        """
        if axes is None:
            axes = pyplot.gca()

        rvals = numpy.linspace(0.0, self.rmax(edge_correction), 3 * RSAMPLES)
        lvals_low = self.lcritical(low, rvals, edge_correction=edge_correction)
        lvals_high = self.lcritical(high, rvals,
                                    edge_correction=edge_correction)

        h = axes.fill_between(rvals, lvals_low, lvals_high, alpha=alpha,
                              **kwargs)
        return h

    def plot_lmean(self, axes=None, edge_correction='stationary', **kwargs):
        """
        Plot an the mean of the empirical L-function values

        The mean can be added to an existing plot via the optional 'axes'
        argument.

        :axes: Axes instance to add the mean to. If None (default), the current
               Axes instance is used if any, or a new one created.
        :edge_correction: flag to select the handling of edges. See the
                          documentation for PointPattern.kfunction() for
                          details.
        :kwargs: additional keyword arguments passed on to the axes.plot()
                 method. Note in particular the keywords 'linewidth',
                 'linestyle', 'color', and 'label'.
        :returns: list containing the Line2D of the plotted mean.

        """
        if axes is None:
            axes = pyplot.gca()

        rvals = numpy.linspace(0.0, self.rmax(edge_correction), 3 * RSAMPLES)
        lmean = self.lmean(rvals, edge_correction=edge_correction)

        h = axes.plot(rvals, lmean, **kwargs)
        return h

    def plot_aggregate_kfunction(self, axes=None, edge_correction='stationary',
                                 linewidth=2.0, csr=False, csr_kw=None,
                                 **kwargs):
        """
        Plot the aggregate of the empirical K-function over all patterns in the
        collection

        The K-function can be added to an existing plot via the optional 'axes'
        argument.

        :axes: Axes instance to add the K-function to. If None (default), the
               current Axes instance is used if any, or a new one created.
        :edge_correction: flag to select the edge_correction to use in the
                          K-function. See the documentation for
                          PointPattern.kfunction() for details.
        :linewidth: number giving the width of the K-function line. Defaults to
                    2.0
        :csr: if True, overlay the curve K(r) = pi r^2, which is the
              theoretical K-function for complete spatial randomness. Defaults
              to a dashed line, but this and all other style parameters may be
              changed using csr_kw.
        :csr_kw: dict of keyword arguments to pass to the axes.plot() method
                 used to plot the CSR curve. Default: None (empty dict)
        :kwargs: additional keyword arguments passed on to the axes.plot()
                 method used to plot angles. Note in particular the keywords
                 'linestyle', 'color' and 'label'.
        :returns: list of handles to the Line2D instances added to the plot, in
                  the following order: empirical K-function, CSR curve
                  (optional).

        """
        if axes is None:
            axes = pyplot.gca()

        rmax = self.window.rmax(edge_correction)
        rvals = numpy.linspace(0.0, rmax, 3 * RSAMPLES)
        kvals = self.aggregate_kfunction(
            rvals, edge_correction=edge_correction)

        lines = axes.plot(rvals, kvals, linewidth=linewidth, **kwargs)

        if csr:
            if csr_kw is None:
                csr_kw = {}

            kcsr = numpy.pi * rvals * rvals
            lines += axes.plot(rvals, kcsr, linestyle='dashed', **csr_kw)

        return lines

    def plot_aggregate_lfunction(self, axes=None, edge_correction='stationary',
                                 linewidth=2.0, csr=False, csr_kw=None,
                                 **kwargs):
        """
        Plot the aggregate of the empirical L-function over all patterns in the
        collection

        The L-function can be added to an existing plot via the optional 'axes'
        argument.

        :axes: Axes instance to add the L-function to. If None (default), the
               current Axes instance is used if any, or a new one created.
        :edge_correction: flag to select the edge_correction to use in the
                          K-function. See the documentation for
                          PointPattern.kfunction() for details.
        :linewidth: number giving the width of the L-function line. Defaults to
                    2.0
        :csr: if True, overlay the curve L(r) = r, which is the theoretical
              L-function for complete spatial randomness. Defaults to a dashed
              line, but this and all other style parameters may be changed
              using csr_kw.
        :csr_kw: dict of keyword arguments to pass to the axes.plot() method
                 used to plot the CSR curve. Default: None (empty dict)
        :kwargs: additional keyword arguments passed on to the axes.plot()
                 method used to plot angles. Note in particular the keywords
                 'linestyle', 'color' and 'label'.
        :returns: list of handles to the Line2D instances added to the plot, in
                  the following order: empirical L-function, CSR curve
                  (optional).

        """
        if axes is None:
            axes = pyplot.gca()

        rmax = self.window.rmax(edge_correction)
        rvals = numpy.linspace(0.0, rmax, 3 * RSAMPLES)
        lvals = self.aggregate_lfunction(
            rvals, edge_correction=edge_correction)

        lines = axes.plot(rvals, lvals, linewidth=linewidth, **kwargs)

        if csr:
            if csr_kw is None:
                csr_kw = {}

            lines += axes.plot(rvals, rvals, linestyle='dashed', **csr_kw)

        return lines
