#!/usr/bin/env python

"""File: imaps.py
Module defining classes to represent intensity maps, with methods for
convolving, correlating, smoothing, computing Fourier transforms, plotting,
peak detection etc.

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
from scipy import signal, fftpack, special
from scipy.ndimage import filters, measurements, interpolation
from skimage import feature  #, exposure
from matplotlib import pyplot

from .utils import (AlmostImmutable, sensibly_divide, pearson_correlogram,
                    disc_overlap)
from .ndfit import fit_ndgaussian

try:
    __ = basestring
except NameError:
    basestring = str

_SQRT2 = numpy.sqrt(2.0)
_SQRT3 = numpy.sqrt(3.0)


class BinnedSet(AlmostImmutable):
    """
    Represent a rectangle, or in general a hyperrectangle, divided into bins,
    and provide methods for computing derived BinnedSet instances such as those
    corresponding to correlograms and Fourier transforms.

    """

    def __init__(self, edges):
        """
        Initilize a BinnedSet instance

        :edges: sequence of array-like containers for bin edge values, such as
                the tuple returned from numpy.histogramdd. A single array-like
                may be provided in the one-dimensional case.

        """
        # Standardize the bin edge arrays and assign to an attribute
        edgearrs, dims = [], []
        for e in edges:
            earr = numpy.array(e)
            edgearrs.append(earr)
            dims.append(earr.ndim)

        try:
            d = dims[0]
        except IndexError:
            self.edges = ()
        else:
            equal = (dim == d for dim in dims)
            if d == 0 and all(equal):
                self.edges = (numpy.array(edgearrs),)
            elif d == 1 and all(equal):
                self.edges = tuple(edgearrs)
            else:
                raise ValueError("'edges' must be sequence of one-dimensional "
                                 "array-like containers")

        # Compute bin widths and check validity
        valid = True
        binwidths = []
        for e in self.edges:
            ds = numpy.diff(e)
            valid = valid and numpy.all(ds > 0.0)
            binwidths.append(ds)

        if not valid:
            raise ValueError("valid bin edge arrays can only contain unique, "
                             "sorted values")

        self.binwidths = tuple(binwidths)

    @classmethod
    def from_center_binwidth_shape(cls, center, binwidth, shape):
        """
        Factory method: construct a BinnedSet instance with regular bins from
        the center, bin width and number of bins along each axis

        :center: sequence of centers. Each number gives the center of the
                 BinnedSet along each axis. A single number may be provided in
                 the one-dimensional case.
        :binwidth: sequence of bin widths. Each number gives the bin width
                   along each direction. A single number may be provided in the
                   one-dimensional case.
        :shape: sequence of bin numbers. Each number gives the number of bins
                along one axis. A single number may be provided in the
                one-dimensional case.

        """
        l = []
        vars_ = []
        for var in (center, binwidth, shape):
            try:
                lv = len(var)
            except TypeError:
                var = (var,)
                lv = 1
            l.append(lv)
            vars_.append(var)

        if not (l[0] == l[1] and l[0] == l[2]):
            raise ValueError("lengths of 'center', 'binwidth' and 'shape' not "
                             "equal")

        edges = [c + b * numpy.arange(start=(-0.5 * (n)), stop=(0.5 * (n + 1)))
                 for (c, b, n) in zip(*vars_)]

        return cls(edges)

    @classmethod
    def from_range_shape(cls, range_, shape):
        """
        Factory method: construct a BinnedSet instance with regular bins from
        the range and number of bins

        :range_: sequence of range tuples. Each range tuple gives the limit of
                 the BinnedSet along one axis. A single edge tuple may be
                 provided in the one-dimensional case.
        :shape: sequence of bin numbers. Each number gives the number of bins
                along one axis. A single number may be provided in the
                one-dimensional case.

        """
        try:
            ln = len(shape)
        except TypeError:
            shape = (shape,)
            ln = 1

        lr = len(range_)
        if ln != lr:
            if ln == 1 and lr == 2:
                range_ = (range_,)
            else:
                raise ValueError("lengths of 'range_' and 'shape' not equal")

        center, binwidth = [], []
        for (r, n) in zip(range_, shape):
            center.append(0.5 * (r[0] + r[1]))
            binwidth.append((r[1] - r[0]) / n)

        return cls.from_center_binwidth_shape(center, binwidth, shape)

    def __eq__(self, other):
        """
        Define equality as the exact equality of all bin edge arrays

        :other: another BinnedSet instance
        :returns: True if equal, otherwise False

        """
        if not type(other) == type(self):
            return False
        sedges, oedges = self.edges, other.edges
        return ((len(sedges) == len(oedges)) and
                all(numpy.all(se == oe) for (se, oe) in zip(sedges, oedges)))

    def __neq__(self, other):
        return not self.__eq__(other)

    def __add__(self, vector):
        """
        Define addition of a BinnedSet with a sequence of numbers as
        translation

        :vector: sequence of numbers to add to the existing bin edge values,
                 one for each dimension. In the one-dimensional case, a single
                 number is accepted.
        :returns: new, translated BinnedSet instance

        """
        return self.translated(vector)

    def __neg__(self):
        """
        Define negation of a BinnedSet as reflection through the origin

        :returns: new, negated (reflected) BinnedSet instance

        """
        return self.reflected()

    @property
    def ndim(self):
        """
        The dimensionality of this BinnedSet instance

        """
        return len(self.edges)

    @property
    def shape(self):
        """
        The number of bins in the BinnedSet instance along each axis

        """
        return tuple(e.size - 1 for e in self.edges)

    @property
    def size(self):
        """
        The total number of bins in the BinnedSet

        """
        return numpy.prod(self.shape)

    @property
    def range_(self):
        """
        The range of values spanned by the BinnedSet along each axis

        """
        return tuple((e[0], e[-1]) for e in self.edges)

    @property
    def centers(self):
        """
        Tuple of arrays giving the bin center values

        """
        return tuple(0.5 * (e[1:] + e[:-1]) for e in self.edges)

    @property
    def mesh(self):
        """
        Meshgrid arrays created from the bin edge arrays

        The matrix ('ij') indexing convention is followed, such that if the
        length of the bin edge arrays is n0, n1, ..., and nk_1 (in the order
        they appear in self.edges), the shape of each of the mesh arrays is
        (n0, n2, ..., nk_1).

        """
        return numpy.meshgrid(*self.edges, indexing='ij')

    @property
    def cmesh(self):
        """
        Meshgrid arrays created from the bin center arrays

        The matrix ('ij') indexing convention is followed, such that if the
        length of the bin center arrays is n0, n1, ..., and nk_1 (in the order
        they appear in self.edges), the shape of each of the centermesh arrays
        is (n0, n2, ..., nk_1).

        """
        return numpy.meshgrid(*self.centers, indexing='ij')

    @property
    def area(self):
        """
        Array giving the area (in general, volume) volume of each bin

        The matrix ('ij') indexing convention is followed, such that if the
        a bin is defined by
        (self.edges[0][i0], self.edges[0][i0 + 1],
         self.edges[1][i1], self.edges[0][i1 + 1],
         ...
         self.edges[k - 1][ik_1], self.edges[k - 1][ik_1 + 1]),
        its area is given by self.area[i0, i1, ..., ik_1).

        """
        return numpy.prod(
            numpy.meshgrid(*self.binwidths, indexing='ij'), axis=0)

    @property
    def total_area(self):
        """
        The total area (in general, volume) of the BinnedSet

        """
        return numpy.prod([r[1] - r[0] for r in self.range_])

    @property
    def regular(self):
        """
        Whether this BinnedSet instance is regular (has bins that are equal in
        size and shape)

        """
        return all(numpy.allclose(w, w[0]) for w in self.binwidths)

    @property
    def square(self):
        """
        Whether the bins defined by this BinnedSet instance are square, or in
        general hypercubical

        """
        return self.regular and numpy.allclose(*(w[0] for w in self.binwidths))

    def compatible(self, other):
        """
        Check if this instance is compatible with another instance (both
        instances have bins of equal size and shape)

        Compatibility requires that both instances are regular and have the
        same dimensionality.

        :other: another BinnedSet instance
        :returns: True if compatible, otherwise False

        """
        compatible = ((self.ndim == other.ndim) and
                      self.regular and
                      other.regular and
                      all(numpy.allclose(sw[0], ow[0]) for (sw, ow) in
                          zip(self.binwidths, other.binwidths)))
        return compatible

    def translated(self, vector):
        """
        Return a translated copy of the BinnedSet instance

        :vector: sequence of numbers to add to the existing bin edge values,
                 one for each dimension. In the one-dimensional case, a single
                 number is accepted.
        :returns: new, translated BinnedSet instance

        """
        try:
            l = len(vector)
        except TypeError:
            vector = (vector,)
            l = 1

        if not l == self.ndim:
            raise ValueError("'vector' must be a sequence containing a number "
                             "for each dimension in the {} instance"
                             .format(self.__class__.__name__))

        new_edges = [e + v for (e, v) in zip(self.edges, vector)]
        return type(self)(new_edges)

    def reflected(self):
        """
        Return a spatially reflected copy of the BinnedSet instance

        :returns: new, reflected BinnedSet instance

        """
        # Manually memoize to take advantage of the fact that this is an
        # involution
        try:
            return self._reflected
        except AttributeError:
            new_edges = [-e[::-1] for e in self.edges]
            self._reflected = type(self)(new_edges)
            self._reflected._reflected = self  # Involution at work
            return self._reflected

    def fft(self):
        """
        Compute a frequency space BinnedSet matching the FFT of a map over this
        BinnedSet

        The zero-frequency is placed in the center of the BinnedSet along each
        axis. Care has been taken to ensure compatibility with the
        scipy.fftpack.fftn() regarding the handling of odd/even number of bins.

        To use the returned BinnedSet with an FFT array returned from
        scipy.fftpack.fftn(), scipy.fftpack.fftshift() should be called on the
        FFT array to center the zero frequency.

        :returns: new FFT-matching BinnedSet instance

        """
        if not self.regular:
            raise ValueError("a {} instance must be regular (have bins of "
                             "equal size and shape) to compute an "
                             "FFT-compatible instance"
                             .format(self.__class__.__name__))

        center, fbinwidth, shape = [], [], []
        for (sw, se) in zip(self.binwidths, self.edges):
            nb = se.size - 1
            fbw = 1 / (numpy.mean(sw) * nb)
            center.append(0.5 * fbw * ((nb % 2) - 1))
            fbinwidth.append(fbw)
            shape.append(nb)

        return self.from_center_binwidth_shape(center, fbinwidth, shape)

    def convolve(self, other, mode='full'):
        """
        Compute a BinnedSet instance matching a convolution of maps over this
        and another instance

        Care has been taken to ensure compatibility with scipy.signal.convolve
        regarding the handling of different modes and odd/even number of bins.

        :other: another BinnedSet instance
        :mode: string indicating the size of the output. See
               scipy.signal.convolve for details. Valid options:
               'full', 'valid', 'same'. Default is 'full'.
        :returns: new convolution-mathcing BinnedSet instance

        """
        if not self.compatible(other):
            raise ValueError("{} instances must be compatible (have bins of "
                             "equal size and shape) to compute a "
                             "convolution-matching instance"
                             .format(self.__class__.__name__))

        if mode == 'full':
            def shape_shift(sshape, oshape):
                return sshape + oshape - 1, 0
        elif mode == 'valid':
            def shape_shift(sshape, oshape):
                return abs(sshape - oshape) + 1, 0
        elif mode == 'same':
            def shape_shift(sshape, oshape):
                return sshape, (oshape % 2) - 1
        else:
            raise ValueError("unknown mode {}".format(mode))

        center, binwidth, shape = [], [], []
        for (sw, se, oe) in zip(self.binwidths, self.edges, other.edges):
            nb, sh = shape_shift(se.size - 1, oe.size - 1)
            bw = numpy.mean(sw)
            center.append(0.5 * ((se[-1] + se[0]) + (oe[-1] + oe[0]) +
                          bw * sh))
            binwidth.append(bw)
            shape.append(nb)

        return self.from_center_binwidth_shape(center, binwidth, shape)

    def correlate(self, other, mode='full'):
        """
        Compute a BinnedSet instance matching a correlogram of maps over this
        and another instance

        Care has been taken to ensure compatibility with scipy.signal.correlate
        regarding the handling of different modes and odd/even number of bins.

        :other: another BinnedSet instance
        :mode: string indicating the size of the output. See
               scipy.signal.correlate for details. Valid options:
                   'full', 'valid', 'same'. Default is 'full'.
        :returns: new correlogram-mathcing BinnedSet instance

        """
        return self.convolve(-other, mode=mode)

    def autocorrelate(self, mode='full'):
        """
        Compute a BinnedSet instance matching an autocorrelogram of a map over
        this instance

        Convenience method for calling self.correlate(self, mode=mode)

        :mode: see BinnedSet.correlate()
        :returns: new autocorrelogram-mathcing BinnedSet instance

        """
        return self.correlate(self, mode=mode)

    def coordinates(self, bin_indices):
        """
        Find the coordinates of positions given by (possibly fractional) bin
        indices

        It is assumed that the given indices refer to the bin centers, and not
        the bin edges.

        The coordinates of the positions are computed by interpolating with the
        given indices in the map from array indices to bin centers.

        :bin_indices: a sequence of (possibly fractional) index tuples.
                      A single index tuple is also accepted.
        :returns: an array where element [i, j] gives the jth coordinate of the
                  ith point
        """
        indices = numpy.asarray(bin_indices).transpose()
        coords = [numpy.interp(i, range(n), c)
                  for (i, n, c) in zip(indices, self.shape, self.centers)]
        coords_arr = numpy.array(coords).transpose()
        return numpy.atleast_2d(coords_arr)

    def bin_indices(self, coordinates, fractional=True):
        """
        Find the (possibly fractional) bin indices of positions given by
        coordinates

        The returned indices will refer to the bin centers, not to the bin
        edges.

        The bin indices are computed by interpolating with the
        given coordinates in the map from bin centers to array indices.

        Parameters
        ----------
        coordinates : array-like
            Sequence of coordinate tuples.
        fractional : bool, optional
            If True, fractional indices are returned. Otherwise, integer bin
            indices to the bins that the coordinates lie within are returned.
            Coordinates exactly at the bin edges are rounded downwards.

        Returns
        ------
        ndarray
            Array where element [i, j] gives the jth (fractional) index
            corresponding to the ith coordinate position
        """
        coords = numpy.asarray(coordinates).transpose()
        indices = [numpy.interp(coo, cen, range(n))
                   for (coo, cen, n) in zip(coords, self.centers, self.shape)]
        index_arr = numpy.atleast_2d(numpy.array(indices).transpose())
        if fractional:
            return index_arr
        return numpy.floor(index_arr + 0.5).astype(numpy.int_)

    def extend(self, extension):
        """
        Extend the BinnedSet with more bins along its dimension

        This is only possible for regular BinnedSets (self.regular == True).

        Parameters
        ----------
        extension : sequence
            Sequence where each element is a two-element sequence containing
            the number of bins to add at the beginning and end, respectively,
            of the corresponding dimension. For example, if `extension=[(2, 3),
            (1, 0)]`, the BinnedSet is extended with 2 new bins before and
            3 new bins after the existing bins along the first dimension, and
            1 new bin before existing bins along the second dimension.

        Returns
        -------
        BinnedSet
            Extended BinnedSet

        """
        if not self.regular:
            raise ValueError("{} instances must be regular (have bins of "
                             "equal size and shape) to compute extensions"
                             .format(self.__class__.__name__))

        old_edges = self.edges
        new_edges = []
        widths = (numpy.mean(w) for w in self.binwidths)
        for (ext, old_edge, width) in zip(extension, old_edges, widths):
            old_min, old_max = old_edge[(0, -1), ]
            new_start = numpy.arange(old_min - width * ext[0],
                                     old_min - width * 0.5, width)
            new_end = numpy.arange(old_max + width,
                                   old_max + width * (ext[1] + 0.5), width)
            new_edge = numpy.concatenate((new_start, old_edge, new_end))
            new_edges.append(new_edge)

        # Append remaining unchanged edge arrays
        new_edges += old_edges[len(new_edges):]

        return type(self)(new_edges)


class BinnedSet2D(BinnedSet):
    """
    A specialization of the BinnedSet to two-dimensional rectangles. Features
    a simplified call signature, 2D-specific properties, and a plotting method.

    """

    def __init__(self, edges, yedges=None):
        """
        Initilize a BinnedSet2D instance

        :edges: list or tuple of numpy bin edge arrays such as those returned
                from numpy.histogram2d, in the format (xedges, yedges). If the
                optional argument yedges is provided, this is instead assumed
                to be the numpy bin edge array for the first (x) axis. Thus,
                both the call signatures BinnedSet((xedges, yedges)) and
                BinnedSet(xedges, yedges) are valid and give identical results.
                This provides a flatter call signature for direct use, while
                maintaining compatibility with the BinnedSet call signature.
        :yedges: numpy bin edge array for the second (y) axis, such as that
                 returned from numpy.histogram2d.

        """
        if yedges is None:
            if len(edges) == 2:
                BinnedSet.__init__(self, edges)
            else:
                raise ValueError("'edges' does not contain exactly two bin "
                                 "edge arrays")
        else:
            BinnedSet.__init__(self, (edges, yedges))

    @property
    def xedges(self):
        """
        Numpy array giving the values of the bin edges along the x axis

        """
        return self.edges[0]

    @property
    def yedges(self):
        """
        Numpy array giving the values of the bin edges along the y axis

        """
        return self.edges[1]

    @property
    def xcenters(self):
        """
        Numpy array giving the values of the bin centers along the x axis

        """
        return self.centers[0]

    @property
    def ycenters(self):
        """
        Numpy array giving the values of the bin centers along the y axis

        """
        return self.centers[1]

    @property
    def xbinwidths(self):
        """
        Numpy array giving the bin widths along the x axis

        """
        return self.binwidths[0]

    @property
    def ybinwidths(self):
        """
        Numpy array giving the bin widths along the y axis

        """
        return self.binwidths[1]

    def plot(self, axes=None, frame_lwfactor=2.0, color='0.5', **kwargs):
        """
        Plot the bin edges

        The edges can be added to an existing plot via the optional 'axes'
        argument.

        :axes: Axes instance to add the bin edges to. If None (default), the
               current Axes instance with equal aspect ratio is used if any, or
               a new one created.
        :frame_lwfactor: a number used to scale the thickness of the outer
                         edges as a multiple of the thickness of the interior
                         edges (which is given by the current rcParams or
                         **kwargs).  Default value: 2.0.
        :color: a valid matplotlib color specification giving the color to plot
                the bin edges with. Defaults to '0.5', a moderately light gray.
        :kwargs: additional keyword arguments passed on to axes.plot() for
                 specifying line properties.
        :returns: the Axes instance in which the edges have been plotted

        """
        if axes is None:
            axes = pyplot.gca(aspect='equal')

        xmin, xmax = self.xedges[(0, -1), ]
        ymin, ymax = self.yedges[(0, -1), ]

        xe = numpy.tile(self.xedges, (2, 1))
        ye = numpy.tile(self.yedges, (2, 1))

        xlines = axes.plot(xe, [ymin, ymax], color=color, **kwargs)
        ylines = axes.plot([xmin, xmax], ye, color=color, **kwargs)

        # Make the outer edges extra thick
        for line in (xlines[0], xlines[-1], ylines[0], ylines[-1]):
            line.set_linewidth(frame_lwfactor * line.get_linewidth())

        return axes


class IntensityMap(AlmostImmutable):
    """
    Represent an intensity map defined over a BinnedSet, and provide methods
    for convolving, correlating, smoothing, computing Fourier transforms, peak
    detection etc.

    Parameters
    ----------
    data : array-like
        A map of scalar intensities. Nan-valued and/or masked elements (using
        the `numpy.ma` module) may be used to represent missing data. Note that
        while several `IntensityMap` methods are explicitly designed to handle
        missing values (e.g. `smoothing`, `convolve` and `correlate`, when
        `normalize == True`), other methods are not designed with this in mind
        (e.g. `fft`).  In the latter case, missing values will be represented
        as nans in an unmasked array and handled by floating-point arithmetic.
    bset : BinnedSet or sequence
        A `BinnedSet` instance, or a valid argument list to the `BinnedSet`
        constructor, defining the spatial bins where the `IntensityMap`
        instance takes values. The ordering of axes in `bset` and `data` should
        correspond, such that `data[i0, i1, ..., ik_1]` gives the intensity in
        the bin defined by
        ```
        (bset.edges[0][i0], bset.edges[0][i0 + 1],
         bset.edges[1][i1], bset.edges[1][i1 + 1],
         ...,
         bset[k-1][ik_1], bset[k-1][ik_1 + 1])
        ```
    **kwargs : dict, optional
        Any other keyword arguments are passed to the `numpy.ma.array`
        constructor along with `data` and `mask=numpy.isnan(data)`.

    """

    def __init__(self, data, bset, **kwargs):
        self._data = numpy.ma.array(data, mask=numpy.isnan(data),
                                    keep_mask=True, copy=True, **kwargs)

        if not isinstance(bset, BinnedSet):
            bset = BinnedSet(*bset)

        # Check that data and bset are compatible
        shape, edges = self.shape, bset.edges
        dims_equal = (len(shape) == len(edges))
        size_equal = (s == e.size - 1 for (s, e) in zip(shape, edges))
        if not (dims_equal and all(size_equal)):
            raise ValueError("`bset` shape not compatible with this {} "
                             "instance".format(self.__class__.__name__))

        self.bset = bset

    def new_from_array(self, arr):
        """
        Create a new IntensityMap instance which is compatible ("convolutable",
        "correlatable") with this instance, from a numpy array.

        This is only possible if this instance is defined over a regular
        BinnedSet. The new instance will be centered over the origin.

        :arr: numpy array of the same dimensionality as self.data
        :returns: new IntensityMap instance, compatible with this instance

        """
        if not self.bset.regular:
            raise ValueError("an {} instance must be defined over a regular "
                             "{} instance (having bins of equal size and "
                             "shape) to be able to construct a compatible "
                             "instance from an array"
                             .format(self.__class__.__name__,
                                     self.bset.__class__.__name__))

        center = [0.0] * self.ndim
        binwidth = [numpy.mean(w) for w in self.bset.binwidths]
        shape = arr.shape
        new_bset = self.bset.from_center_binwidth_shape(center,
                                                        binwidth,
                                                        shape)

        return type(self)(arr, new_bset)

    def _inherit_binary_operation(self, other, op):
        """
        Define the general pattern for inheriting a binary operation on the
        data as a binary operation on the IntensityMap

        Parameters
        ----------
        other : array-like or IntensityMap
            The binary operation is applied to `self` and `other`. If `other`
            is also an `IntensityMap` instance, an exception is raised if they
            are not defined over `BinnedSet` instances that compare equal.
        op : string or callable
            Either a string naming the attribute of `self.data` that implements
            the binary operation, or a callable implementing the binary
            operation on two `self.data`-like objects.

        Returns
        -------
        IntensityMap
            The result of the binary operation applied to the `IntensityMap`
            instances.

        """
        sdata = self.data
        if isinstance(op, basestring) and hasattr(sdata, op):
            bound_op = getattr(sdata, op)
        else:
            def bound_op(odata):
                return op(sdata, odata)

        bset = self.bset
        if isinstance(other, type(self)) or isinstance(self, type(other)):
            obset = other.bset
            if not ((bset == obset) or
                    bset.shape == () or
                    obset.shape == ()):
                raise ValueError("instances of {} must be defined over "
                                 "instances of {} that compare equal for "
                                 "binary operations to be defined"
                                 .format(self.__class__.__name__,
                                         bset.__class__.__name__))
            new_data = bound_op(other.data)
            if bset.shape == ():
                bset = obset
        else:
            new_data = bound_op(other)

        return type(self)(new_data, bset)

    def __eq__(self, other):
        return self._inherit_binary_operation(other, '__eq__')

    def __neq__(self, other):
        return ~self.__eq__(other)

    def __lt__(self, other):
        return self._inherit_binary_operation(other, '__lt__')

    def __le__(self, other):
        return self._inherit_binary_operation(other, '__le__')

    def __gt__(self, other):
        return self._inherit_binary_operation(other, '__gt__')

    def __ge__(self, other):
        return self._inherit_binary_operation(other, '__ge__')

    def __add__(self, other):
        return self._inherit_binary_operation(other, '__add__')

    def __radd__(self, other):
        return self._inherit_binary_operation(other, '__radd__')

    def __sub__(self, other):
        return self._inherit_binary_operation(other, '__sub__')

    def __rsub__(self, other):
        return self._inherit_binary_operation(other, '__rsub__')

    def __mul__(self, other):
        return self._inherit_binary_operation(other, '__mul__')

    def __rmul__(self, other):
        return self._inherit_binary_operation(other, '__rmul__')

    def __div__(self, other):
        return self._inherit_binary_operation(other, '__div__')

    def __truediv__(self, other):
        def truediv(sdata, odata):
            return sensibly_divide(sdata, odata, masked=True)
        return self._inherit_binary_operation(other, truediv)

    def __rtruediv__(self, other):
        def rtruediv(sdata, odata):
            return sensibly_divide(odata, sdata, masked=True)
        return self._inherit_binary_operation(other, rtruediv)

    def __floordiv__(self, other):
        return self._inherit_binary_operation(other, '__floordiv__')

    def __rfloordiv__(self, other):
        return self._inherit_binary_operation(other, '__rfloordiv__')

    def __pow__(self, other):
        return self._inherit_binary_operation(other, '__pow__')

    def __rpow__(self, other):
        return self._inherit_binary_operation(other, '__rpow__')

    def __and__(self, other):
        return self._inherit_binary_operation(other, '__and__')

    def __rand__(self, other):
        return self._inherit_binary_operation(other, '__rand__')

    def __xor__(self, other):
        return self._inherit_binary_operation(other, '__xor__')

    def __rxor__(self, other):
        return self._inherit_binary_operation(other, '__rxor__')

    def __or__(self, other):
        return self._inherit_binary_operation(other, '__or__')

    def __ror__(self, other):
        return self._inherit_binary_operation(other, '__ror__')

    # Negation is implemented explicitly to take advantage of the involution
    # property
    #def __neg__(self):
    #    return type(self)(self.data.__neg__(), self.bset)

    def __pos__(self):
        return type(self)(self.data.__pos__(), self.bset)

    def __abs__(self):
        return type(self)(self.data.__abs__(), self.bset)

    def __invert__(self):
        return type(self)(self.data.__invert__(), self.bset)

    def __neg__(self):
        """
        Define negation of an IntensityMap instance as the negation of its
        intensity array

        :returns: negated IntensityMap instance

        """
        # Manually memoize to take advantage of the fact that this is an
        # involution
        try:
            return self._neg
        except AttributeError:
            self._neg = type(self)(self.data.__neg__(), self.bset)
            self._neg._neg = self  # Involution at work
            return self._neg

    def reflected(self):
        """
        Create a spatially reflected copy of the IntensityMap instance

        :returns: new, reflected IntensityMap instance

        """
        # Manually memoize to take advantage of the fact that this is an
        # involution
        try:
            return self._reflected
        except AttributeError:
            sl = tuple(slice(None, None, -1) for __ in range(self.ndim))
            self._reflected = type(self)(self.data[sl], -self.bset)
            self._reflected._reflected = self  # Involution at work
            return self._reflected

    def astype(self, dtype):
        """
        Return a copy of the IntensityMap instance cast to a new dtype

        Parameters
        ----------
        dtype : dtype
            Type to cast the `IntensityMap` instance to

        Returns
        -------
        IntensityMap
            A copy of `self`, cast to `dtype`.

        """
        return type(self)(self.data.astype(dtype), self.bset)

    @staticmethod
    def mean_map(maps, ignore_missing=True):
        """
        Compute the mean of a sequence of intensity maps

        Parameters
        ----------
        maps : iterable
            Sequence or iterable yielding maps to compute the mean over.
            A ValueError is raised if not all maps in the sequence are defined
            over equal-comparing BinnedSets.
        ignore_missing : bool, optional
            If True, missing values are ignored, and the mean value in a bin is
            the mean of all the non-missing values in this bin from `maps`. If
            False, any bin where at least one map in `maps` has a missing value
            will be have a missing value in the output.

        Returns
        -------
        IntensityMap
            Map of the mean of the intensities in `maps`.

        """
        maps = iter(maps)  # We accept iterators/generators
        ref_map = next(maps)
        ref_bset = ref_map.bset

        data = ref_map.data
        new_data = numpy.ma.filled(data, fill_value=0.0)
        num_valid = (~numpy.ma.getmaskarray(data)).astype(numpy.int_)
        length = 0  # Since iterators have no len(), we must sum it up manually
        for map_ in maps:  # No [1:]: the first element was consumed by next()
            if not (map_.bset == ref_bset):
                raise ValueError("instances of {} must be defined over "
                                 "instances of {} that compare equal for "
                                 "the mean to be defined"
                                 .format(ref_map.__class__.__name__,
                                         ref_bset.__class__.__name__))
            data = map_.data
            new_data += numpy.ma.filled(data, fill_value=0.0)
            num_valid += (~numpy.ma.getmaskarray(data)).astype(numpy.int_)
            length += 1

        new_data = sensibly_divide(new_data, num_valid, masked=True)
        if not ignore_missing:
            full_mask = (num_valid < length)
            new_data = numpy.ma.array(new_data, mask=full_mask, keep_mask=True)

        return type(ref_map)(new_data, ref_bset)

    def mean(self, weight_by_area=True):
        """
        Compute the mean intensity of the instance, ignoring missing values

        Returns
        -------
        scalar
            The mean intensity.
        weight_by_area : bool, optional
            If True, the contribution from each bin is weighted by the area
            (volume) of the bin.

        """
        if weight_by_area:
            return self.integral() / self.indicator.integral()
        else:
            return self.sum() / self.indicator.sum()

    def std(self, ddof=0, weight_by_area=True):
        """
        Compute the standard deviation of the intensity of the instance,
        ignoring missing values

        Parameters
        ----------
        ddof : int, optional
            Delta degrees of freedom: see the documentation for
            `IntensityMap.var` for details.
        weight_by_area : bool, optional
            If True, the contribution from each bin is weighted by the area
            (volume) of the bin.

        Returns
        -------
        scalar
            The standard deviation of the intensity.

        """
        return numpy.sqrt(self.var(ddof=ddof, weight_by_area=weight_by_area))

    def var(self, ddof=0, weight_by_area=True):
        """
        Compute the variance of the intensity of the instance, ignoring missing
        values

        Parameters
        ----------
        ddof : int, optional
            Delta degrees of freedom: the divisor used in calculating the
            variance is `N - ddof`, where `N` is the number of bins with values
            present. See the documentation for `numpy.var` for details.
        weight_by_area : bool, optional
            If True, the contribution from each bin is weighted by the area
            (volume) of the bin.

        Returns
        -------
        scalar
            The variance of the intensity.

        """
        mean = self.mean()
        dev = self - mean
        square = dev * dev
        n = self.count()
        if weight_by_area:
            return (n * square.integral() /
                    ((n - ddof) * self.indicator.integral()))
        else:
            return (n * square.sum() / ((n - ddof) * self.indicator.sum()))

    def min(self):
        """
        Find the minimum intensity of the instance, ignoring missing values

        Returns
        -------
        scalar
            The minimum intensity.

        """
        return numpy.ma.min(self.data)

    def max(self):
        """
        Find the maximum intensity of the instance, ignoring missing values

        Returns
        -------
        scalar
            The maximum intensity.

        """
        return numpy.ma.max(self.data)

    def sum(self, axis=None):
        """
        Find the sum of the intensity over the binned set, ignoring missing
        values

        Parameters
        ----------
        axis : None or int or tuple of ints, optional
            Axis or axes along which the sum is performed. If None, the
            sum is computed over all dimensions.

        Returns
        -------
        scalar or IntensityMap
            If axis is None, the value of the sum of the intensity over the
            bset is returned. Otherwise, the marginal summed map resulting from
            summing the intensity along the specified axes is returned.

        """
        if axis is None:
            return numpy.ma.sum(self.data)

        new_data = numpy.ma.sum(self.data, axis=axis)
        remaining_axes = numpy.setdiff1d(range(self.ndim), axis)
        remaining_edges = [self.bset.edges[ax] for ax in remaining_axes]

        # This is kind of a hack that breaks good OO design, but is there
        # a better solution?
        if len(remaining_edges) == 2:
            return IntensityMap2D(new_data, (remaining_edges,))
        else:
            return IntensityMap(new_data, (remaining_edges,))

    def integral(self, axis=None):
        """
        Find the integral of the intensity over the binned set, ignoring
        missing values

        Parameters
        ----------
        axis : None or int or tuple of ints, optional
            Axis or axes along which the integral is performed. If None, the
            integral is computed over all dimensions.

        Returns
        -------
        scalar or IntensityMap
            If axis is None, the value of the integral of the intensity over
            the bset is returned. Otherwise, the marginal intensity map
            resulting from integrating the intensity along the specified axes
            is returned.

        """
        if axis is None:
            return (self * self.bset.area).sum()

        try:
            measure = numpy.prod([self.bset.binwidths[ax] for ax in axis],
                                 axis=0)
        except TypeError:
            measure = self.bset.binwidths[axis]
        (self * measure).sum(axis=axis)

    @property
    def mask(self):
        """
        A boolean IntensityMap instance with value True in all bins where this
        instance has missing values

        """
        return type(self)(self.data.mask, self.bset)

    @property
    def indicator(self):
        """
        An integer IntensityMap instance with value 0 in all bins where this
        instance has missing values, and 1 otherwise

        """
        return (~self.mask).astype(numpy.int_)

    @property
    def ndim(self):
        """
        The dimensionality of this IntensityMap instance

        """
        return self.data.ndim

    @property
    def shape(self):
        """
        The number of elements in the IntensityMap instance along each axis

        """
        return self.data.shape

    @property
    def size(self):
        """
        The total number of elements in the IntensityMap instance

        """
        return self.data.size

    @property
    def dtype(self):
        """
        The dtype of the IntensityMap instance

        """
        return self.data.dtype

    @property
    def range_(self):
        """
        The range of values spanned by the underlying BinnedSet along each axis

        """
        return self.bset.range_

    @property
    def data(self):
        return self._data

    @property
    def unmasked_data(self):
        """
        A representation of the intensity data with missing values as nans
        instead of masked. This requires casting the data to float.

        """
        return numpy.ma.filled(self.data.astype(numpy.float_),
                               fill_value=numpy.nan)

    def count(self, axis=None):
        """
        Count the number of non-missing elements along the given axis

        Parameters
        ----------
        axis : integer, optional
            Axis along which to count non-missing elements

        Returns
        -------
        integer or ndarray
            If axis is None, the total count of non-missing elements.
            Otherwise, an array with counts along the given axis is returned.

        """
        return self.data.count(axis=axis)

    def filled(self, fill_value):
        """
        Create an IntensityMap instance from this instance, with missing values
        replaced by a fill value

        :fill_value: value to replace masked values with
        :returns: filled IntensityMap instance. This instance is returned
                  unchanged if there are no missing values.

        """
        sdata = self.data
        new_data = numpy.ma.filled(sdata, fill_value=fill_value)
        if new_data == sdata:
            return self
        else:
            return type(self)(new_data, self.bset)

    def fft(self):
        """
        Compute the Discrete Fourier Transform of this IntensityMap instance

        :returns: new IntensityMap instance representing the FFT of this
                  instance

        """
        new_bset = self.bset.fft()
        new_data = fftpack.fftn(self.unmasked_data)

        return type(self)(new_data, new_bset)

    def smooth(self, bandwidth, kernel='gaussian', normalize=False,
               periodic=False):
        """
        Return a new IntensityMap instance which is a smoothed version of this

        :bandwidth: smoothing filter bandwidth. This is a physical length
               in the same units as the lengths in the BinnedSet instance over
               which the IntensityMap is defined. The exact interpretation of
               'bandwidth' will vary with 'kernel'.
        :kernel: choice of smoothing filter kernel. Supported values:
            'gaussian': an isotropic gaussian filter with standard deviation
                        'bandwidth'
            'tophat': a (hyper)square box filter with standard deviation
                      'bandwidth' (that is, side lengths 'sqrt(12)
                      * bandwidth')
                 If a more general smoothing filter is needed, the user should
                 construct the desired kernel manually and use the
                 IntensityMap.convolve() method. Remember to take into account
                 the conversion between physical lengths in the units used by
                 the BinnedSet, and bin/pixel numbers,
                  Default is 'gaussian'.
        :normalize: if True, the smoothed IntensityMap is renormalized for
                     each bin to eliminate the influence of missing values and
                     values beyond the edges. Where only missing values would
                     have contributed, the resulting value is missing. If
                     False, values beyond the boundary are interpreted as 0.0,
                     and the presence of missing values will raise
                     a ValueError.
        periodic : bool, optional
            If True, periodic boundary conditions are assumed. Otherwise,
            values beyond the boundary are assumed to be 0.0 (the bias from
            this assumption can be compensated by setting `normalize=True`).
        :returns: new, smoothed IntensityMap instance defined over the same
                  BinnedSet instance as this

        """
        if not self.bset.regular:
            raise ValueError("a {} instance must be defined over a {} "
                             "instance which is regular (has bins of equal "
                             "size and shape) for kernel filter smoothing"
                             .format(self.__class__.__name__,
                                     self.bset.__class__.__name__))

        if bandwidth == 0.0:
            return self

        size = [bandwidth / numpy.mean(w) for w in self.bset.binwidths]

        if periodic:
            options = dict(mode='wrap')
        else:
            options = dict(mode='constant', cval=0.0)
        if kernel == 'gaussian':
            def smoothfunc(arr):
                return filters.gaussian_filter(arr, size, **options)
        elif kernel == 'tophat':
            # Multiply by sqrt(12)
            size = [2.0 * _SQRT3 * s for s in size]
            # Round bin size to nearest odd integer
            size = [2 * int(0.5 * s) + 1 for s in size]

            def smoothfunc(arr):
                return filters.uniform_filter(arr, size, **options)
        else:
            raise ValueError("unknown filter {}".format(kernel))

        new_data = _safe_mmap(normalize, smoothfunc, (self.data,))

        return type(self)(new_data, self.bset)

    def convolve(self, other, mode='full', normalize=False):
        """
        Compute the convolution of this and another IntensityMap instance

        :other: another IntensityMap instance. May also be a numpy array of
                appropriate dimensionality. In the latter case, the array will
                be taken as centered over the origin and given bin widths that
                are compatible with this instance (if possible).
        :mode: string indicating the size of the output. See
               scipy.signal.convolve for details. Valid options:
               'full', 'valid', 'same'
        :normalize: if True, the convolved IntensityMap is renormalized for
                     each bin to eliminate the influence of missing values and
                     values beyond the edges. Where only missing values would
                     have contributed, the resulting value is missing. If
                     False, values beyond the boundary are interpreted as 0.0,
                     and the presence of missing values will raise
                     a ValueError.
        :returns: new IntensityMap instance representing the convolution of
                  this and the other instance

        """
        sdata = self.data
        if isinstance(other, type(self)) or isinstance(self, type(other)):
            odata = other.data
        else:
            other = self.new_from_array(other)
            odata = other.data

        new_bset = self.bset.convolve(other.bset, mode=mode)

        def convfunc(arr1, arr2):
            return signal.convolve(arr1, arr2, mode=mode)

        new_data = _safe_mmap(normalize, convfunc, (sdata, odata))

        return type(self)(new_data, new_bset)

    def correlate(self, other, mode='full', pearson=None, normalize=False):
        """
        Compute the cross-correlogram of this and another IntensityMap instance

        Parameters
        ----------
        other : IntensityMap or array-like
            Intensity map to correlate with. If given as an array-like type,
            the array will be centered over the origin and given bin widths
            that are compatible with this IntensityMap (if possible).
        mode : {'full', 'valid', 'same'}, optional
            String indicating the size of the output. See
            `scipy.signal.convolve` for details.
        pearson : {None, 'global', 'local'}, optional
            If either `'global'` or `'local'`, each entry in the result is the
            Pearson correlation coefficient between the overlapping parts of
            the IntensityMap instances at the corresponding displacement. If
            `'global'`, the global means and standard deviations of the
            IntensityMaps are used to compute standard scores, while if
            `'local'`, new means and standard deviations are computed for the
            overlapping parts of the IntensityMaps at each displacement. If
            None, the plain (non-Pearson) correlation is computed.
        normalize : bool, optional
            If True, the correlated IntensityMap is renormalized for each bin
            to eliminate the influence of missing values and values beyond the
            edges. Where only missing values would have contributed, the
            resulting value is missing. If False, values beyond the boundary
            are interpreted as 0.0, and the presence of missing values will
            raise a ValueError.
            ..note:: Not applicable if `pearson == 'local'` -- the local
            Pearson correlation coefficient is implicitly normalized and
            handlies missing-value gracefully (see `utils.pearson_correlogram`
            for details).

        Returns
        -------
        IntensityMap
            IntensityMap instance representing the cross-correlogram of this
            and the other instance.

        """
        sdata = self.data
        if isinstance(other, type(self)) or isinstance(self, type(other)):
            odata = other.data
        else:
            other = self.new_from_array(other)
            odata = other.data

        new_bset = self.bset.correlate(other.bset, mode=mode)

        if pearson == 'local':
            new_data = pearson_correlogram(sdata, odata, mode=mode)

            # Remove the outer frame of nonsense
            n = self.ndim
            sl_all = slice(None)
            sl_firstlast = (0, -1)
            for axis in range(n):
                s = [sl_all] * n
                s[axis] = sl_firstlast
                new_data[s] = numpy.ma.masked
        else:
            if pearson == 'global':
                sdata = ((sdata - numpy.ma.mean(sdata)) /
                         (numpy.ma.std(sdata)))
                odata = ((odata - numpy.ma.mean(odata)) /
                         (numpy.ma.std(odata)))
            elif pearson is not None:
                raise ValueError("'pearson' must be either 'global', 'local', "
                                 "or None.")

            def corrfunc(arr1, arr2):
                max_overlap = [min(s1, s2)
                               for (s1, s2) in zip(arr1.shape, arr2.shape)]
                max_size_sq = numpy.sqrt(numpy.prod(max_overlap))
                return signal.correlate(arr1 / max_size_sq, arr2 / max_size_sq,
                                        mode=mode)

            new_data = _safe_mmap(normalize, corrfunc, (sdata, odata))

        return type(self)(new_data, new_bset)

    def autocorrelate(self, mode='full', pearson=None, normalize=False):
        """
        Compute the autocorrelogram of this IntensityMap instance

        This is a convenience wrapper for calling `self.correlate(self, ...)`.

        Parameters
        ----------
        mode, pearson, normalize
            See `IntensityMap.correlate`.

        Returns
        -------
        IntensityMap
            See `IntensityMap.correlate`.

        """
        return self.correlate(self, mode=mode, pearson=pearson,
                              normalize=normalize)

    def labels(self, threshold):
        """
        Label connected regions of intensities larger than a threshold

        This method is a simple wrapper around scipy.measurements.label()

        :threshold: lower bound defining the connected regions to be labeled
        :returns: array of labels, and the number of labeled regions

        """
        data = self.unmasked_data
        regions = (data > threshold)
        data_thres = numpy.zeros_like(data)
        data_thres[regions] = data[regions]
        labels, n = measurements.label(data_thres)
        return labels, n

    def peaks(self, threshold):
        """
        Detect peaks in the IntensityMap instance

        The algorithm calculates the center of mass in each connected region
        of intensities larger than the given threshold, and estimates the size
        of the support region of each peak by finding the area (volume) of the
        region and computing the radius of the circle (hyperball) with this
        area (volume).

        :threshold: lower intensity bound defining the regions in which peaks
                    are found. This value must be high enough that the supports
                    of different peaks become disjoint regions, but lower than
                    the peak intensities themselves.
        :returns:
            - an array with a row [x, y, r] for each detected peak, where x and
              y are the coordinates of the peak and r is an estimate of the
              radius of the elevated region around the peak
            - array of labels giving the connected region around the peaks,
              such that labels == i is an index to the region surrounding the
              ith peak detected (at index i - 1 in the returned array of peaks)
            - the number of peaks and labeled regions found

        """
        labels, n = self.labels(threshold)
        if n == 0:
            raise ValueError("no peaks found, try lowering 'threshold'")

        index = range(1, n + 1)
        # Weight the the value in each cell by the size of its bin to get more
        # precise peak positions
        weighted_data = self.unmasked_data * self.bset.area
        peak_list = measurements.center_of_mass(weighted_data, labels=labels,
                                                index=index)

        labeled_areas = [numpy.sum(self.bset.area * (labels == i))
                         for i in index]
        radii = _n_ball_rad(self.ndim, numpy.array(labeled_areas))

        peaks = numpy.column_stack((self.bset.coordinates(peak_list), radii))
        return peaks, labels, n

    def fit_gaussian(self, mask=None):
        """
        Fit a multidimensional Gaussian to a region in the IntensityMap
        instance

        A mask may be provided, such that only values in self.data[~mask]
        contribute to the fit. If self.data is already masked, the intrinsic
        and provided masks are combined. Any nan entries in self.data are also
        removed.

        In case anyone is wondering: the reason this method is not memoized is
        that the argument 'mask' usually takes a numpy array, which is not
        hashable.

        :mask: boolean array of the same shape as self.data, used to mask bins
               from contributing to the fit. Only the values in
               self.data[~mask] are used. This can for example be used to fit
               the Gaussian to one of the labeled regions returned from
               IntensityMap.labels() or IntensityMaps.peaks() by using
               ~(self.labels(threshold)[0] == label) as the mask. If None
               (default), the whole IntensityMap contributes.
        :returns: fitted parameters scale (scalar), mean (numpy array of shape
                  (n,)) and cov (numpy array of shape (n, n)) such that scale
                  * gaussian(x, mean=mean, cov=cov) returns the values of the
                  fitted Gaussian at the positions in x. Here, n == self.ndim.

        """
        data = self.data
        mask = numpy.logical_or(mask, numpy.ma.getmaskarray(data))
        fdata = data[~mask].data
        xdata = numpy.asarray([cm[~mask]
                              for cm in self.bset.cmesh]).transpose()
        scale, mean, cov = fit_ndgaussian(xdata, fdata)
        return scale, mean, cov

    def crop_missing(self):
        """
        Remove slices containing only missing values from the edges of an
        IntensityMap instance

        NOTE: This method is not properly tested!

        :returns: a new IntensityMap instance, identical to this except that
                  all-masked slices along edges are removed.

        """
        new_data = numpy.ma.copy(self.data)
        new_edges = list(self.bset.edges)  # Mutable copy

        # Remove all-masked edge slices along all dimensions
        for axis in range(new_data.ndim):
            # Bring axis to front
            new_data = numpy.ma.swapaxes(new_data, 0, axis)

            # Find first slice to keep
            try:
                first = next(i for (i, mask) in
                             enumerate(numpy.ma.getmaskarray(new_data))
                             if not mask.all())
                new_data = new_data[first:]
                new_edges[axis] = new_edges[axis][first:]
            except StopIteration:
                pass

            # Find last slice to keep
            try:
                last = next(i for (i, mask) in
                            enumerate(numpy.ma.getmaskarray(new_data)[::-1])
                            if not mask.all())
                if last != 0:
                    new_data = new_data[:-last]
                    new_edges[axis] = new_edges[axis][:-last]
            except StopIteration:
                pass

            # Swap back axis
            new_data = numpy.ma.swapaxes(new_data, 0, axis)

        return type(self)(new_data, new_edges)

    def rotate(self, angle, axes=(1, 0), reshape=False):
        """
        Compute a rotated version of the intensity map

        The map is rotated around its center.

        Parameters
        ----------
        angle : scalar
            Angle to rotate, in degrees.
        axes : sequence of 2 ints, optional
            Axes defining the plane of rotation.
        reshape : bool, optional
            If True, the returned intensity map is resized to fit the whole
            rotated map. If False, the returned map is of the same size
            as this (corners will be cropped unless `angle` is a multiple of
            90).

        Returns
        -------
        IntensityMap
            Rotated version of the intensity map.

        """
        rotated_data = interpolation.rotate(self.unmasked_data, angle,
                                            axes=axes, reshape=reshape,
                                            order=1, cval=numpy.nan)
        if reshape:
            extension = []
            for (rs, s) in zip(rotated_data.shape, self.shape):
                ext_total = rs - s
                ext_start = ext_total // 2
                ext_end = ext_total - ext_start
                extension.append((ext_start, ext_end))
            new_bset = self.bset.extend(extension)
        else:
            new_bset = self.bset

        return type(self)(rotated_data, new_bset)

    def shell(self, inner_radius, outer_radius):
        """
        Return an intensity map with values outside a given shell masked out

        Parameters
        ----------
        inner_radius, outer_radius : scalars
            Values with distance from the origin (defined by self.bset) between
            these radii are kept, while all others are masked out. If either is
            None, only the other is used.

        Returns
        -------
        IntensityMap
            IntensityMap containing only values within the specified shell.

        """
        dmesh = numpy.sqrt(numpy.sum([cm * cm for cm in self.bset.cmesh],
                                     axis=0))
        mask = False
        if inner_radius is not None:
            mask = dmesh < inner_radius
        if outer_radius is not None:
            mask = numpy.logical_or(mask, dmesh >= outer_radius)
        shell_data = numpy.ma.array(self.data, mask=mask, keep_mask=True)
        return type(self)(shell_data, self.bset)


class IntensityMap2D(IntensityMap):
    """
    A specialization of the IntensityMap to two-dimensional rectangles.
    Features a simplified call signature, 2D-specific properties, and
    a plotting method.

    Parameters
    ----------
    data : array-like
        A 2D array of scalar intensities. Nan-valued and/or masked elements
        (using the `numpy.ma` module) may be used to represent missing data.
        See the documentation for `IntensityMap` for information.
    bset : BinnedSet2D or sequence
        A `BinnedSet2D` instance, or a valid argument list to the `BinnedSet2D`
        constructor, defining the spatial bins where the `IntensityMap`
        instance takes values. The ordering of axes in `bset` and `data` should
        correspond, such that `data[i, j]` gives the intensity in
        the bin defined by
        ```
        (bset.edges[0][i], bset.edges[0][i + 1],
         bset[1][j], bset[1][j + 1])
        ```
    **kwargs : dict, optional
        Any other keyword arguments are passed to the `numpy.ma.array`
        constructor along with `numpy.ma.masked_where(numpy.isnan(data),
        data)`.

    """
    def __init__(self, data, bset, **kwargs):
        # Need to explicitly make sure that we get a BinnedSet2D instance
        # before invoking the parent constructor
        if not isinstance(bset, BinnedSet2D):
            bset = BinnedSet2D(*bset)

        IntensityMap.__init__(self, data, bset, **kwargs)

    def blobs(self, min_sigma=None, max_sigma=None, num_sigma=25,
              threshold=-numpy.inf, max_overlap=0.0, log_scale=False,
              ignore_missing=False, exclude_edges=0):
        """
        Detect blobs in the IntensityMap2D instance

        An alternative to self.peaks() for detecting elevated regions in
        IntensityMap2D instances.

        This method is essentially a wrapper around the Laplacian of Gaussian
        blob detection function from scikit-image, which identifies centers and
        scales of blobs in arrays. Some (hopefully sensible) default parameters
        are provided. Note that this implementation of the Laplacian of
        Gaussian algorithm is only appropriate if the underlying BinnedSet
        instance is square, and the method will raise an error if this is not
        the case.

        :min_sigma: minimum standard deviation of the Gaussian kernel used in
                    the Gaussian-Laplace filter, given in units given by the
                    underlying BinnedSet2D (NOT in bins/pixels, unlike
                    skimage.feature.blob_log()). The method will not detect
                    blobs with radius smaller than approximately \sqrt{2}
                    * min_sigma. If None (default), a default value
                    corresponding to a blob radius of about one twelfth of the
                    shortest width of the intensity map will be used.
        :max_sigma: maximum standard deviation of the Gaussian kernel used in
                    the Gaussian-Laplace filter, given in the units used by the
                    underlying BinnedSet2D (NOT in bins/pixels, unlike
                    skimage.feature.blob_log()). The method will not detect
                    blobs with radius larger than approximately \sqrt{2}
                    * max_sigma. If None (default), a default value
                    corresponding to a blob radius of about one fourth of the
                    shortest width of the intensity map will be used.
        :num_sigma: the number of intermediate filter scales to consider
                    between min_sigma and max_sigma. Increase for improved
                    precision, decrease to reduce computational cost. Note that
                    for coarse-binned intensity maps, precision will often be
                    limited by resolution rather than this parameter. Default
                    value: 25.
        :threshold: lower bound for scale-space maxima (that is, the minimum
                    height of detected peaks in the arrays created by passing
                    the intensity array through a Gaussian-Laplace filter).
        max_overlap : scalar, optional
            Maximum permitted overlap fraction. If two blobs overlap by more
            than this, only the larger of two overlapping blobs are kept.
            Overlap is defined as the ratio of the area of the intersection and
            uunion of the blobs, using `sqrt(2) * s` as the blob radius for
            a blob detected at sigma `s`.
        :log_scale: if True, intermediate sigma values are chosen with regular
                    spacing on a logarithmic scale betweeen min_sigma
                    and max_sigma. This way, the errors in estimated blob
                    centers and sizes will scale approximately with the blob
                    sizes. If False, the intermediate sigmas are chosen with
                    regular spacing on a linear scale, and the errors will be
                    approximately constant. Default is True.
        :ignore_missing: by default, blob detection fails for intensity maps
                         with missing values. When this parameter is set to
                         True,
                         #this limitation is overcome by applying a normalized
                         #smoothing filter before blob detection.  The
                         #smoothing filter is just wide enough to extrapolate
                         #into one row of missing values while barely affecting
                         #the other intensities, and will be applied repeatedly
                         #until all missing values have been replaced.
                         missing values are replaced by zeroes.
        exclude_edges : integer, optional
            Number of rows along the edges in which to disallow blob centers.
        :returns: an array with a row [x, y, s] for each detected blob, where
                  x and y are the coordinates of the blob center and s is the
                  sigma (standard deviation) of the Gaussian kernel that
                  detected the peak. Once, again, this sigma value is given the
                  units used by the underlying BinnedSet2D (NOT in bins/pixels,
                  unlike skimage.feature.blob_log()). An estimate of the radius
                  of the blob can be obtained by taking \sqrt{2} s.

        """
        if not self.bset.square:
            raise ValueError("instances of {} must be defined over instances "
                             "of {} with square bins for Laplacian of "
                             "Gaussian blob detection"
                             .format(self.__class__.__name__,
                                     self.bset.__class__.__name__))

        data = numpy.copy(self.unmasked_data)

        if ignore_missing:
            #bandwidth = 0.125 * numpy.amax(self.bset.binwidths)
            #filled = self
            #while numpy.isnan(data).any():
            #    filled = filled.smooth(bandwidth, kernel='gaussian',
            #                           normalize=True)
            #    data = filled.unmasked_data
            data[numpy.isnan(data)] = 0.0
        elif numpy.isnan(data).any():
            raise ValueError("cannot detect blobs in IntensityMap2D instances "
                             "with missing data unless ignore_missing == "
                             "True.")

        #data = exposure.equalize_hist(data)  # Improves detection

        # In case of different binwidths in different directions, use the
        # geometric mean
        binwidth = numpy.sqrt(numpy.prod(
            numpy.mean(self.bset.binwidths, axis=-1)))

        if min_sigma is None:
            min_sigma = min(self.shape) / (_SQRT2 * 12.0)
        else:
            min_sigma /= binwidth
        if max_sigma is None:
            # Taking the blob radius as sqrt(2) * sigma, setting max_sigma to
            # min(self.shape) / (sqrt(2) * 2.0 * k)
            # means limiting the largest possible blobs to regions covering at
            # most a fraction 1 / k of the shortest length in the intensity map
            max_sigma = min(self.shape) / (_SQRT2 * 4.0)
        else:
            max_sigma /= binwidth

        blob_indices = feature.blob_log(data, min_sigma=min_sigma,
                                        max_sigma=max_sigma,
                                        num_sigma=num_sigma,
                                        threshold=threshold,
                                        overlap=1.0,
                                        log_scale=log_scale)

        # Sort blobs in order of decreasing blob size
        try:
            blob_indices = blob_indices[
                numpy.argsort(blob_indices[:, 2])[::-1]]
        except IndexError:
            blob_indices = numpy.array([[]])

        if exclude_edges > 0:
            n, m = data.shape
            edge = numpy.logical_or(
                numpy.logical_or(
                    numpy.logical_or(
                        blob_indices[:, 0] < exclude_edges,
                        n - 1 - blob_indices[:, 0] < exclude_edges,
                    ),
                    blob_indices[:, 1] < exclude_edges,
                ),
                m - 1 - blob_indices[:, 1] < exclude_edges,
            )
            blob_indices = blob_indices[~edge]

        if max_overlap < 1.0:
            prune = numpy.zeros((len(blob_indices), ), dtype=numpy.bool_)
            for (i, b1) in enumerate(blob_indices):
                if prune[i]:
                    continue
                blob1 = (b1[0], b1[1], _SQRT2 * b1[2])
                for (j_, b2) in enumerate(blob_indices[i + 1:]):
                    j = j_ + i + 1
                    if prune[j]:
                        continue
                    blob2 = (b2[0], b2[1], _SQRT2 * b2[2])
                    overlap = disc_overlap(blob1, blob2)
                    if overlap > max_overlap:
                        prune[j] = True
            blob_indices = blob_indices[~prune]

        blob_list = blob_indices[:, :2]

        blob_coords = self.bset.coordinates(blob_list)

        sigma = blob_indices[:, 2] * binwidth

        blobs = numpy.column_stack((blob_coords, sigma))

        # Sort blobs: first by x coordinate, then by y coordinate, then by size
        blobs = blobs[numpy.argsort(blobs[:, 2])]
        blobs = blobs[numpy.argsort(blobs[:, 1])]
        blobs = blobs[numpy.argsort(blobs[:, 0])]

        return blobs

    def plot(self, axes=None, cax=None, threshold=None, vmin=None, vmax=None,
             cmap=None, cbar=True, cbar_kw=None, **kwargs):
        """
        Plot the IntensityMap

        The map can be added to an existing plot via the optional 'axes'
        argument.

        :axes: Axes instance to add the intensity map to. If None (default),
               a new Figure is created (this method never plots to the current
               Figure or Axes). In the latter case, equal aspect ratio will be
               enforced on the newly created Axes instance.
        cax : Axes, optional
            Axes instance to add the colorbar to. If None (default), matplotlib
            automatically makes space for a colorbar on the right-hand side of
            the plot.
        :threshold: if not None, values below this threshold are masked from
                    the plot. This may be useful to visualize regions
                    surrounding peaks.
        :vmin, vmax: scaling parameters for mapping the intensity map onto the
                     colormap. An intensity smaller than or equal to vmin will
                     be mapped to the lowest value in the colormap, while an
                     intensity of greater than or equal to vmax will be mapped
                     to the highest value in the colormap. If None (default),
                     the most extreme values in the intensity map will be used.
        cmap : Colormap or registered colormap name, optional
            Colormap to use for the plot. If None (default), the default
            colormap from rcParams is used.
            ..note:: The default map might be 'jet', and this is something you
            certainly DON'T WANT to use! If you're clueless, try 'YlGnBu_r' or
            'gray'.
        cbar : bool, optional
            If True, add colorbar. If False, don't.
        :cbar_kw: dict of keyword arguments to pass to the pyplot.colorbar()
                  function. Default: None (empty dict)
        :kwargs: additional keyword arguments passed on to axes.pcolormesh()
        :returns: the axes instance containing the plot, and the colorbar
                  instance (if plotted).

        """
        if axes is None:
            fig, axes = pyplot.subplots(subplot_kw={'aspect': 'equal'})
        ret = [axes]

        x, y = self.bset.mesh

        data = self.data
        if threshold is not None:
            data = numpy.ma.masked_where(data <= threshold, data)

        mesh = axes.pcolormesh(x, y, data, vmin=vmin, vmax=vmax, cmap=cmap,
                               **kwargs)
        if cbar:
            if cbar_kw is None:
                cbar_kw = {}
            ret.append(pyplot.colorbar(mesh, cax=cax, **cbar_kw))

        axes.set(xlim=self.bset.xedges[(0, -1), ],
                 ylim=self.bset.yedges[(0, -1), ])

        return ret

    def rotate(self, angle, reshape=False):
        """
        Compute a rotated version of the intensity map

        The map is rotated around its center.

        Parameters
        ----------
        angle : scalar
            Angle to rotate, in degrees.
        reshape : bool, optional
            If True, the returned intensity map is resized to fit the whole
            rotated map. If False, the returned map is of the same size
            as this (corners will be cropped unless `angle` is a multiple of
            90).

        Returns
        -------
        IntensityMap
            Rotated version of the intensity map.

        """
        return IntensityMap.rotate(self, angle, reshape=reshape)


def _safe_mmap(normalize, mapfunc, arrs):
    """
    Safely compute a general multilinear map (e.g. filtering, convolution) over
    any number of arrays, optionally normalizing the result to eliminate the
    effect of missing values

    In this context, safely means that the presence of nans or masked values in
    any of the input arrays will cause an exception to be raised, unless
    normalize == True, in which case nans and masked values are treated as
    missing values.

    :normalize: if True, the output of the map is normalized at each element
                 to eliminate the contribution from nans, masked values and
                 values beyond the edges of the array. Where only such values
                 would have contributed, the resulting value is nan. If False,
                 the map is applied without normalization, and the presence of
                 nans or masked values in any of the arrays will raise
                 a ValueError.
    :mapfunc: callable defining the multilinear map: mapfunc(arrs) returns the
              unnormalized mapping
    :arrs: argument list to `mapfunc`, containing the arrays to compute the map
           over
    :returns: result of the mapping, optionally normalized. If normalized, the
              result is masked where only missing values would have
              contributed.

    """
    filled_arrs = []
    indicators = []
    ones = []
    for arr in arrs:
        masked_arr = numpy.ma.masked_where(numpy.isnan(arr), arr)
        mask = numpy.ma.getmaskarray(masked_arr)
        ind = (~mask).astype(numpy.float_)
        indicators.append(ind)
        ones.append(numpy.ones_like(ind))
        filled_arrs.append(numpy.ma.filled(masked_arr, fill_value=0.0))

    if normalize:
        perfect_weights = mapfunc(*ones)
        weights = mapfunc(*indicators)
        weights /= numpy.max(perfect_weights)
        new_arr = sensibly_divide(mapfunc(*filled_arrs), weights, masked=True)
    else:
        if any(numpy.any(ind == 0.0) for ind in indicators):
            raise ValueError("cannot filter IntensityMap instances with "
                             "masked values or nans unless normalize == True")
        new_arr = mapfunc(*arrs)

    return new_arr


def _n_ball_vol(n, rad):
    """
    Compute the volume of an n-ball given its radius

    :returns: volume

    """
    n_2 = n / 2
    unitvol = (numpy.pi ** n_2) / special.gamma(n_2 + 1)
    vol = unitvol * (rad ** n)
    return vol


def _n_ball_rad(n, vol):
    """
    Compute the radius of an n-ball given its volume

    :returns: radius

    """
    unitvol = _n_ball_vol(n, 1)
    radius = (vol / unitvol) ** (1.0 / n)
    return radius
