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
from scipy.ndimage import filters, measurements
from skimage import feature  # , exposure
from matplotlib import pyplot

from .utils import AlmostImmutable, sensibly_divide, pearson_correlogram
from .memoize.memoize import memoize_method
from .ndfit import fit_ndgaussian


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

        return cls.from_center_binwidth_shape(cls, center, binwidth, shape)

    def __eq__(self, other):
        """
        Define equality as the equality of all bin edge arrays

        :other: another BinnedSet instance
        :returns: True if equal, otherwise False

        """
        if other is self:
            return True

        sedges, oedges = self.edges, other.edges
        return ((len(sedges) == len(oedges)) and
                all(numpy.all(se == oe) for (se, oe) in zip(sedges, oedges)))

    def __ne__(self, other):
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
    @memoize_method
    def shape(self):
        """
        The number of bins in the BinnedSet instance along each axis

        """
        return tuple(e.size - 1 for e in self.edges)

    @property
    @memoize_method
    def range_(self):
        """
        The range of values spanned by the BinnedSet along each axis

        """
        return tuple((e[0], e[-1]) for e in self.edges)

    @property
    @memoize_method
    def centers(self):
        """
        Tuple of arrays giving the bin center values

        """
        return tuple(0.5 * (e[1:] + e[:-1]) for e in self.edges)

    @property
    @memoize_method
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
    @memoize_method
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
    @memoize_method
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
    @memoize_method
    def regular(self):
        """
        Whether this BinnedSet instance is regular (has bins that are equal in
        size and shape)

        """
        return all(numpy.allclose(w, w[0]) for w in self.binwidths)

    @property
    @memoize_method
    def square(self):
        """
        Whether the bins defined by this BinnedSet instance are square, or in
        general hypercubical

        """
        return self.regular and numpy.allclose(*(w[0] for w in self.binwidths))

    @memoize_method
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

    @memoize_method
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
        return self.__class__(new_edges)

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
            self._reflected = self.__class__(new_edges)
            self._reflected._reflected = self  # Involution at work
            return self._reflected

    @memoize_method
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
                             "FFT-compatible instace"
                             .format(self.__class__.__name__))

        center, fbinwidth, shape = [], [], []
        for (sw, se) in zip(self.binwidths, self.edges):
            nb = se.size - 1
            fbw = 1 / (numpy.mean(sw) * nb)
            center.append(0.5 * fbw * ((nb % 2) - 1))
            fbinwidth.append(fbw)
            shape.append(nb)

        return self.from_center_binwidth_shape(center, fbinwidth, shape)

    @memoize_method
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

        It is assumed that the given indices refer to the bin centers, NOT the
        bin edges.

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
        `normalized == True`), other methods are not designed with this in mind
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
        self.data = numpy.ma.array(data, mask=numpy.isnan(data),
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
        new_bset = self.bset.__class__.from_center_binwidth_shape(center,
                                                                  binwidth,
                                                                  shape)

        return self.__class__(arr, new_bset)

    def _inherit_binary_operation(self, other, op):
        """
        Define the general pattern for inheriting a binary operation on the
        data as a binary operation on the IntensityMap

        Parameters
        ----------
        other : array-like
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
        try:
            bound_op = getattr(sdata, op)
        except TypeError:
            def bound_op(odata):
                return op(sdata, odata)

        bset = self.bset
        try:
            obset = other.bset
        except AttributeError:
            # Apparently, other is not an IntensityMap
            new_data = bound_op(other)
        else:
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

        return self.__class__(new_data, bset)

    def __eq__(self, other):
        return self._inherit_binary_operation(other, '__eq__')

    def __ne__(self, other):
        return self._inherit_binary_operation(other, '__ne__')

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
    #    return self.__class__(self.data.__neg__(), self.bset)

    def __pos__(self):
        return self.__class__(self.data.__pos__(), self.bset)

    def __abs__(self):
        return self.__class__(self.data.__abs__(), self.bset)

    def __invert__(self):
        return self.__class__(self.data.__invert__(), self.bset)

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
            self._neg = self.__class__(self.data.__neg__(), self.bset)
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
            self._reflected = self.__class__(self.data[sl], -self.bset)
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
        return self.__class__(self.data.astype(dtype), self.bset)

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
        ref_map = maps.next()
        ref_bset = ref_map.bset

        data = ref_map.data
        new_data = numpy.ma.filled(data, fill_value=0.0)
        num_valid = (~numpy.ma.getmaskarray(data)).astype(numpy.int_)
        length = 0  # Since iterators have no len(), we must sum it up manually
        for map_ in maps:  # No [1:]: the first element was consumed by .next()
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

        return ref_map.__class__(new_data, ref_bset)

    @memoize_method
    def mean(self):
        """
        Compute the mean intensity of the instance, ignoring missing values

        The contribution each bin is correctly weighted by the area (volume) of
        the bin.

        Returns
        -------
        scalar
            The mean intensity.

        """
        return self.integral() / self.indicator.integral()

    @memoize_method
    def std(self, ddof=0):
        """
        Compute the standard deviation of the intensity of the instance,
        ignoring missing values

        The contribution each bin is correctly weighted by the area (volume) of
        the bin.

        Parameters
        ----------
        ddof : int, optional
            Delta degrees of freedom: see the documentation for
            `IntensityMap.var` for details.

        Returns
        -------
        scalar
            The standard deviation of the intensity.

        """
        return numpy.sqrt(self.var(ddof=ddof))

    @memoize_method
    def var(self, ddof=0):
        """
        Compute the variance of the intensity of the instance, ignoring missing
        values

        The contribution each bin is correctly weighted by the area (volume) of
        the bin.

        Parameters
        ----------
        ddof : int, optional
            Delta degrees of freedom: the divisor used in calculating the
            variance is `N - ddof`, where `N` is the number of bins with values
            present. See the documentation for `numpy.var` for details.

        Returns
        -------
        scalar
            The variance of the intensity.

        """
        mean = self.mean()
        dev = self - mean
        square = dev * dev
        n = self.count()
        return n * square.integral() / ((n - ddof) * self.indicator.integral())

    @memoize_method
    def min(self):
        """
        Find the minimum intensity of the instance, ignoring missing values

        Returns
        -------
        scalar
            The minimum intensity.

        """
        return numpy.ma.min(self.data)

    @memoize_method
    def max(self):
        """
        Find the maximum intensity of the instance, ignoring missing values

        Returns
        -------
        scalar
            The maximum intensity.

        """
        return numpy.ma.max(self.data)

    @memoize_method
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
            return numpy.ma.sum(self.data * self.bset.area)

        try:
            measure = numpy.prod([self.bset.binwidths[ax] for ax in axis],
                                 axis=0)
        except TypeError:
            measure = self.bset.binwidths[axis]
        new_data = numpy.ma.sum(self.data * measure, axis=axis)
        remaining_axes = numpy.setdiff1d(range(self.ndim), axis)
        remaining_edges = [self.bset.edges[ax] for ax in remaining_axes]

        # This is kind of a hack that breaks good OO design, but is there
        # a better solution?
        if len(remaining_edges) == 2:
            return IntensityMap2D(new_data, (remaining_edges,))
        else:
            return IntensityMap(new_data, (remaining_edges,))

    @property
    @memoize_method
    def mask(self):
        """
        A boolean IntensityMap instance with value True in all bins where this
        instance has missing values

        """
        return self.__class__(self.data.mask, self.bset)

    @property
    @memoize_method
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
    @memoize_method
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

    @memoize_method
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
            return self.__class__(new_data, self.bset)

    @memoize_method
    def fft(self):
        """
        Compute the Discrete Fourier Transform of this IntensityMap instance

        :returns: new IntensityMap instance representing the FFT of this
                  instance

        """
        new_bset = self.bset.fft()
        new_data = fftpack.fftn(self.unmasked_data)

        return self.__class__(new_data, new_bset)

    @memoize_method
    def smoothed(self, size, filter_='gaussian', normalized=False):
        """
        Return a new IntensityMap instance which is a smoothed version of this

        :size: characteristic smoothing length. This is a physical length
               in the same units as the lengths in the BinnedSet instance over
               which the IntensityMap is defined. The exact interpretation of
               'size' will vary with 'filter_'.
        :filter_: choice of smoothing filter. Supported values:
            'gaussian': an isotropic gaussian filter with standard deviation
                        'size'
            'uniform': a (hyper)square box filter with side lengths 'size'
                 If a more general smoothing filter is needed, the user should
                 construct the desired kernel manually and use the
                 IntensityMap.convolve() method. Remember to take into account
                 the conversion between physical lengths in the units used by
                 the BinnedSet, and bin/pixel numbers,
                  Default is 'gaussian'.
        :normalized: if True, the smoothed IntensityMap is renormalized for
                     each bin to eliminate the influence of missing values and
                     values beyond the edges. Where only missing values would
                     have contributed, the resulting value is missing. If
                     False, values beyond the boundary are interpreted as 0.0,
                     and the presence of missing values will raise
                     a ValueError.
        :returns: new, smoothed IntensityMap instance defined over the same
                  BinnedSet instance as this

        """
        if not self.bset.regular:
            raise ValueError("a {} instance must be defined over a {} "
                             "instance which is regular (has bins of equal "
                             "size and shape) for kernel filter smoothing"
                             .format(self.__class__.__name__,
                                     self.bset.__class__.__name__))

        size_b = [size / numpy.mean(w) for w in self.bset.binwidths]

        options = {'mode': 'constant', 'cval': 0.0}
        if filter_ == 'gaussian':
            def smoothfunc(arr):
                return filters.gaussian_filter(arr, size_b, **options)
        elif filter_ == 'uniform':
            def smoothfunc(arr):
                return filters.uniform_filter(arr, size_b, **options)
        else:
            raise ValueError("unknown filter {}".format(filter_))

        new_data = _safe_mmap(normalized, smoothfunc, (self.data,))

        return self.__class__(new_data, self.bset)

    @memoize_method
    def convolve(self, other, mode='full', normalized=False):
        """
        Compute the convolution of this and another IntensityMap instance

        :other: another IntensityMap instance. May also be a numpy array of
                appropriate dimensionality. In the latter case, the array will
                be taken as centered over the origin and given bin widths that
                are compatible with this instance (if possible).
        :mode: string indicating the size of the output. See
               scipy.signal.convolve for details. Valid options:
               'full', 'valid', 'same'
        :normalized: if True, the convolved IntensityMap is renormalized for
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
        try:
            odata = other.data
        except AttributeError:
            other = self.new_from_array(other)
            odata = other.data

        new_bset = self.bset.convolve(other.bset, mode=mode)

        def convfunc(arr1, arr2):
            return signal.convolve(arr1, arr2, mode=mode)

        new_data = _safe_mmap(normalized, convfunc, (sdata, odata))

        return self.__class__(new_data, new_bset)

    @memoize_method
    def correlate(self, other, mode='full', pearson=True, normalized=False):
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
        pearson : bool, optional
            If True, each entry in the result is the Pearson correlation
            coefficient between the overlapping parts of the IntensityMap
            instances at the corresponding displacement. See
            `utils.pearson_correlogram` for details. If False, the result
            contains plain cross-correlations.
        normalized : bool, optional
            If True, the correlated IntensityMap is renormalized for each bin
            to eliminate the influence of missing values and values beyond the
            edges. Where only missing values would have contributed, the
            resulting value is missing. If False, values beyond the boundary
            are interpreted as 0.0, and the presence of missing values will
            raise a ValueError.
            ..note:: Not applicable if `pearson == True` -- the Pearson
            correlation coefficient is by definition normalized (albeit in
            a slightly different way), and can handle missing values
            gracefully.

        Returns
        -------
        IntensityMap
            IntensityMap instance representing the cross-correlogram of this
            and the other instance.

        """
        sdata = self.data
        try:
            odata = other.data
        except AttributeError:
            other = self.new_from_array(other)
            odata = other.data

        new_bset = self.bset.correlate(other.bset, mode=mode)

        if pearson:
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
            def corrfunc(arr1, arr2):
                return signal.correlate(arr1, arr2, mode=mode)

            new_data = _safe_mmap(normalized, corrfunc, (sdata, odata))

        return self.__class__(new_data, new_bset)

    def autocorrelate(self, mode='full', pearson=True, normalized=False):
        """
        Compute the autocorrelogram of this IntensityMap instance

        Convenience method for calling self.correlate(self, mode=mode)

        :mode, pearson, normalized: see self.correlate()
        :returns: new IntensityMap instance representing the autocorrelogram of
                  this instance

        """
        return self.correlate(self, mode=mode, pearson=pearson,
                              normalized=normalized)

    @memoize_method
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

    @memoize_method
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

    @memoize_method
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

        return self.__class__(new_data, new_edges)


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

    @memoize_method
    def blobs(self, min_sigma=None, max_sigma=None, num_sigma=25,
              threshold=None, overlap=0.5, log_scale=True,
              ignore_missing=False):
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
                    corresponding to the width of a single bin/pixel will be
                    used.
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
                    the intensity array through a Gaussian-Laplace filter). If
                    None (default), the value self.data.min() + 0.2
                    * self.data.ptp() will be used.
        :overlap: maximum fraction of overlap of two detected blobs.
                  If any two blobs overlap by a greater fraction than this, the
                  smaller blob is discarded. Default value: 0.5.
        :log_scale: if True, intermediate sigma values are chosen with regular
                    spacing on a logarithmic scale betweeen min_sigma
                    and max_sigma. This way, the errors in estimated blob
                    centers and sizes will scale approximately with the blob
                    sizes. If False, the intermediate sigmas are chosen with
                    regular spacing on a linear scale, and the errors will be
                    approximately constant. Default is True.
        :ignore_missing: by default, blob detection fails for intensity maps
                         with missing values. When this parameter is set to
                         True, however, this limitation is overcome by applying
                         a normalized smoothing filter before blob detection.
                         The smoothing filter is just wide enough to fill one
                         row of missing values while barely affecting the other
                         intensities, and will be applied repeatedly until all
                         missing values have been replaced.
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

        data = self.unmasked_data
        if ignore_missing:
            size = 0.125 * numpy.amax(self.bset.binwidths)
            filled = self
            while numpy.isnan(data).any():
                filled = filled.smoothed(size, filter_='gaussian',
                                         normalized=True)
                data = filled.data
        elif numpy.isnan(data).any():
            raise ValueError("cannot detect blobs in IntensityMap2D instances "
                             "with missing data unless ignore_missing == "
                             "True.")

        #data = exposure.equalize_hist(data)  # Improves detection

        binwidth = numpy.mean(self.bset.binwidths)

        if min_sigma is None:
            min_sigma = 1
        else:
            min_sigma /= binwidth
        if max_sigma is None:
            # Taking the blob radius as sqrt(2) * sigma, setting max_sigma to
            # min(self.shape) / (sqrt(2) * 2.0 * k)
            # means limiting the largest possible blobs to regions covering at
            # most a fraction 1 / k of the shortest length in the intensity map
            max_sigma = min(self.shape) / (numpy.sqrt(2) * 4.0)
        else:
            max_sigma /= binwidth
        if threshold is None:
            threshold = data.min() + 0.2 * data.ptp()

        blob_indices = feature.blob_log(data, min_sigma=min_sigma,
                                        max_sigma=max_sigma,
                                        num_sigma=num_sigma,
                                        threshold=threshold,
                                        overlap=overlap,
                                        log_scale=log_scale)
        try:
            blob_list = blob_indices[:, :2]
        except IndexError:
            raise ValueError("no blobs found, try lowering 'threshold'")

        blob_coords = self.bset.coordinates(blob_list)

        sigma = blob_indices[:, 2] * binwidth

        blobs = numpy.column_stack((blob_coords, sigma))

        return blobs

    def plot(self, axes=None, cax=None, threshold=None, vmin=None, vmax=None,
             cmap=None, cbar_kw=None, **kwargs):
        """
        Plot the IntensityMap

        The map can be added to an existing plot via the optional 'axes'
        argument.

        :axes: Axes instance to add the intensity map to. If None (default),
               a new Figure is created (this method never plots to the current
               Figure or Axes). In the latter case, equal aspect ration will be
               enforced on the newly created Axes instance.
        :cax: Axes instance to plot the colorbar into. If None (default),
              matplotlib automatically makes space for a colorbar on the
              right-hand side of the plot.
        :threshold: if not None, values below this threshold are masked from
                    the plot. This may be useful to visualize regions
                    surrounding peaks.
        :vmin, vmax: scaling parameters for mapping the intensity map onto the
                     colormap. An intensity smaller than or equal to vmin will
                     be mapped to the lowest value in the colormap, while an
                     intensity of greater than or equal to vmax will be mapped
                     to the highest value in the colormap. If None (default),
                     the most extreme values in the intensity map will be used.
        :cmap: colormap to use for the plot. All valid matplotlib colormap
               arguments can be used. If None (default), the default colormap
               from rcParams is used (BEWARE: the default map might be 'jet',
               and this is something you certainly DON'T WANT to use! If you're
               clueless, try "YlGnBu_r" or "gray").
        :cbar_kw: dict of keyword arguments to pass to the pyplot.colorbar()
                  function. Default: None (empty dict)
        :kwargs: additional keyword arguments passed on to axes.pcolormesh()
        :returns: the axes instance containing the plot, and the colorbar
                  instance

        """
        if axes is None:
            fig, axes = pyplot.subplots(subplot_kw={'aspect': 'equal'})

        if cbar_kw is None:
            cbar_kw = {}

        x, y = self.bset.mesh

        data = self.data
        if threshold is not None:
            data = numpy.ma.masked_where(data <= threshold, data)

        mesh = axes.pcolormesh(x, y, data, vmin=vmin, vmax=vmax, cmap=cmap,
                               **kwargs)
        cbar = pyplot.colorbar(mesh, cax=cax, **cbar_kw)
        axes.set(xlim=self.bset.xedges[(0, -1), ],
                 ylim=self.bset.yedges[(0, -1), ])

        return axes, cbar


def _safe_mmap(normalized, mapfunc, arrs):
    """
    Safely compute a general multilinear map (e.g. filtering, convolution) over
    any number of arrays, optionally normalizing the result to eliminate the
    effect of missing values

    In this context, safely means that the presence of nans or masked values in
    any of the input arrays will cause an exception to be raised, unless
    normalized == True, in which case nans and masked values are treated as
    missing values.

    :normalized: if True, the output of the map is normalized at each element
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
    for arr in arrs:
        masked_arr = numpy.ma.masked_where(numpy.isnan(arr), arr)
        mask = numpy.ma.getmaskarray(masked_arr)
        indicators.append((~mask).astype(numpy.float_))
        filled_arrs.append(numpy.ma.filled(masked_arr, fill_value=0.0))

    if normalized:
        new_arr = sensibly_divide(mapfunc(*filled_arrs), mapfunc(*indicators),
                                  masked=True)
    else:
        if any(numpy.any(ind == 0.0) for ind in indicators):
            raise ValueError("cannot filter IntensityMap instances with "
                             "masked values or nans unless normalized == True")
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
