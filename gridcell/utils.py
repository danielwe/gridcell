#!/usr/bin/env python

"""File: utils.py
Module defining classes and functions that may come in handy throughout the
package

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
from scipy import stats
from scipy.ndimage import measurements
from sklearn.neighbors import KernelDensity
from sklearn.grid_search import GridSearchCV
import seaborn
from matplotlib import pyplot


class AlmostImmutable(object):
    """
    A base class for "almost immutable" objects: instance attributes that have
    already been assigned cannot (easily) be reassigned or deleted, but
    creating new attributes is allowed.

    """

    def __setattr__(self, name, value):
        """
        Override the __setattr__() method to avoid member reassigment

        """
        if hasattr(self, name):
            raise TypeError("{} instances do not support attribute "
                            "reassignment".format(self.__class__.__name__))
        object.__setattr__(self, name, value)

    def __delattr__(self, name):
        """
        Override the __detattr__() method to avoid member deletion

        """
        raise TypeError("{} instances do not support attribute deletion"
                        .format(self.__class__.__name__))


class StepFunction(AlmostImmutable):
    """
    Provide a step function that can be used as an ordinary callable function,
    in addition to providing arrays with the step locations and corresponding
    function values as public attributes.

    """

    def __init__(self, xsteps, ysteps):
        """
        Initialize the StepFunction instance

        :xsteps: the x values of step locations
        :ysteps: the function values corresponding to the steps. The function
                 is assumed to have the value y[i] for x values between x[i]
                 and x[i + 1].

        """
        # Here, we use numpy
        self.xsteps = numpy.array(xsteps)
        self.ysteps = numpy.array(ysteps)

        # Check for validity
        if not (numpy.diff(self.xsteps) >= 0.0).all():
            raise ValueError("'xsteps' should be monotonically increasing")

    def __call__(self, x):
        """
        Call the function

        :x: array-like with x values at which to evaluate the function

        """
        indices = numpy.searchsorted(self.xsteps, x, side='right') - 1
        return self.ysteps[indices]

    def plot(self, axes=None, **kwargs):
        """
        Plot the step function

        The step function can be added to an existing plot via the optional
        'axes' argument.

        :axes: Axes instance to add the step function to. If None (default),
               the current Axes instance is used if any, or a new one created.
        :kwargs: additional keyword arguments passed on to the axes.step()
                 method used to plot the function. Note in particular
                 keywords such as 'linestyle', 'linewidth', 'color' and
                 'label'.
        :returns: list containing the Line2D instance for the plotted step
                  function

        """
        if axes is None:
            axes = pyplot.gca()
        return axes.step(self.xsteps, self.ysteps, where='post', **kwargs)


def gaussian(x, mean=0.0, cov=1.0):
    """
    Compute a multivariate Gaussian (normal distribution pdf)

    Seriously, you never know when you're gonna need a Gaussian. Best to keep
    it here in the utils module.

    :x: array-like of shape (m, n1, n2, ...), giving m n-dimensional x-values
        at which to evaluate the gaussian, where n = n1 * n2 * .... In the 1d
        case, the array can be of shape (m,).
    :mean: array-like, broadcastable to shape (n,), giving the n-dimensional
           location of the Gaussian peak (the mean of the distribution for
           which this is the pdf).
    :cov: array-like, broadcastable to shape (n, n), giving the shape of the
          Gaussian around the peak (the covariance matrix of the distribution
          for which this is the pdf).

    """
    m = len(x)
    x = numpy.reshape(x, (m, -1))
    n = x.shape[1]
    mean, __ = numpy.broadcast_arrays(numpy.asarray(mean), numpy.empty(n))
    cov, __ = numpy.broadcast_arrays(numpy.asarray(cov), numpy.empty((n, n)))
    return stats.multivariate_normal.pdf(x, mean=mean, cov=cov)


def sensibly_divide(num, denom, masked=False):
    """
    Sensibly divide two numbers or arrays of numbers (or any combination
    thereof)

    Sensibly in this case means that division by zero error is only raised if
    the denominator is zero and the numerator is non-zero and non-nan. If both
    the numerator and denominator are zero, or if the numerator is nan, the
    result of the division is nan.

    The use of nans means that the output is always a float array, regardless
    of the input types.

    :num: numerator
    :denom: denominator
    :masked: if True, the result is a masked array with masked values rather
             than nans at the problematic locations. Note that if either num or
             denom are masked, the result will always be a masked array, but if
             masked == False the mask will not be extended by this function,
             and any problematic locations encountered will be filled with
             unmasked nans.
    :returns: num / denom, sensibly

    """
    # Get broadcasted views
    num_bc, denom_bc = numpy.broadcast_arrays(num, denom)

    # Get float versions, for exact comparison to 0.0 and nan
    if isinstance(num, numpy.ma.MaskedArray):
        # Manually broadcast mask
        num_bc_mask, __ = numpy.broadcast_arrays(
            numpy.ma.getmaskarray(num), denom)
        num_bc = numpy.ma.array(num_bc, mask=num_bc_mask)
        num_bc_float = numpy.ma.array(num_bc, dtype=numpy.float_,
                                      keep_mask=True)
    else:
        num_bc_float = numpy.array(num_bc, dtype=numpy.float_)

    if isinstance(denom, numpy.ma.MaskedArray):
        __, denom_bc_mask = numpy.broadcast_arrays(
            num, numpy.ma.getmaskarray(denom))
        denom_bc = numpy.ma.array(denom_bc, mask=denom_bc_mask)
        denom_bc_float = numpy.ma.array(denom_bc, dtype=numpy.float_,
                                        copy=True, keep_mask=True)
    else:
        denom_bc_float = numpy.array(denom_bc, dtype=numpy.float_, copy=True)

    # Identify potentially problematic locations
    denom_zero = (denom_bc_float == 0.0)
    if numpy.any(denom_zero):
        num_zero_or_nan = numpy.logical_or(num_bc_float == 0.0,
                                           numpy.isnan(num_bc_float))
        problems = numpy.logical_and(denom_zero, num_zero_or_nan)
        if numpy.any(problems):
            # Either mask the problematic locations, or set them to nan
            if masked:
                denom_bc = numpy.ma.masked_where(problems, denom_bc)
            else:
                # denom_bc_float is a copy (safe to modify), and float (accepts
                # nan)
                denom_bc = denom_bc_float
                denom_bc[problems] = numpy.nan

    return num_bc / denom_bc


def add_ticks(axis, ticklocs, ticklabels):
    """
    Add new ticks and ticklabels to a matplotlib Axis instance while keeping
    existing, non-conflicting old ones

    :axis: Axis instance to add ticklabels to
    :ticklocs: sequence of tick locations
    :ticklabels: sequence of ticklabels

    """
    current_locs = axis.get_ticklocs()
    current_labels = [tl.get_text() for tl in axis.get_ticklabels()]
    new_ticklocs = numpy.hstack((current_locs, ticklocs))
    new_ticklabels = numpy.hstack((current_labels, ticklabels))
    axis.set_ticks(new_ticklocs)
    axis.set_ticklabels(new_ticklabels)


def edge_regions(arr):
    """
    Identify the connected regions of non-zero values along the edges of an
    array

    :arr: array
    :returns: a boolean array of the same shape as arr, valued True for
              elements corresponding to elements in the connected edge regions
              in arr, and False everywhere else.

    """
    labels, _ = measurements.label(arr)
    n = labels.ndim

    edgelabels = []
    for axis in range(n):
        # Bring axis to front
        labels = numpy.swapaxes(labels, 0, axis)

        # Collect labels present in edge slices
        edgelabels = numpy.union1d(edgelabels, labels[0])
        edgelabels = numpy.union1d(edgelabels, labels[-1])
        # Swap back axis
        labels = numpy.swapaxes(labels, 0, axis)

    edgelabels = numpy.trim_zeros(edgelabels)

    mask = numpy.zeros_like(labels, dtype=bool)
    for label in edgelabels:
        mask = numpy.logical_or(mask, labels == label)

    return mask


def pearson_correlogram(in1, in2, mode='full'):
    """
    Compute the Pearson correlation coefficients of two arrays at all
    displacements.

    Z-score normalization of the overlapping parts of the arrays is performed
    separately at each displacement. Each entry in the resulting array is thus
    in the interval `[-1.0, 1.0]`.

    Masked arrays are supported, and the treatment is fully automatic.

    Parameters
    ----------
    in1, in2 : array-like, with the same number of dimensions
        Arrays to correlate.
    mode : {'full', 'valid', 'same'}, optional
        String indicating the size of the output. See `scipy.signal.convolve`
        for details.

    Returns
    -------
    ndarray
        Correlogram with Pearson correlation coefficients

    """
    masked = False
    if isinstance(in1, numpy.ma.MaskedArray):
        mask1 = numpy.ma.getmaskarray(in1)
        in1 = numpy.ma.getdata(in1)
        masked = True
    else:
        in1 = numpy.array(in1)
        mask1 = numpy.zeros_like(in1, dtype=numpy.bool_)

    if isinstance(in2, numpy.ma.MaskedArray):
        mask2 = numpy.ma.getmaskarray(in2)
        in2 = numpy.ma.getdata(in2)
        masked = True
    else:
        in2 = numpy.array(in2)
        mask2 = numpy.zeros_like(in2, dtype=numpy.bool_)

    ndim = in1.ndim
    if in2.ndim != ndim:
        raise ValueError("input arrays must have the same number of "
                         "dimensions")

    shape1, shape2 = numpy.array(in1.shape), numpy.array(in2.shape)

    if mode == 'full':
        corrshape = tuple(shape1 + shape2 - 1)
        offset = 1 - shape2
    elif mode == 'valid':
        corrshape = tuple(numpy.abs(shape1 - shape2) + 1)
        offset = numpy.zeros((ndim,), dtype=numpy.int_)
    elif mode == 'same':
        corrshape = tuple(shape1)
        offset = (1 - shape2) // 2
    else:
        raise ValueError("unknown mode {}".format(mode))

    if masked:
        corr = numpy.ma.zeros(corrshape)
        missing = numpy.ma.masked
    else:
        corr = numpy.zeros(corrshape)
        missing = numpy.nan

    # Bind functions used in the loop to local names
    maximum, minimum, column_stack, logical_or, sqrt, newaxis, slice_ = (
        numpy.maximum, numpy.minimum, numpy.column_stack, numpy.logical_or,
        numpy.sqrt, numpy.newaxis, slice)
    dims = range(ndim)
    for ind in numpy.ndindex(corrshape):
        # Compute displacement
        disp = offset + ind

        # Extract overlap
        start = maximum(disp, 0)
        stop = minimum(disp + shape2, shape1)
        range1 = column_stack((start, stop))
        range2 = range1 - disp[:, newaxis]
        sl1, sl2 = (), ()
        for i in dims:
            sl1 += (slice_(*range1[i]),)
            sl2 += (slice_(*range2[i]),)

        # Only consider valid overlapping values
        valid = ~logical_or(mask1[sl1], mask2[sl2])

        c1 = in1[sl1][valid]
        c2 = in2[sl2][valid]
        n = c1.size

        if n == 0:
            # Nothing to see here here (avoid division by zero)
            corr[ind] = missing
            continue

        ninv = 1.0 / n
        c1 = c1 - ninv * c1.sum()  # Much faster than using -= and c1.mean()
        c2 = c2 - ninv * c2.sum()

        denom_sq = (c1 * c1).sum() * (c2 * c2).sum()
        if denom_sq == 0.0:
            corr[ind] = missing
        else:
            num = (c1 * c2).sum()
            corr[ind] = num / sqrt(denom_sq)

    return corr


def kde_bw(data, n_bw=100, cv=5):
    """
    Estimate the optimal KDE bandwith for a single-variable dataset using cross
    validation

    To estimate individual bandwidths for each feature in a multivariate
    dataset, apply this function to each single-feature subset separately.

    Parameters
    ----------
    data : array-like, shape (n_data, n_features)
        The dataset.
    n_bw : integer, optional
        Number of bandwidths to try out. Increasing this number increases the
        accuracy of the best bandwidth estimate, but also increases the
        computational demands of the function.
    cv : integer, optional
        Number of folds to use for cross validation.

    Returns
    -------
    scalar
        Estimated optimal bandwidth.

    """
    n, std = len(data), numpy.std(data)
    # Use the silverman rule times 1.1 as the maximal candidate bandwidth, an
    # one tenth of this as the minimal
    silverman = (0.75 * n) ** (-0.2)
    max_bw = 1.1 * silverman * std
    grid = GridSearchCV(
        KernelDensity(),
        {'bandwidth': numpy.linspace(0.1 * max_bw, max_bw, n_bw)},
        cv=cv)
    grid.fit(data)
    return grid.best_params_['bandwidth']


def plot_kde(data, data2=None, *args, **kwargs):
    """
    Plot a kernel density estimate of univariate or bivariate data

    This is a wrapper around `seaborn.kdeplot`, using `kde_bw` to estimate
    optimal bandwidth separately for each feature. The call signature is the
    same as `seaborn.kdeplot`.

    Parameters
    ----------
    data, data2, *args, **kwargs
        See the documentation for `seaborn.kdeplot`.

    Returns
    -------
        See the documentation for `seaborn.kdeplot`.

    """
    data_ = numpy.asarray(data)
    if data2 is not None:
        data2_ = numpy.asarray(data2)
        data_ = numpy.column_stack((data, data2_))
        bw = (kde_bw(data_[:, 0][:, numpy.newaxis]),
              kde_bw(data_[:, 1][:, numpy.newaxis]))
    else:
        if data_.ndim == 1:
            data_ = data[:, numpy.newaxis]
        bw = kde_bw(data_)

    kwargs.update({'bw': bw})
    return seaborn.kdeplot(data, data2=data2, *args, **kwargs)
