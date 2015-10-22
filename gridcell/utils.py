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
from scipy import stats, linalg
from scipy.ndimage import measurements
from sklearn.neighbors import KernelDensity
from sklearn.grid_search import GridSearchCV
import seaborn


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


def kde_bw(data, n_folds=None, n_bw=100):
    """
    Estimate the optimal KDE bandwiths for a dataset using cross validation

    The function estimates bandwiths separately for each feature in the
    dataset.
    To estimate individual bandwidths for each feature in a multivariate
    dataset, apply this function to each single-feature subset separately.

    Parameters
    ----------
    data : array-like, shape (n_data, n_features)
        The dataset.
    n_folds : integer, optional
        Number of folds to use for cross validation. If None, the default
        number of folds in the underlying cross validation functions are used.
    n_bw : integer, optional
        Number of bandwidths to try out. Increasing this number increases the
        accuracy of the best bandwidth estimate, but also increases the
        computational demands of the function.

    Returns
    -------
    list
        List of estimated optimal bandwidths, one for each feature in the
        dataset.

    """
    cv_kw = {}
    if n_folds is not None:
        cv_kw.update(cv=n_folds)

    data = numpy.asarray(data)
    if data.ndim == 1:
            data = data[:, numpy.newaxis]
    nd, nf = data.shape
    bw = []
    silverman_constant = (0.75 * nd) ** (-0.2)
    for i in range(nf):
        feature = data[:, i]
        # Use the silverman rule times 1.1 as the maximal candidate bandwidth,
        # an one tenth of this as the minimal
        std = numpy.std(feature)
        max_bw = 1.1 * silverman_constant * std
        grid = GridSearchCV(
            KernelDensity(),
            dict(bandwidth=numpy.linspace(0.1 * max_bw, max_bw, n_bw)),
            **cv_kw)
        grid.fit(data)
        bw.append(grid.best_params_['bandwidth'])
    return bw


def kdeplot(data, data2=None, n_folds=None, n_bw=None, **kwargs):
    """
    Plot a kernel density estimate of univariate or bivariate data

    This is a wrapper around `seaborn.kdeplot`, using `kde_bw` to estimate
    optimal bandwidths. The call signature accepts all keywords that
    `seaborn.kdeplot` accepts, plus additional optional keyword arguments as
    stated below.

    Parameters
    ----------
    data, data2, *args, **kwargs
        See the documentation for `seaborn.kdeplot`.
    n_folds, n_bw
        See `kde_bw`.

    Returns
    -------
    See `seaborn.kdeplot`.

    """
    bw_kw = {}
    if n_folds is not None:
        bw_kw.update(n_folds=n_folds)
    if n_bw is not None:
        bw_kw.update(n_bw=n_bw)

    if data2 is not None:
        data_ = numpy.column_stack((data, data2))
        bw = kde_bw(data_, **bw_kw)
    else:
        bw = kde_bw(data, **bw_kw)[0]

    kwargs.update(bw=bw)
    return seaborn.kdeplot(data, data2=data2, **kwargs)


def distplot(data, n_folds=None, n_bw=None, **kwargs):
    """
    Plot a univariate distribution of observations

    This is a wrapper around `seaborn.distplot`, using `kde_bw` to estimate
    optimal bandwidths for kernel density estimates. The call signature accepts
    all keywords that `seaborn.kdeplot` accepts, plus additional optional
    keyword arguments as stated below.

    Parameters
    ----------
    data, data2, *args, **kwargs
        See the documentation for `seaborn.kdeplot`.
    n_folds, n_bw
        See `kde_bw`. To use the default value in the underlying function, let
        the corresponding keyword be None in this function.

    Returns
    -------
    See `seaborn.kdeplot`.

    """
    bw_kw = {}
    if n_folds is not None:
        bw_kw.update(n_folds=n_folds)
    if n_bw is not None:
        bw_kw.update(n_bw=n_bw)

    kde = kwargs.get('kde', True)
    if kde:
        bw = kde_bw(data, **bw_kw)[0]
        kde_kws = kwargs.pop('kde_kws', {})
        kde_kws.update(bw=bw)
        kwargs.update(kde_kws=kde_kws)
    return seaborn.distplot(data, **kwargs)


def check_stable(n_calls, function, *args, **kwargs):
    """
    Check that the output of a function is stable over a number of calls

    Useful e.g. for checking if the output of non-deterministic clustering
    algorithms is self-consistent.

    Parameters
    ----------
    n_calls : integer
        Number of times to call `function`.
    function : callable
        Function to investigate.
    args : sequence, optional
        Argument list to pass to function.
    kwargs : dict, optional
        Keyword arguments to pass to function.

    Returns
    -------
    bool
        True if the return value from `function` was equal for every call,
        False otherwise.

    Examples
    --------
    >>> # cells is a CellCollection instance. The stability of cells.k_means
    >>> # over 10 calls with n_clusters=4 and n_runs=10 is assessed.
    >>> def k_means_keys(*a, **k):
    >>>     return module_keys(*cells.k_means(*a, **k))
    >>> check_stable(10, k_means_keys, n_clusters=4, n_runs=10)
    True

    """
    out = function(*args, **kwargs)
    for __ in range(1, n_calls):
        new_out = function(*args, **kwargs)
        if not numpy.all(new_out == out):  # Support array-valued functions
            return False
    return True


def project_vectors(vectors, basis_vectors):
    """
    Project a set of vectors onto an arbitrary (possibly non-orthogonal) basis

    Parameters
    ----------
    vectors : array-like, shape (n1, n2, ..., k)
        Array of k-dimensional vectors to project.
    basis_vectors : array-like, shape (k, k)
        Array of k linearly independent k-dimensional vectors to compute the
        projections onto. The vectors do not have to be orthogonal.

    Returns
    -------
    coefficients : ndarray, shape (n1, n2, ..., k)
        Array of projection coefficients such that
        `coefficients[i1, i2, ..., j] * basis_vectors[j]` is the projection of
        `vectors[i1, i2, ...]` along `basis_vectors[j]`, and
        `coefficients[i1, i2, ...].dot(basis_vectors) == vectors[i1, i2, ...]`.

    """
    vectors = numpy.asarray(vectors)
    basis_vectors = numpy.asarray(basis_vectors)

    vshape = vectors.shape
    k = vshape[-1]
    (k1, k2) = basis_vectors.shape
    if k != k1 or k1 != k2:
        raise ValueError("need {0} linearly independent basis vectors of "
                         "dimension {0} to compute projection of 'vectors' of "
                         "dimension {0} ('basis_vectors' must be a square, "
                         "nonsingular array of shape ({0}, {0}).".format(k))

    a = numpy.inner(basis_vectors, basis_vectors)
    b = numpy.inner(basis_vectors, vectors)
    coeffs = linalg.solve(a, b, sym_pos=True)
    coeffs = numpy.rollaxis(coeffs, 0, coeffs.ndim)
    return coeffs
