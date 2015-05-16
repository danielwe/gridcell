#!/usr/bin/env python

"""File: utils.py
Module defining classes and functions that may come in handy throughout the
package

"""

import numpy
from scipy import stats
from scipy.ndimage import measurements
from matplotlib import pyplot


class AlmostImmutable(object):
    """
    A base class for "almost immutable" objects: instance attributes that have
    already been assigned cannot (easily) be reassigned or deleted, but creating
    new attributes is allowed.

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
        :ysteps: the function values corresponding to the steps. The function is
                 assumed to have the value y[i] for x values between x[i] and
                 x[i + 1].

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

        :axes: Axes instance to add the step function to. If None (default), the
               current Axes instance is used if any, or a new one created.
        :kwargs: additional keyword arguments passed on to the axes.step()
                 method used to plot the function. Note in particular
                 keywords such as 'linestyle', 'linewidth', 'color' and 'label'.
        :returns: list containing the Line2D instance for the plotted step
                  function

        """
        if axes is None:
            axes = pyplot.gca()
        return axes.step(self.xsteps, self.ysteps, where='post', **kwargs)


def gaussian(x, mean=0.0, cov=1.0):
    """
    Compute a multivariate Gaussian (normal distribution pdf)

    Seriously, you never know when you're gonna need a Gaussian. Best to keep it
    here in the utils module.

    :x: array-like of shape (m, n), giving m n-dimensional x-values at which to
        evaluate the gaussian
    :mean: array-like, broadcastable to shape (n,), giving the n-dimensional
           location of the Gaussian peak (the mean of the distribution for which
           this is the pdf).
    :cov: array-like, broadcastable to shape (n, n), giving the shape of the
          Gaussian around the peak (the covariance matrix of the distribution
          for which this is the pdf).

    """
    x = numpy.asarray(x)
    __, n = x.shape
    mean = numpy.asarray(mean) * numpy.ones(n)
    cov = numpy.eye(n).dot(numpy.asarray(cov))
    return stats.multivariate_normal.pdf(x, mean=mean, cov=cov)


def sensibly_divide(num, denom, masked=False):
    """
    Sensibly divide two numbers or arrays of numbers (or any combination
    thereof)

    Sensibly in this case means that division by zero error is only raised if
    the denominator is zero and the numerator is non-zero and non-nan. If both
    the numerator and denominator are zero, or if the numerator is nan, the
    result of the division is nan.

    :num: numerator
    :denom: denominator
    :masked: if True, the result is a masked array with masked values rather
             than nans at the problematic locations.
    :returns: numerator / denominator

    """
    # Fill masked entries so that we are sure they won't compare close to zero
    # or equal to nan
    denom_filled = numpy.ma.filled(denom, fill_value=1.0)
    num_filled = numpy.ma.filled(num, fill_value=1.0)

    #denom_zero = numpy.isclose(denom_filled, 0.0)
    denom_zero = (denom_filled == 0.0)
    if numpy.any(denom_zero):
        #num_zero_or_nan = numpy.logical_or(numpy.isclose(num_filled, 0.0),
        #                                   numpy.isnan(num_filled))
        num_zero_or_nan = numpy.logical_or(num_filled == 0.0,
                                           numpy.isnan(num_filled))
        match = numpy.logical_and(denom_zero, num_zero_or_nan)
        if masked:
            denom = numpy.ma.masked_where(match, denom)
        else:
            try:
                denom[match] = numpy.nan
            except TypeError:
                denom = numpy.nan

    return num / denom


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
    :returns: a boolean array of the same shape as arr, valued True for elements
              corresponding to elements in the connected edge regions in arr,
              and False everywhere else.

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


def my_pearson_corr(in1, in2):
    """
    Compute the Pearson correlation of two arrays at all possible displacements

    Normalization of the arrays is performed separately at each displacement,
    subtracting the mean and dividing by the standard deviation of the
    overlapping values only.

    This is probably not useful for anything, but the code has been written and
    shouldn't be removed without closer consideration.

    :in1: first array to correlate.
    :in2: second array to correlate.

    """
    #n = in1.ndim
    #if in2.ndim != n:
    #    raise ValueError("input arrays must have the same number of "
    #                     "dimensions.")

    shape1, shape2 = in1.shape, in2.shape

    ## Normalize
    #in1 = (in1 - in1.mean()) / in1.std()
    #in2 = (in2 - in2.mean()) / in2.std()

    corr = numpy.zeros([s1 + s2 - 1 for (s1, s2) in zip(shape1, shape2)])

    it = numpy.nditer(corr, flags=['multi_index'], op_flags=['writeonly'])
    while not it.finished:
        # Compute displacement
        disp = [1 - s2 + i for (s2, i) in zip(shape2, it.multi_index)]

        # Extract overlap
        sl1, sl2 = [], []
        for (d, s1, s2) in zip(disp, shape1, shape2):
            sl1.append(slice(max(0, d), min(s1, s2 + d)))
            sl2.append(slice(max(0, -d), min(s1, s2 - d)))
        c1 = in1[sl1]
        c2 = in2[sl2]

        # Subtract mean
        c1 = c1 - c1.mean()
        c2 = c2 - c2.mean()

        if not (numpy.allclose(c1, 0.0) or numpy.allclose(c2, 0.0)):
            ## Normalize
            #c1 = c1 / c1.std()
            #c2 = c2 / c2.std()

            # Compute correlation
            #it[0] = numpy.sum(c1 * c2) / c1.size
            it[0] = numpy.sum(c1 * c2) / (numpy.sqrt(numpy.sum(c1 * c1))
                                          * numpy.sqrt(numpy.sum(c2 * c2)))

        it.iternext()

    return corr
