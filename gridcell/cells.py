#!/usr/bin/env python

"""File: cells.py
Module defining classes to represent multiple tetrode spike train recordings
from neurons during behavior, and facilitate the analysis of the spatial
modulation of the neuronal activity. The module is specifically geared at
analyzing spatial firing rate maps from grid cells, and the grid patterns they
define, but can be extended to analyze all kinds of cells with spatially
modulated behavior.

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
from abc import ABCMeta, abstractmethod

import numpy
import pandas
from scipy import signal, linalg, spatial
from sklearn import cluster
from matplotlib import pyplot
from collections import MutableSequence

from .utils import (AlmostImmutable, gaussian, sensibly_divide,  #add_ticks
                    project_vectors)
from .shapes import Ellipse
from .imaps import IntensityMap2D
from .pointpatterns import PointPattern, Window
from .memoize.memoize import memoize_method, Memparams


_PI = numpy.pi
_2PI = 2.0 * _PI


class FeatureNames(object):
    """
    Lists of cell feature names and associated latex strings

    """
    peak_features = sum(
        ((
            ''.join(('peak', str(i), '_x')),
            ''.join(('peak', str(i), '_y')),
            ''.join(('peak', str(i), '_rad')),
            ''.join(('peak', str(i), '_ang')),
            ''.join(('peak', str(i), '_logx')),
            ''.join(('peak', str(i), '_logy')),
            ''.join(('peak', str(i), '_lograd')),
        ) for i in range(1, 7)),
        ()
    )

    grid_features = (
        'scale',
        'logscale'
    ) + peak_features + (
        'ellipse_x',
        'ellipse_y',
        'ellipse_rad',
        'ellipse_ang',
    )

    default_features = (
        'logscale',
    ) + sum(
        ((
            ''.join(('peak', str(i), '_x')),
            ''.join(('peak', str(i), '_y')),
        ) for i in range(1, 7)),
        ()
    )

    @staticmethod
    def grid_features_latex(normalize_peaks=True, **kwargs):
        end_unnorm = r"}$"
        if normalize_peaks:
            end = r"} / l$"
        else:
            end = end_unnorm

        return (
            r"$l$",
            r"$\log l$",
        ) + sum(
            (
                (
                    "".join((r"$a_{x, ", str(i), end)),
                    "".join((r"$a_{y, ", str(i), end)),
                    "".join((r"$l_{", str(i), end)),
                    "".join((r"$\beta_{", str(i), end_unnorm)),
                    "".join((r"$\log a_{x, ", str(i), end)),
                    "".join((r"$\log a_{y, ", str(i), end)),
                    "".join((r"$\log l_{", str(i), end)),
                )
                for i in range(1, 7)
            ),
            ()
        ) + (
            r"$\epsilon \cos 2 \theta$",
            r"$\epsilon \sin 2 \theta$",
            r"$\epsilon$",
            r"$2 \theta$",
        )

    @classmethod
    def latex_mapping(cls, **kwargs):
        def _lm(key):
            if key in cls.grid_features:
                return cls.grid_features_latex(
                    **kwargs)[cls.grid_features.index(key)]
            if isinstance(key, str):
                return key.replace('_', ' ')
                #return "".join([r"$\textrm{",
                #                key.replace("_", " "),
                #                r"}$"])
            return key
        return _lm


class Position(AlmostImmutable):
    """
    Represent positional data recorded over time

    Parameters
    ----------
    t : array-like
        Array containing the times of the position samples.
    x, y : array-like
        Arrays containing the x- and y-positions of the position samples.
        Missing samples can be represented by nans or by using a masked array
        for at least one of the arrays.
    speed_window : non-negative scalar, optional
        Length of the time span over which to average the computed speed at
        each sample. Should be given in the same unit as the time samples in
        `t`.
        ..note:: The window length is converted to a number of samples using
        the mean time interval between samples, and rounded up to the nearest
        odd number to get a symmetric interval. Hence, if the positions are
        sampled very irregularly the averaging procedure doesn't make much
        sense, and one should rather let `speed_window=0.0`.
    min_speed : non-negative scalar, optional
        Lower speed limit for samples to be considered valid. Should be given
        in the unit for speed derived from the position samples in `x, y` and
        the time samples in `t`.
    **kwargs : dict, optional
        Any additional keyword arguments are stored as a dict in the attribute
        `info`. They are not used for internal computations.

    """
    params = Memparams(dict, 'params')

    def __init__(self, t, x, y, speed_window=0.0, min_speed=0.0, **kwargs):
        self.params = dict(speed_window=speed_window, min_speed=min_speed)
        self.info = kwargs

        t = numpy.squeeze(t)
        x = numpy.ma.squeeze(x)
        y = numpy.ma.squeeze(y)

        nanmask = numpy.logical_or(numpy.isnan(x), numpy.isnan(y))
        x = numpy.ma.masked_where(nanmask, x)
        y = numpy.ma.masked_where(nanmask, y)
        tweights, dweights = self.time_and_distance_weights(t, x, y)

        self.data = dict(t=t, x=x, y=y, tweights=tweights, dweights=dweights)

    @staticmethod
    def time_and_distance_weights(t, x, y):
        """
        Compute time- and distance weights for position samples

        The time- and distance weights are arrays that assign a time interval
        and a distance interval to each sample.

        Parameters
        ----------
        t : array-like
            Array containing the times of the position samples.
        x, y : array-like
            Arrays containing the x- and y-positions of the position samples.
            Missing samples can be represented by using masked arrays.

        Returns
        -------
        tweights : ndarray
            The length of time (time weight) associated with each position
            sample.
        dweights : masked ndarray
            The covered distance (distance weight) associated with each
            position sample.

        """
        tsteps = numpy.diff(t)
        tweights = 0.5 * numpy.hstack((tsteps[0], tsteps[:-1] + tsteps[1:],
                                       tsteps[-1]))

        xsteps = numpy.ma.diff(x)
        ysteps = numpy.ma.diff(y)
        dsteps = numpy.ma.sqrt(xsteps * xsteps + ysteps * ysteps)
        dweights = 0.5 * numpy.ma.hstack((dsteps[0], dsteps[:-1] + dsteps[1:],
                                          dsteps[-1]))

        return tweights, dweights

    def speed(self):
        """
        Compute the speed at each sample.

        The length of time over which to average the computed speed around each
        sample is given by `self.params['speed_window']`.

        Returns
        -------
        speed : masked ndarray
            The speed at each position sample.

        """
        speed_window = self.params['speed_window']
        if not speed_window >= 0.0:
            raise ValueError("'speed_window' must be a non-negative number")

        data = self.data

        t = data['t']
        mean_tstep = numpy.mean(numpy.diff(t))

        window_length = 2 * int(0.5 * speed_window / mean_tstep) + 1
        window_sequence = numpy.empty(window_length)
        window_sequence.fill(1.0 / window_length)

        tweights, dweights = data['tweights'], data['dweights']
        dw_mask = numpy.ma.getmaskarray(dweights)
        dw_filled = numpy.ma.filled(dweights, fill_value=0.0)

        tweights_filt = sensibly_divide(
            signal.convolve(tweights, window_sequence, mode='same'),
            signal.convolve(numpy.ones_like(tweights), window_sequence,
                            mode='same'))
        dweights_filt = sensibly_divide(
            signal.convolve(dw_filled, window_sequence, mode='same'),
            signal.convolve((~dw_mask).astype(numpy.float_),
                            window_sequence, mode='same'), masked=True)

        return dweights_filt / tweights_filt

    def filtered_data(self):
        """
        Apply a speed mask to the data in `self.data`

        Data at samples where the speed is below the minimum speed will be
        masked.

        Returns
        -------
        dict
            Dict similar to `self.data`, but with all data masked at samples
            where `self.speed() < self.params['min_speed']`. ..note::
            `self.data['t']` is not masked -- the time is of interest even for
            invalid samples.

        """
        data = self.data
        t, x, y = data['t'], data['x'], data['y']
        tweights, dweights = data['tweights'], data['dweights']
        speed = self.speed()

        min_speed = self.params['min_speed']
        if not min_speed >= 0.0:
            raise ValueError("'speed_window' must be a non-negative number")

        speedmask = numpy.logical_or(speed < min_speed,
                                     numpy.ma.getmaskarray(speed))

        x = numpy.ma.masked_where(speedmask, x)
        y = numpy.ma.masked_where(speedmask, y)
        dweights = numpy.ma.masked_where(speedmask, dweights)
        tweights = numpy.ma.masked_where(speedmask, tweights)

        return dict(t=t, x=x, y=y, tweights=tweights, dweights=dweights)

    @memoize_method
    def _occupancy(self, bins, range_):
        """
        Compute a map of occupancy times

        This part of the computation in `BaseCell.occupancy` is factored out
        to optimize memoization.

        """
        data = self.filtered_data()
        x, y, tweights = data['x'], data['y'], data['tweights']
        hist, xedges, yedges = numpy.histogram2d(
            numpy.ma.compressed(x),
            numpy.ma.compressed(y),
            bins=bins,
            range=range_,
            normed=False,
            weights=numpy.ma.compressed(tweights))

        return IntensityMap2D(hist, (xedges, yedges))

    def occupancy(self, bins, range_=None, distribution=False):
        """
        Compute a map of occupancy times

        Only samples considered valid according to the speed filtering are
        included in the histogram.

        Parameters
        ----------
        bins : int or [int, int] or array-like or [array, array]
            Bin specification defining the bins to use in the histogram. The
            simplest formats are an integer `nbins` or a tuple `(nbins_x,
            nbins_y)`, giving the number of bins of equal widths in each
            direction. For information about other valid formats, see the
            documentation for `numpy.histogram2d`.
        range_ : array-like, shape(2,2), optional
            Range specification giving the x and y values of the outermost bin
            edges. The format is `[[xmin, xmax], [ymin, ymax]]`. Samples
            outside this region will be discarded.
        distribution : bool, optional
            If True, the occupancy map will be normalized to have sum 1, such
            that it can be interpreted as a frequency distribution for the
            occupancy.

        Returns
        -------
        IntensityMap2D
            Occupancy map.

        """
        occupancy = self._occupancy(bins, range_)

        if distribution:
            occupancy /= self.total_time()

        return occupancy

    def total_time(self):
        """
        Compute the total recording time

        Returns
        -------
        scalar
            Total time.

        """
        return numpy.ma.sum(self.filtered_data()['tweights'])

    def plot_path(self, axes=None, linewidth=0.5, color='0.5', alpha=0.5,
                  **kwargs):
        """
        Plot the path through the valid positions

        Parameters
        ----------
        axes : Axes or None, optional
            Axes instance to add the path to. If None, the most current Axes
            instance with `aspect='equal'` is grabbed or created.
        linewidth : scalar, optional
            The width of the plotted path.
        color : valid Matplotlib color specification
            The color to use for the path.
        alpha : scalar in [0.0, 1.0], optional
            The opacity of the path.
        kwargs : dict, optional
            Additional keyword arguments passed to `axes.plot`. Note in
            particular the keywords 'linestyle' and 'label'.

        Returns
        -------
        list
            List containing the plotted Line2D instance.

        """
        if axes is None:
            axes = pyplot.gca(aspect='equal')

        data = self.filtered_data()
        x, y = data['x'], data['y']

        kwargs.update(linewidth=linewidth, color=color, alpha=alpha)
        return axes.plot(x, y, **kwargs)

    def plot_samples(self, axes=None, marker='.', s=1.0, c=None,
                     alpha=0.5, **kwargs):
        """
        Make a scatter plot of the recorded positions without drawing a line

        Parameters
        ----------
        axes : Axes or None, optional
            Axes instance to add the path to. If None, the most current Axes
            instance with `aspect='equal'` is grabbed or created.
        marker : matplotlib marker spec, optional
            Marker to use when drawing the positions.
        s : scalar or sequence, optional
            Marker sizes.
        c : matplotlib color spec, a sequence of such, or None, optional
            Color(s) to use when drawing the markers. If None, a grayscale
            color is computed based on the time weight associated with each
            sample, such that the sample that accounts for the longest time is
            completely black.
        alpha : scalar in [0, 1], optional
            The opacity of the markers.
        **kwargs : dict, optional
            Additional keyword arguments are passed to `axes.scatter`.  Note
            in particular the keyword 'label'.

        Returns
        -------
        PathCollection
            The plotted PathCollection instance.

        """
        if axes is None:
            axes = pyplot.gca(aspect='equal')

        data = self.filtered_data()
        x, y, tweights = data['x'], data['y'], data['tweights']

        if c is None:
            c = 1.0 - (tweights / tweights.max())

        kwargs.update(marker=marker, s=s, c=c, alpha=alpha)
        return axes.scatter(x, y, **kwargs)

    def plot_speed(self, axes=None, tstart=None, tend=None, color='black',
                   min_speed=True, min_speed_kw=None, **kwargs):
        """
        Plot the speed versus time

        Parameters
        ----------
        axes : Axes or None, optional
            Axes instance to add the path to. If None, the most current Axes is
            grabbed or created.
        tstart : scalar or None, optional
            Start time of the interval to plot the speed over. If None, the
            interval begins at the beginning of the sample.
        tend : scalar or None, optional
            End time of the interval to plot the speed over. If None, the
            interval ends at the end of the sample.
        min_speed : bool, optional
            If True, a horizontal line is added to the plot at the lower speed
            limit for valid samples. By default, a solid, red line with
            'linewidth' 0.5 will be plotted, but this can be overridden using
            the parameter `min_speed_kw`. If false, no horizontal line is
            plotted.
        min_speed_kw : dict or None, optional
            Optional keyword arguments to pass to `axes.axhline` when plotting
            minimum speed.
        color : matplotlib color spec, optional
            Color to draw the speed curve with.
        **kwargs : dict, optional
            Additional keyword arguments are passed to `axes.plot`. Note
            in particular the keywords 'linestyle', 'linewidth' and 'label'.

        Returns
        -------
        list
            List containing the plotted Line2D instances.

        """
        if axes is None:
            axes = pyplot.gca()

        data = self.filtered_data()
        t, speed = (data['t'], data['speed'])

        start_index, end_index = 0, -1
        if tstart is not None:
            start_index = numpy.argmin(numpy.abs(t - tstart))
        if tend is not None:
            end_index = numpy.argmin(numpy.abs(t - tend))
        t = t[start_index:end_index]
        speed = speed[start_index:end_index]

        kwargs.update(color=color)
        lines = axes.plot(t, speed, **kwargs)

        if min_speed:
            mkw = dict(linewidth=0.5, color='r')
            if min_speed_kw is not None:
                mkw.update(min_speed_kw)

            min_line = axes.axhline(self.params['min_speed'], **mkw)
            lines.append(min_line)

        return lines


# This is how we create a python2+3-compatible abstract base class: create an
# intermediate empty abstract class inheriting anything we wish to inherit in
# our abstract base class, and inherit from this
AbstractAlmostImmutable = ABCMeta(str('AbstractAlmostImmutable'),
                                  (AlmostImmutable,), {})


class BaseCell(AbstractAlmostImmutable):
    """
    Base class for representing a cell with spatially modulated firing rate.

    This class defines most of the available operations and computations on
    a grid cell, but lacks one important piece: information about the firing of
    the cell. It is therefore only useful as an abstract base class for
    functional cell classes.

    Parameters
    ----------
    threshold : scalar, optional
        Threshold value used to detect peaks in the autocorrelogram of the
        firing rate of the cell.
    **kwargs : dict, optional
        Any additional keyword arguments are stored as a dict in the attribute
        `info`. They are not used for internal computations. The
        `features` attribute may look up features stored in this way.

    """
    params = Memparams(dict, 'params')

    def __init__(self, threshold=0.0, **kwargs):
        self.params = dict(threshold=threshold)
        self.info = kwargs

    def spikemap(self, distribution_spikemap=False, **kwargs):
        """
        Compute a map of the spatial spike distribution

        No temporal information is available here, so the spikemap is defined
        as the ratemap times the occupancy map. Note that up to a constant
        factor equal to the total time, this is equivalent to the rate map
        weighted by the occupancy distribution. This method should be
        reimplemented in derived classes where temporal information is present.

        Parameters
        ----------
        distribution_spikemap : bool, optional
            This purpose of this parameter is to distinguish between the
            distribution of spikes and the normalized distribution of spatial
            spike frequency. It has no effect here, since no temporal
            information is available, but is included in the argument list to
            transparently avoid passing the keyword further.
        **kwargs : dict, optional
            Keyword arguments are passed to `BaseCell.ratemap` and
            `BaseCell.occupancy`.

        Returns
        -------
        IntensityMap2D
            Spike map.

        """
        kwargs.update(distribution_occupancy=False)
        return (self.ratemap(**kwargs) * self.occupancy(**kwargs))

    def occupancy(self, distribution_occupancy=False, **kwargs):
        """
        Compute a map of occupancy times

        No temporal information is available here, so the occupancy is defined
        to be 1 in each of the bins used in the ratemap computed by
        `BaseCell.ratemap`. This method should be reimplemented in derived
        classes where temporal information is present.

        Parameters
        ----------
        distribution_occupancy : bool, optional
            If True, the occupancy map will be normalized to have sum 1, such
            that it can be interpreted as a frequency distribution for the
            occupancy.
        **kwargs : dict, optional
            Keyword arguments are passed to `BaseCell.ratemap`.

        Returns
        -------
        IntensityMap2D
            Occupancy map.

        """
        occupancy = self.ratemap(**kwargs).indicator
        if distribution_occupancy:
            occupancy /= self.total_time(**kwargs)
        return occupancy

    def total_time(self, **kwargs):
        """
        Compute the total recording time

        No temporal information is available here, so the total time is defined
        to be the sum of the occupancy map from `BaseCell.occupancy`. This
        should be reasonably close to the truth in all cases, but may be
        inaccurate due to filtering artifacts. This method should be
        reimplemented in derived classes where temporal information is present.

        Parameters
        ----------
        **kwargs : dict, optional
            Keyword arguments are passed to `BaseCell.occupancy`.

        Returns
        -------
        scalar
            Total time.

        """
        kwargs.update(distribution_occupancy=False)
        return self.occupancy(**kwargs).sum()

    @abstractmethod
    def ratemap(self, normalize_ratemean=False, **kwargs):
        """
        Compute the firing rate map of the cell

        No firing information is available here, so no rate map is computed.
        This method must be implemented in all functional derived classes.

        Parameters
        ----------
        normalize_ratemean : bool, optional
            If True, the firing rate is normalized such that the mean firing
            rate is 1.0.
        **kwargs : dict, optional
            Not in use.

        Returns
        -------
        IntensityMap2D
            Firing rate map.

        """
        pass

    def rate_mean(self, normalize_ratemean=False, **kwargs):
        """
        Compute the spatial mean of the firing rate

        The contribution of the firing rate in each bin is weighted by the
        occupancy frequency in that bin. If no smoothing has been applied to
        the ratemap or occupancy map, this mean is thus equivalent to the
        mean firing rate calculated directly from spike and time data.

        Parameters
        ----------
        normalize_ratemean : bool, optional
            If True, the firing rate is normalized such that the mean firing
            rate is 1.0.
        **kwargs : dict, optional
            Keyword arguments are passed to `BaseCell.spikemap` and
            `BaseCell.total_time`.

        Returns
        -------
        scalar
            The spatial mean of the firing rate.

        """
        if normalize_ratemean:
            return 1.0
        kwargs.update(distribution_spikemap=False)
        return (self.spikemap(**kwargs).sum() / self.total_time(**kwargs))

    def rate_std(self, **kwargs):
        """
        Compute the spatial standard deviation of the firing rate

        The contribution of the firing rate in each bin is weighted by the
        value of occupancy frequency in that bin.

        Parameters
        ----------
        **kwargs : dict, optional
            Keyword arguments are passed to `BaseCell.rate_var`

        Returns
        -------
        scalar
            The spatial variance of the firing rate.

        """
        return numpy.sqrt(self.rate_var(**kwargs))

    def rate_var(self, ddof=0, **kwargs):
        """
        Compute the spatial variance of the firing rate

        The contribution of the firing rate in each bin is weighted by the
        occupancy frequency in that bin.

        Parameters
        ----------
        ddof : int, optional
            Delta degrees of freedom: the divisor used in calculating the
            variance is `N - ddof`, where `N` is the number of bins with firing
            rate values present. See the documentation for `numpy.var` for
            details.
        **kwargs : dict, optional
            Additional keyword arguments are passed to `BaseCell.ratemap`,
            `BaseCell.rate_mean` and `BaseCell.occupancy`.

        Returns
        -------
        scalar
            The spatial variance of the firing rate.

        """
        kwargs.update(distribution_occupancy=False)
        ratemap = self.ratemap(**kwargs)
        dev = ratemap - self.rate_mean(**kwargs)
        wdev_sq = self.occupancy(**kwargs) * dev * dev
        n = ratemap.count()
        return wdev_sq.sum() * n / (n - ddof)

    @memoize_method
    def acorr(self, mode='full', pearson='global', normalize_acorr=True,
              **kwargs):
        """
        Compute the autocorrelogram of the firing rate map

        This is a convenience wrapper for calling
        `self.ratemap(**kwargs).autocorrelate(...)`.

        Parameters
        ----------
        mode, pearson
            See `IntensityMap2D.autocorrelate`.
        normalize_acorr
            Passed as keyword 'normalize' to `IntensityMap2D.autocorrelate`.
        **kwargs : dict, optional
            Additional keyword arguments are passed to `BaseCell.ratemap`.

        Returns
        -------
        IntensityMap2D
            See `IntensityMap2D.autocorrelate`.

        """
        return self.ratemap(**kwargs).autocorrelate(mode=mode, pearson=pearson,
                                                    normalize=normalize_acorr)

    #@memoize_method
    def corr(self, other, mode='full', pearson='global', normalize_corr=True,
             **kwargs):
        """
        Compute the cross-correlogram of another cell's firing rate to this

        This is a convenience wrapper for calling
        `self.ratemap(**kwargs).correlate(other.ratemap(**kwargs), ...)`.

        Parameters
        ----------
        other : BaseCell
            BaseCell instance to correlate ratemaps with.
        mode, pearson
            See `IntensityMap2D.correlate`.
        normalize_corr
            Passed as keyword 'normalize' to `IntensityMap2D.correlate`.
        **kwargs : dict, optional
            Additional keyword arguments are passed to `BaseCell.ratemap`.

        Returns
        -------
        IntensityMap2D
            See `IntensityMap2D.correlate`.

        """
        # Register this cell for memoize cache clearance whenever this is
        # performed on the other cell.
        #memoize_method.register_friend(other, self)
        return self.ratemap(**kwargs).correlate(other.ratemap(**kwargs),
                                                mode=mode,
                                                pearson=pearson,
                                                normalize=normalize_corr)

    @staticmethod
    def detect_central_peaks(imap, threshold, n):
        """
        Identify the most central peaks in an IntensityMap

        The detected peaks are the peak closest to the center, and the `n - 1`
        peaks closest to it.

        A label array identifying the regions surrounding each peak is also
        returned.

        Parameters
        ----------
        imap : IntensityMap
            IntensityMap to detect peaks in.
        threshold : scalar
            Peak detection threshold -- a connected region of bins with
            intensity greater than this value corresponds to one peak.
        n : integer
            The number of peaks to detect and return.

        Returns
        -------
        ndarray, shape (n, 2)
            Array containing the peak locations. The most central peak has [x,
            y]-coordinates given by `peaks[0]`, and the coordinates of the six
            peaks closest to it follow in `peaks[i], i = 1, ..., n`. The peaks
            are sorted by the angle between the positive x axis and the line
            from the central peak out to the peripheral peak.
        labels : ndarray
            Label array identifying the regions surrounding the detected
            peaks: `labels == i` is an index to the region surrounding
            the `peak[i - 1]`.

        """
        all_peaks, labels, npeaks = imap.peaks(threshold)

        if npeaks < n:
            raise ValueError("Too few peaks detected")

        # Find the peak closest to the center
        apx, apy = all_peaks[:, 0], all_peaks[:, 1]
        apr = apx * apx + apy * apy
        cindex = numpy.argmin(apr)

        if n == 1:
            sort = numpy.array([cindex])
        else:
            # Find the n peaks closest to the center (incl. the center peak)
            cpeak = all_peaks[cindex]
            cpx = apx - cpeak[0]
            cpy = apy - cpeak[1]
            cpr = cpx * cpx + cpy * cpy
            rsort = numpy.argsort(cpr)

            # Sort the peaks surrounding the center peak by angle, starting
            # with the one closest to the x axis
            cpx = cpx[rsort[1:n]]
            cpy = cpy[rsort[1:n]]

            angles = numpy.arctan2(cpy, cpx)
            asort = numpy.argsort(angles)
            start_index = numpy.argmin(numpy.abs(angles[asort]))
            asort = numpy.roll(asort, -start_index)
            sort = numpy.hstack((rsort[0], rsort[1:n][asort]))

        # Grab the sorted peaks, discard the radius information for now
        peaks = all_peaks[sort, :2]

        # Find corresponding labels
        new_labels = numpy.zeros_like(labels)
        for (i, s) in enumerate(sort):
            new_labels[labels == (s + 1)] = i + 1

        return peaks, new_labels

    @memoize_method
    def _peaks(self, threshold, **kwargs):
        acorr = self.acorr(**kwargs)
        peaks, __ = self.detect_central_peaks(acorr, threshold, 7)

        # Discard center peak
        return peaks[1:]

    def grid_peaks(self, threshold=None, polar_peaks=False, project_peaks=True,
                   normalize_peaks=False, **kwargs):
        """
        Find the six inner peaks in the autocorrelogram of the firing rate map.

        The peaks are sorted by angle with the positive x axis.

        Parameters
        ----------
        threshold : scalar in [-1, 1] or None, optional
            Threshold value for peak detection. If None, the parameter
            `self.params['threshold']` is used.
        polar_peaks : bool, optional
            If True, the peak locations are returned as polar coordinates. If
            False, they are returned as cartesian coordinates.
        project_peaks : bool, optional
            If True, the peaks from the autocorrelogram are projected into the
            space of valid lattice vectors, such that each peak is equal to the
            sum of its two neighbors. If False, the peaks are returned without
            projection.
        normalize_peaks : bool, optional
            If True, peak coordinates are normalized by division with the grid
            scale, returned from `BaseCell.scale`, such that they only
            parametrize the shape of the grid pattern, not the size.
        **kwargs : dict, optional
            Additional keyword arguments are passed to `BaseCell.acorr` and
            `BaseCell.scale`.

        Returns
        -------
        ndarray, shape (6, 2)
            Array where `peaks[i], i = 1, ..., 5` contains the x- and
            y-coordinates, or the r- and theta-coordinates if polar_peaks is
            True, of each of the six peaks.

        """
        if threshold is None:
            threshold = self.params['threshold']

        peaks = self._peaks(threshold, **kwargs)

        if project_peaks:
            # Project to true lattice vectors
            pmat = (1 / 6) * linalg.toeplitz([2, 1, -1, -2, -1, 1])
            peaks = pmat.dot(peaks)

        if normalize_peaks:
            kwargs.update(project_peaks=project_peaks)
            peaks /= self.scale(**kwargs)

        if polar_peaks:
            px, py = peaks[:, 0], peaks[:, 1]
            radii = numpy.sqrt(px * px + py * py)
            angles = numpy.arctan2(py, px)
            peaks = numpy.column_stack((radii, angles))

        return peaks

    def scale(self, scale_mean='geometric', **kwargs):
        """
        Calculate the grid scale of the cell

        Parameters
        ----------
        scale_mean : {'geometric', 'arithmetic'}, optional
            Flag to select which grid scale definition to use:

            ``geometric``
                The scale is defined as the geometric mean of the semi-minor
                and semi-major axes of the grid ellipse.
            ``arithmetic``
                The scale is defined as the arithmetic mean of the distances
                from the origin to the each of the grid peaks.
        **kwargs : dict, optional
            Additional keyword arguments are passed to `BaseCell.grid_peaks`
            and `BaseCell.grid_ellipse`.

        Returns
        -------
        scalar
            The grid scale.

        """
        kwargs.update(normalize_peaks=False)
        if scale_mean == 'geometric':
            ellipse = self.grid_ellipse(**kwargs)
            scale = numpy.sqrt(ellipse.a * ellipse.b)
        elif scale_mean == 'arithmetic':
            kwargs.update(polar_peaks=True)
            radii = self.grid_peaks(**kwargs)[:, 0]
            scale = numpy.mean(radii)
        else:
            raise ValueError("unknown scale mean {}".format(scale_mean))

        return scale

    def logscale(self, **kwargs):
        """
        Calculate the natural logarithm of the scale of the cell

        Parameters
        ----------
        **kwargs : dict, optional
            Keyword arguments are passed to `BaseCell.scale`.

        Returns
        -------
        scalar
            The logarithm of the grid scale.

        """
        return numpy.log(self.scale(**kwargs))

    @memoize_method
    def grid_ellipse(self, **kwargs):
        """
        Fit an ellipse through the six peaks around the center of the
        autocorrelogram

        Parameters
        ----------
        **kwargs : dict, optional
            Keyword arguments are passed to `BaseCell.grid_peaks`.

        Returns
        -------
        Ellipse
            Ellipse instance representing the fitted ellipse.

        """
        kwargs.update(polar_peaks=False)
        peaks = self.grid_peaks(**kwargs)
        xscale, yscale = 0.5 * peaks[:, 0].ptp(), 0.5 * peaks[:, 1].ptp()
        f0 = numpy.sqrt(xscale * yscale)
        ellipse = Ellipse(fitpoints=peaks, f0=f0)
        return ellipse

    def ellpars(self, cartesian_ellpars=False, **kwargs):
        """
        Extract shape and orientation parameters of the grid ellipse

        The ellipse parameters consist of a polar coordinate tuple representing
        the shape and tilt of the ellipse. The radial coordinate is the ellipse
        eccentricity, while the angular coordinate is twice the angle between
        the x-axis and the semimajor axis of the ellipse (the multiplication by
        2 is done to match the ..math::`2 \pi`-degeneracy of polar coordinates
        with the rotation by ..math::`\pi`-symmetry of an ellipse).

        Parameters
        ----------
        **kwargs : dict, optional
            Keyword arguments are passed to `BaseCell.grid_ellipse`.

        Returns
        -------
        ndarray
            The coordinate tuple [eccentricity, 2 * tilt] representing the
            ellipse parameters. If `cartesian_ellpars == True`, the coordinate
            tuple is transformed from polar to cartesian coordinates, and
            becomes [eccentricity * cos(2 * tilt), eccentricity * sin(2
            * tilt)].

        """
        ellipse = self.grid_ellipse(**kwargs)
        ecc = ellipse.ecc
        theta = 2.0 * ellipse.tilt
        if cartesian_ellpars:
            return numpy.array((ecc * numpy.cos(theta),
                                ecc * numpy.sin(theta)))
        return numpy.array((ecc, theta))

    def sparsity(self, **kwargs):
        """
        Compute the spatial sparsity of the firing rate map of the cell

        References
        ----------

        Skaggs, W. E., McNaughton, B. L., Wilson, M. A., & Barnes, C. A.
        (1996). Theta phase precession in hippocampal neuronal populations and
        the compression of temporal sequences. Hippocampus, 6(2), 149--172.
        http://doi.org/10.1002/(SICI)1098-1063(1996)6:2<149::AID-HIPO6>3.0.CO;2-K  # noqa

        Buetfering, C., Allen, K., & Monyer, H. (2014). Parvalbumin
        interneurons provide grid cell-driven recurrent inhibition in the
        medial entorhinal cortex. Nature Neuroscience, 17(5), 710--8.
        http://doi.org/10.1038/nn.3696

        Returns
        -------
        scalar in range [0, 1]
            The spatial sparsity.
        **kwargs : dict, optional
            Keyword arguments will be passed to `BaseCell.rate_mean`,
            `BaseCell.ratemap`, `BaseCell.occupancy`.

        """
        kwargs.update(distribution_occupancy=True)
        fmean = self.rate_mean(**kwargs)
        rmap = self.ratemap(**kwargs)
        wrate_sq = rmap * rmap * self.occupancy(**kwargs)
        return 1.0 - fmean * fmean / wrate_sq.sum()

    def stability(self, **kwargs):
        """
        Compute a measure of the spatial stability of the cell firing pattern

        No temporal information is available here, so the stability is defined
        to be 1.0. This method should be reimplemented in derived classes where
        temporal information is present.

        Returns
        -------
        scalar
            The spatial stability of the cell firing pattern.

        """
        return 1.0

    @memoize_method
    def _grid_score(self, pearson, inner_factor, remove_cpeak, difference,
                    **kwargs):
        """
        Compute the grid score and the scale at which it was found

        This part of `BaseCell.grid_score` is factored out to optimize
        memoization.

        """
        angles_peaks = [60, 120]
        angles_troughs = [30, 90, 150]

        acorr = self.acorr(**kwargs)
        if pearson == 'global':
            acorr = (acorr - acorr.mean()) / acorr.std()

        bset = acorr.bset

        inner_radius = 0.0
        if remove_cpeak:
            # Define central peak radius
            ffield = self.firing_field(**kwargs)
            sigma = numpy.sqrt(numpy.amax(linalg.eigvalsh(ffield)))
            inner_radius = 2.0 * sigma

            acorr = acorr.shell(inner_radius, None)

        min_binwidth = min(numpy.mean(w) for w in bset.binwidths)
        min_corrwidth = min(numpy.max(numpy.abs(e)) for e in bset.edges)

        scale_step = min_binwidth
        scale_min = inner_radius + 4 * min_binwidth
        scale_max = min_corrwidth
        scales = numpy.arange(scale_min, scale_max, scale_step)

        acorr_rot_peaks = [[acorr.rotate(a)
                            for a in numpy.linspace(angle - 6.0, angle + 6.0,
                                                    num=5, endpoint=True)]
                           for angle in angles_peaks]
        acorr_rot_troughs = [[acorr.rotate(a)
                              for a in numpy.linspace(angle - 6.0, angle + 6.0,
                                                      num=5, endpoint=True)]
                             for angle in angles_troughs]

        def _score(scale):
            # Find doughnut in autocorrelgroam
            acorr_ring = acorr.shell(scale * inner_factor, scale)
            ring_data = acorr_ring.data
            ring_mask = ring_data.mask

            def _doughnut_corr(arot):
                # Extract doughnut from rotated autocorrelogram
                arot_ring = arot.shell(scale * inner_factor, scale)
                rot_ring_data = arot_ring.data
                rot_ring_mask = rot_ring_data.mask

                # Compute doughnut overlap
                full_mask = numpy.logical_or(ring_mask, rot_ring_mask)
                ring_olap = numpy.ma.array(ring_data, mask=full_mask)
                rot_ring_olap = numpy.ma.array(rot_ring_data,
                                               mask=full_mask)

                # Compute correlation
                if pearson == 'local':
                    ring_olap = ((ring_olap - numpy.ma.mean(ring_olap)) /
                                 numpy.ma.std(ring_olap))
                    rot_ring_olap = ((rot_ring_olap -
                                      numpy.ma.mean(rot_ring_olap)) /
                                     numpy.ma.std(rot_ring_olap))

                nbins = numpy.sum((~full_mask).astype(numpy.int_))
                corr = numpy.ma.sum(ring_olap * rot_ring_olap) / nbins

                return corr, nbins

            corr_peaks, nbins_peaks = zip(
                *(max(_doughnut_corr(arot) for arot in acorr_rot)
                  for acorr_rot in acorr_rot_peaks))
            corr_troughs, nbins_troughs = zip(
                *(min(_doughnut_corr(arot) for arot in acorr_rot)
                  for acorr_rot in acorr_rot_troughs))

            if difference == 'min':
                return min(corr_peaks) - max(corr_troughs)
            elif difference == 'mean':
                corr_peaks_mean = (
                    sum(c * n for (c, n) in zip(corr_peaks, nbins_peaks)) /
                    sum(nbins_peaks))
                corr_troughs_mean = (
                    sum(c * n for (c, n) in zip(corr_troughs, nbins_troughs)) /
                    sum(nbins_troughs))
                return corr_peaks_mean - corr_troughs_mean
            else:
                raise ValueError("unknown grid score mode: {}"
                                 .format(difference))

        max_score, max_scale = max((_score(scale), scale) for scale in scales)

        return max_score, max_scale

    def grid_score(self, return_scale=False, grid_score_pearson='local',
                   grid_score_inner_factor=0.0, grid_score_remove_cpeak=False,
                   grid_score_difference='mean', **kwargs):
        """
        Compute the grid score of the firing rate map of the cell

        Parameters
        ----------
        return_scale : bool, optional
            If True, the scale at which the maximum grid score was found is
            returned.
        grid_score_pearson : {None, 'global', 'local'}, optional
            If `'global'`, the autocorrelogram is Pearson normalized once and
            for all before rotations and doughnut extractions and correlations
            are computed. If `'local'`, Pearson normalization is performed
            independently on each doughnut. Otherwise, the plain (non-Pearson)
            correlations between rotated autocorrelogram doughnuts are used.
        grid_score_inner_factor : scalar in [0, 1], optional
            The inner radius of each grid score doughnut is equal to this
            factor times the outer radius.
        grid_score_remove_cpeak : bool, optional
            If True, the central peak is removed from all doughnuts.
        grid_score_difference : {'min', 'mean'}, optional
            If `'min'`, the grid score is defined as the difference between the
            minimum peak rotation correlation and maximum trough rotation
            correlation. If `'mean'`, the grid score is defined as the
            difference between the weighted mean of each set of correlations.
            The weights used are the number of valid autocorrelogram bins used
            for computing each correlation.
        **kwargs : dict, optional
            Additional keyword arguments are passed to `BaseCell.acorr` and
            `BaseCell.firing_field`.

        Returns
        -------
        scalar in [-1, 1]
            The grid score.
        scalar, optional (only returned if `return_scale == True`)
            The scale at which the maximum grid score was found.

        """
        max_score, max_scale = self._grid_score(
            grid_score_pearson,
            grid_score_inner_factor,
            grid_score_remove_cpeak,
            grid_score_difference,
            **kwargs)

        if return_scale:
            return max_score, max_scale
        return max_score

    def firing_field(self, threshold=None, **kwargs):
        """
        Compute a covariance matrix characterizing the average firing field
        shape

        The estimate is found by fitting a gaussian to the region surrounding
        the central peak in the autocorrelogram of the firing rate of the cell.

        Parameters
        ----------
        threshold : scalar in [-1, 1] or None, optional
            Threshold value defining the central peak region. If None, the
            parameter `self.params['threshold']` is used.
        **kwargs : dict, optional
            Additional keyword arguments are passed to `BaseCell.acorr`.

        Returns
        -------
        ndarray, shape (2, 2)
            The estimated covariance matrix.

        """
        if threshold is None:
            threshold = self.params['threshold']

        acorr = self.acorr(**kwargs)
        __, labels = self.detect_central_peaks(acorr, threshold, 1)

        central_mask = ~(labels == 1)
        __, __, firing_field = acorr.fit_gaussian(mask=central_mask)

        return firing_field

    def firing_field_map(self, **kwargs):
        """
        Compute an IntensityMap2D instance showing the fitted firing field

        The intensity map is computed over the same region as the
        autocorrelogram returned from `self.acorr`, and normalized to a maximal
        value of 1.0. It is thus useful for comparing the fitted field with the
        region it was fitted to.

        Parameters
        ----------
        **kwargs : dict, optional
            Keyword arguments are passed to `BaseCell.acorr` and
            `BaseCell.firing_field`.

        Returns
        -------
        IntensityMap2D
            Intensity map containing the fitted firing field.

        """
        acorr = self.acorr(**kwargs)
        bset = acorr.bset
        firing_field = self.firing_field(**kwargs)

        xcm, ycm = bset.cmesh
        xypoints = numpy.vstack((xcm.ravel(), ycm.ravel())).transpose()
        ffarr = gaussian(xypoints, cov=firing_field).reshape(acorr.shape)
        ffarr *= 1.0 / numpy.abs(ffarr).max()

        return IntensityMap2D(ffarr, bset)

    def features(self, index=FeatureNames.default_features, weights=None,
                 roll=0, normalize_peaks=True, **kwargs):
        """
        Compute a series of features of this cell

        The purpose of the feature series is to provide a unified interface to
        all the scalar properties of the cell, useful for statistical analysis,
        clustering etc.

        Parameters
        ----------
        index : sequence, optional
            Index to select which features are included in the series. The
            index must be a sequence containing strings from
            `gridcell.grid_features` and/or names of callable attributes on
            `self`, and/or keys to the `self.params` and/or `self.info` dicts.
            The labels in the index are interpreted as follows:

            ``peaki_x, peaki_y``
                Cartesian coordinates of the i-th peak from
                `BaseCell.grid_peaks`. The peaks are numbered from 1 to 6.
            ``peaki_rad, peaki_ang``
                Polar versions of the peak coordinates described above.
            ``peaki_logx, peaki_logy, peaki_lograd``
                The natural logarithm of the corresponding lengths described
                above.
            ``ellipse_rad, ellipse_ang``
                Polar coordinates representing the shape and orientation of the
                ellipse from `BaseCell.grid_ellipse`, as explained in detail in
                T`BaseCell.ellpars`.
            ``ellipse_x, ellipse_y``
                Cartesian versions of the ellipse parameters described above.
            ``_method_ or _key_``
                If the label does not match any of the candidates explained
                above, the following lookup attempts are made:
                1. The label is assumed to be the name of a callable attribute
                   of `self`, that returns a scalar and can be called without
                   arguments. (The callable can take optional keyword arguments
                   -- see `**kwargs`. It should swallow unused keyword
                   arguments silently). Example: 'grid_score'.
                2. The label is assumed to be a key to the `self.params`
                   dict.
                3. The label is assumed to be a key to the `self.info` dict.

        weights : dict or None, optional
            For each `label` in `index`, the corresponding feature is
            multiplied by `weights[label]` if this value exists. This allows
            for custom weighting of features against each other -- useful for
            e.g. tuning the relative importance of features when clustering.
        roll : integer, optional
            Quantities related to individual peaks are ordered from 1 to 6 in
            the order given by `numpy.roll(self.grid_peaks(), roll, axis=0)`.
            This parameter may be useful when comparing features between cells,
            in order to properly align corresponding peaks with each other.
        normalize_peaks : bool, optional
            See `BaseCell.grid_peaks`. This keyword is listed explicitly here
            because the default value may be different.
        **kwargs : dict, optional
            Additional keyword arguments are passed to all instance methods
            called from this method, such as `grid_peaks`, `grid_ellipse`,
            `scale`, `sparsity`, etc. This gives the user complete control over
            the choices underlying the feature computation. For example, to
            disable Bravais lattice projection of grid peaks and use the
            arithmetic mean definition of grid scale, use the keyword arguments
            `{'project_peaks': False, 'scale_mean': 'arithmetic'}`.

        Returns
        -------
        Series
            Series containing the requested cell features. The Series index is
            the same as `index`.
        """
        def _indexerror(label):
            return ValueError("Unknown label '{}'".format(label))

        # The peaks need their own keyword dict to store the keys
        # 'normalize_peaks' and 'polar_peaks' without sending them all over the
        # place.
        pkw = dict(kwargs)
        pkw.update(normalize_peaks=normalize_peaks)

        # The ellipse needs its own keyword dict to store the key
        # 'cartesian_ellipse' without sending it all over the place.
        ekw = dict(kwargs)

        data = dict(peaks=None, ellipse=None)

        def _lookup_feature(label):
            if label[:4] == 'peak':
                if data['peaks'] is None:
                    pkw.update(polar_peaks=False)
                    data['peaks'] = numpy.roll(self.grid_peaks(**pkw),
                                               roll, axis=0)
                    pkw.update(polar_peaks=True)
                    data['peaks_polar'] = numpy.roll(self.grid_peaks(**pkw),
                                                     roll, axis=0)
                peaks = data['peaks']
                peaks_polar = data['peaks_polar']

                peak_index = int(label[4]) - 1

                component = label[5:]
                if component == '_ang':
                    return peaks_polar[peak_index, 1]
                elif component.startswith('_log'):
                    def f(p):
                        return numpy.log(p)
                    component = '_' + label[9:]
                else:
                    def f(p):
                        return p

                if component == '_x':
                    return f(peaks[peak_index, 0])
                elif component == '_y':
                    return f(peaks[peak_index, 1])
                elif component == '_rad':
                    return f(peaks_polar[peak_index, 0])

            elif label[:7] == 'ellipse':
                if data['ellipse'] is None:
                    ekw.update(cartesian_ellpars=False)
                    data['ellipse'] = self.ellpars(**ekw)
                    ekw.update(cartesian_ellpars=True)
                    data['ellipse_cartesian'] = self.ellpars(**ekw)
                ellipse = data['ellipse']
                ellipse_cartesian = data['ellipse_cartesian']

                component = label[7:]
                if component == '_x':
                    return ellipse_cartesian[0]
                elif component == '_y':
                    return ellipse_cartesian[1]
                elif component == '_rad':
                    return ellipse[0]
                elif component == '_ang':
                    return ellipse[1]

            if hasattr(self, label):
                return getattr(self, label)(**kwargs)

            try:
                if label in self.params:
                    return self.params[label]
                if label in self.info:
                    return self.info[label]
            except TypeError:
                pass
            raise _indexerror(label)

        values = []
        if weights is None:
            weights = {}
        for label in index:
            weight = weights.get(label, 1.0)
            values.append(weight * _lookup_feature(label))

        return pandas.Series(values, index=index)

    def roll(self, other, **kwargs):
        """
        Determine the roll that aligns the grid peaks from this cell most
        closely with those from `other`

        Parameters
        ----------
        other : BaseCell
            Cell to align to.
        **kwargs : dict, optional
            Keyword arguments are passed to `BaseCell.grid_peaks`.

        Returns
        -------
        roll : integer
            The peak roll to apply to `self.grid_peaks` to align with
            `other.grid_peaks`. The peak roll is defined such that
            `numpy.roll(self.grid_peaks(), roll, axis=0)` and
            `other.grid_peaks` give coordinates to the most closely
            corresponding peaks in `self` and `other`.

        """
        kwargs.update(polar_peaks=True)

        sangles = self.grid_peaks(**kwargs)[:, 1]
        oangles = other.grid_peaks(**kwargs)[:, 1]

        roll = 0
        diff = numpy.mod(_PI + sangles - oangles, _2PI) - _PI
        delta = numpy.abs(numpy.sum(diff))
        for r in (-1, 1):
            diff = numpy.mod(
                _PI + numpy.roll(sangles, r) - oangles, _2PI) - _PI
            d = numpy.abs(numpy.sum(diff))
            if d < delta:
                delta = d
                roll = r
        return roll

    def distance(self, other, **kwargs):
        """
        Compute a distance between the grid patterns of cells

        This method defines a metric on the space of grid patterns. The
        distance is defined as the Euclidean distance between the feature
        arrays of the cells, using the relative peak roll given by
        `self.roll(other)`.

        Parameters
        ----------
        other : BaseCell
            BaseCell instance to measure distance to.
        kwargs : dict, optional
            Keyword arguments are passed to `BaseCell.features`.

        Returns
        -------
        scalar
            Distance between `self` and `other`.

        """
        if self is other:
            return 0.0

        roll = self.roll(other)
        ofeat = other.features(roll=0, **kwargs)
        dfeat = self.features(roll=roll, **kwargs) - ofeat
        return numpy.sqrt(numpy.sum(dfeat * dfeat))

    def phase(self, other, threshold=None, **kwargs):
        """
        Find the grid phase of this cell relative to another cell

        Parameters
        ----------
        other : BaseCell
            Cell instance to measure phase relative to.
        threshold : scalar in [-1, 1] or None, optional
            Threshold value defining the central peak region in the
            cross-correlogram. If None, the parameter
            `self.params['threshold']` is used.
        **kwargs : dict, optional
            Additional keyword arguments are passed to `BaseCell.corr`

        Returns
        -------
        ndarray, shape (2,)
            The x- and y-components of the grid phase of `self` relative to
            `other`.

        """
        if threshold is None:
            threshold = self.params['threshold']

        corr = self.corr(other, **kwargs)
        peaks, __ = self.detect_central_peaks(corr, threshold, 1)

        return numpy.array(peaks[0])

    def plot_ratemap(self, vmin=0.0, rate_kw=None, **kwargs):
        """
        Plot the spatial firing rate map of the cell

        This method is essentially a convenience wrapper around
        `self.ratemap().plot` -- the only difference is that it fixes the lower
        end of the colorbar at 0.0.

        Parameters
        ----------
        vmin
            See `IntensityMap2D.plot`.
        rate_kw : dict or None, optional
            Optional keyword arguments to pass to `BaseCell.ratemap`.
        **kwargs : dict, optional
            Additional keyword arguments are passed to `IntensityMap2D.plot`.

        Returns
        -------
        See `IntensityMap2D.plot`.

        """
        if rate_kw is None:
            rate_kw = {}
        kwargs.update(vmin=vmin)
        return self.ratemap(**rate_kw).plot(**kwargs)

    def plot_acorr(self, vmin=-1.0, vmax=1.0, threshold=False,
                   grid_peaks=False, grid_ellipse=False, acorr_kw=None,
                   **kwargs):
        """
        Plot the autocorrelogram of the firing rate map of the cell

        This method is essentially a wrapper around `self.acorr().plot`, but
        adds funcionality related to the grid peaks, peak threshold and grid
        ellipse.

        Parameters
        ----------
        vmin, vmax
            See `IntensityMap2D.plot` (..note:: defaults may differ).
        threshold : bool or scalar, optional
            Bins with values below the threshold value are masked from the
            plot. If True, `self.params['threshold']` is used as the threshold
            value. If False, no thresholding is performed. Otherwise, the
            provided value is used as the threshold value.
        grid_peaks : bool, optional
            If True, the grid peaks are marked in the plot, using some
            (hopefully) sensible plotting defaults. Dashed lines shoing the
            grid axes are also drawn. If more control is required, leave this
            as False and call `self.plot_grid_peaks` on the returned Axes
            instance.
        grid_ellipse : bool, optional
            If True, the grid ellipse is drawn in the plot, using some
            (hopefully) sensible plotting defaults. If more control is
            required, leave this False and call `self.plot_grid_ellipse` on the
            returned Axes instance.
        acorr_kw : dict or None, optional
            Optional keyword arguments to pass to `BaseCell.acorr`.
            ..note:: If the keyword argument 'pearson' is set to `None`, the
            arguments `vmin` and `vmax` in this method should probably be set
            to something other than the default (e.g. `None`).
        cbar_kw : dict or None, optional
            Optional keyword arguments to pass to `pyplot.colorbar`.
        **kwargs : dict, optional
            Additional keyword arguments pass to `IntensityMap2D.plot`.

        Returns
        -------
        See `IntensityMap2D.plot`

        """
        if acorr_kw is None:
            acorr_kw = {}
        acorr = self.acorr(**acorr_kw)

        if threshold is True:
            threshold = self.params['threshold']
        elif threshold is False:
            threshold = None

        kwargs.update(threshold=threshold, vmin=vmin, vmax=vmax)
        ret = acorr.plot(**kwargs)
        axes = ret[0]

        if grid_peaks:
            self.plot_grid_peaks(axes=axes)
        if grid_ellipse:
            self.plot_grid_ellipse(axes=axes)

        return ret

    def plot_grid_peaks(self, axes=None, marker='o', color='black',
                        gridlines=True, gridlines_kw=None, grid_kw=None,
                        **kwargs):
        """
        Plot the locations of grid peaks from the firing rate autocorrelogram

        Parameters
        ----------
        axes : Axes or None, optional
            Axes instance to add the firing rate to. If None, the most current
            Axes instance with `aspect='equal'` is grabbed or created.
        marker : matplotlib marker spec, optional
            Marker to use when drawing the peaks.
        color : matplotlib color spec, or a sequence of such, optional
            Color(s) to use when drawing the markers.
        gridlines : bool, optional
            If True, the axes of the grid pattern are added to the plot as
            lines through the peaks and the origin. By default, dashed lines
            with color '0.5' and opacity (alpha) 0.5 are used, but this can be
            overridden using the parameter `gridlines_kw`. If False, no lines
            are plotted.
        gridlines_kw : dict or None, optional
            Optional keyword arguments to pass to `axes.plot` when plotting
            gridlines.
        grid_kw : dict or None, optional
            Optional keyword arguments to pass to `BaseCell.grid_peaks`.
        **kwargs : dict, optional
            Additional keyword arguments pass to `axes.plot`. Note in
            particular the keyword `markersize`.

        Returns
        -------
        list
            List of plotted Line2D instances

        """
        if axes is None:
            axes = pyplot.gca(aspect='equal')

        gkw = {}
        if grid_kw is not None:
            grid_kw.update(grid_kw)
        gkw.update(polar_peaks=False)
        peaks = self.grid_peaks(**gkw)

        kwargs.update(linestyle='None', marker=marker, color=color)
        h = axes.plot(peaks[:, 0], peaks[:, 1], **kwargs)

        if gridlines:
            glkw = dict(linestyle='dashed', color='0.5', alpha=0.5)
            if gridlines_kw is not None:
                glkw.update(gridlines_kw)
            acorr = self.acorr(mode='full')
            xedges = acorr.bset.xedges
            gkw.update(polar_peaks=True)
            peaks_polar = self.grid_peaks(**gkw)
            angles = peaks_polar[:3, 1]
            line_y = numpy.outer(xedges, numpy.tan(angles))
            h += axes.plot(xedges, line_y, **glkw)

        return h

    def plot_grid_ellipse(self, linestyle='solid', linewidth=2.0, color='red',
                          smajaxis=True, grid_kw=None, **kwargs):
        """
        Plot the grid ellipse

        This method is essentially a wrapper around `self.grid_ellipse().plot`

        Parameters
        ----------
        linestyle, linewidth, color, smajaxis, **kwargs
            See `Ellipse.plot` (..note:: defaults may differ).
        grid_kw : dict or None, optional
            Optional keyword arguments to pass to `BaseCell.grid_ellipse`.
        **kwargs : dict, optional
            Additional keyword arguments are passed to `IntensityMap2D.plot`.

        Returns
        -------
        See `Ellipse.plot`.

        """
        if grid_kw is None:
            grid_kw = {}
        ell = self.grid_ellipse(**grid_kw)
        kwargs.update(linestyle=linestyle, linewidth=linewidth, color=color,
                      smajaxis=smajaxis)
        return ell.plot(**kwargs)

    def plot_corr(self, other, vmin=-1.0, vmax=1.0, threshold=False,
                  center_peak=False, corr_kw=None, **kwargs):
        """
        Plot the cross-correlogram of the firing rate map of this and another
        cell

        This method is essentially a wrapper around `self.corr(other).plot`,
        but adds funcionality related to the grid peaks, peak threshold and
        grid ellipse.

        Parameters
        ----------
        other : BaseCell
            BaseCell instance to plot correlated ratemap with.
        vmin, vmax
            See `IntensityMap2D.plot` (..note:: defaults may differ).
        threshold : bool or scalar, optional
            Bins with values below the threshold value are masked from the
            plot. If True, `self.params['threshold']` is used as the threshold
            value. If False, no thresholding is performed. Otherwise, the
            provided value is used as the threshold value.
        cpeak : bool, optional
            If True, the most central peak in the correlogram is marked in the
            plot, using some (hopefully) sensible plotting defaults. Dashed
            coordinate axes are also drawn for easy visual assesment of the
            offset of the peak from the origin. If more control is required,
            leave this as False and call `self.detect_central_peaks` on
            `self.corr(other)` to find peaks that can be added to the returned
            Axes instance manually.
            ..note:: the peak detection threshold is
            `self.params['threshold']`, regardless of the value of the keyword
            `threshold` (which is only used to mask bins in the plot).
        corr_kw : dict or None, optional
            Optional keyword arguments to pass to `BaseCell.corr`.
            ..note:: If the keyword argument 'pearson' is set to `None`, the
            arguments `vmin` and `vmax` in this method should probably be set
            to something other than the default (e.g. `None`).
        **kwargs : dict, optional
            Additional keyword arguments are passed to `IntensityMap2D.plot`.

        Returns
        -------
        See `IntensityMap2D.plot`

        """
        if corr_kw is None:
            corr_kw = {}
        corr = self.corr(other, **corr_kw)

        if threshold is True:
            threshold = self.params['threshold']
        elif threshold is False:
            threshold = None

        kwargs.update(threshold=threshold, vmin=vmin, vmax=vmax)
        ret = corr.plot(**kwargs)
        axes = ret[0]

        if center_peak:
            peak, __ = self.detect_central_peaks(corr,
                                                 self.params['threshold'], 1)
            axes.plot(peak[0, 0], peak[0, 1], linestyle='None', marker='o',
                      color='black')

            # Add a axes for comparison
            axes.axvline(0.0, linestyle='dashed', color='0.5', alpha=0.5)
            axes.axhline(0.0, linestyle='dashed', color='0.5', alpha=0.5)

        return ret

    def plot_firing_field(self, vmin=-1.0, vmax=1.0, threshold=False,
                          ffield_kw=None, **kwargs):
        """
        Plot the shape of the average firing field

        A Gaussian function with peak in the origin and covariance given by
        `self.firing_field` is plotted on axes identical to those used in
        `self.plot_acorr`.

        This method is essentially a wrapper around
        `self.firing_field_map().plot`, but adds funcionality related to the
        peak threshold.

        Parameters
        ----------
        vmin, vmax
            See `IntensityMap2D.plot` (..note:: defaults may differ).
        threshold : bool or scalar, optional
            Bins with values below the threshold value are masked from the
            plot. If True, `self.params['threshold']` is used as the threshold
            value. If False, no thresholding is performed. Otherwise, the
            provided value is used as the threshold value.
        ffield_kw : dict or None, optional
            Optional keyword arguments to pass to `BaseCell.firing_field_map`.
        **kwargs : dict, optional
            Additional keyword arguments are passed to `IntensityMap2D.plot`.

        Returns
        -------
        See `IntensityMap2D.plot`.

        """
        if ffield_kw is None:
            ffield_kw = {}
        ffield = self.firing_field_map(**ffield_kw)

        if threshold is True:
            threshold = self.params['threshold']
        elif threshold is False:
            threshold = None

        kwargs.update(threshold=threshold, vmin=vmin, vmax=vmax)
        return ffield.plot(**kwargs)


class TemplateGridCell(BaseCell):
    """
    Represent a template grid cell

    The grid cell firing pattern is constructed using lattice vectors and
    a covariance matrix giving the shape of the firing fields.

    Parameters
    ----------
    peaks : array-like, shape (6, 2)
        Array containing the six primary lattice vectors of the grid pattern.
        These vectors are equivalent to the coordinates of the six inner peaks
        in the autocorrelogram of the cell, as returned by e.g.
        `BaseCell.grid_peaks`.
    firing_field : array-like, shape (2, 2)
        Array containing the covariance matrix characterizing the shape of the
        firing fields. See `BaseCell.firing_field` for an explanation.
    bset : BinnedSet2D
        A BinnedSet2D instance defining the region over which to construct the
        firing rate map of the cell.
    **kwargs : dict, optional
        Keyword arguments are passed to `BaseCell`.

    """
    def __init__(self, peaks, firing_field, bset, **kwargs):
        BaseCell.__init__(self, **kwargs)
        if not hasattr(self, 'data'):
            setattr(self, 'data', {})
        self.data.update(
            peaks=peaks,
            firing_field=firing_field,
            bset=bset
        )

    @staticmethod
    def construct_ratemap(peaks, firing_field, bset):
        """
        Construct an idealized grid cell firing rate map

        Parameters
        ----------
        peaks : array-like, shape (6, 2)
            Array containing the six primary lattice vectors of the grid
            pattern.  These vectors are equivalent to the coordinates of the
            six inner peaks in the autocorrelogram of the cell, as returned by
            e.g.  `BaseCell.grid_peaks`.
        firing_field : array-like, shape (2, 2)
            Array containing the covariance matrix characterizing the shape of
            the firing fields. See `BaseCell.firing_field` for an explanation.
        bset : BinnedSet2D
            A BinnedSet2D instance defining the region over which to construct
            the firing rate map of the cell.

        Returns
        -------
        IntensityMap2D
            Firing rate map.

        """
        xcm, ycm = bset.cmesh
        xypoints = numpy.vstack((xcm.ravel(), ycm.ravel())).transpose()

        # Compute the major lattice vectors from one hexagon center to the next
        lattice = peaks + numpy.roll(peaks, 1, axis=0)
        lattice_r1 = numpy.roll(lattice, 1, axis=0)

        # Find the number of rounds of adding hexagonal pattern around the
        # center is needed in order to fill the range defined by the bset
        midlattice = 0.5 * (lattice + lattice_r1)
        mlx, mly = midlattice[:, 0], midlattice[:, 1]
        mldsq = mlx * mlx + mly * mly

        dx, dy = 0.5 * xcm.ptp(), 0.5 * ycm.ptp()
        diagsq = dx * dx + dy * dy

        levels = int(numpy.ceil(numpy.sqrt(diagsq / mldsq.min())))

        pattern = numpy.vstack(((0.0, 0.0), peaks))
        all_peaks = [pattern]
        for i in range(levels + 1):
            for l in range(i):
                k = i - l
                disp = k * lattice + l * lattice_r1
                if (k - l) % 3 == 0:
                    for d in disp:
                        all_peaks.append(pattern + d)
                else:
                    for d in disp:
                        all_peaks.append(d)
        all_peaks = numpy.vstack(all_peaks)

        # Compute the firing rate
        firing_rate = 0.0
        for peak in all_peaks:
            firing_rate += gaussian(xypoints, mean=peak, cov=firing_field)

        # Normalize to a mean firing rate of 1.0
        firing_rate /= numpy.mean(firing_rate)

        # Reshape to the shape of xcm and ycm
        firing_rate = firing_rate.reshape(xcm.shape)

        return IntensityMap2D(firing_rate, bset)

    @memoize_method
    def ratemap(self, normalize_ratemean=False, **kwargs):
        """
        Compute the firing rate map of the template grid cell

        Parameters
        ----------
        normalize_ratemean : bool, optional
            This keyword has no effect here: since the firing rate of
            a TemplateGridCell is synthetic, it is always normalized to mean
            1.0.
        **kwargs : dict, optional
            Not in use.

        Returns
        -------
        IntensityMap2D
            Firing rate map.

        """
        data = self.data
        return self.construct_ratemap(data['peaks'], data['firing_field'],
                                      data['bset'])

    def _peaks(self, *args, **kwargs):
        return self.data['peaks']

    def firing_field(self, **kwargs):
        return self.data['firing_field']


class Cell(BaseCell):
    """
    Represent a real cell with spatially modulated firing rate

    The cell is based on recorded spike and position data.

    Parameters
    ----------
    spike_t : array-like
        Array giving the times at which the cell spiked.
    pos : Position
        Position instance representing the movement of the animal.
    bins, range_ (range is optional)
        Bin and range specification. See `Position.occupancy` for details.
    filter_size : scalar, optional
        Default smoothing width to use when computing the firing rate map.
        Given in the same units as the coordinates in `pos`.
    threshold : scalar, optional
        See `BaseCell`
    **kwargs
        See `BaseCell`.

    """
    def __init__(self, position, spike_t, bins, range_=None, filter_size=0.0,
                 threshold=0.0, **kwargs):
        BaseCell.__init__(self, threshold=threshold, **kwargs)
        for name in ('params', 'data'):
            if not hasattr(self, name):
                setattr(self, name, {})
        self.params.update(
            filter_size=filter_size,
            bins=bins,
            range_=range_
        )

        spike_t = numpy.squeeze(spike_t)

        posdata = position.filtered_data()
        spike_x, spike_y = self.interpolate_spikes(spike_t,
                                                   posdata['t'],
                                                   posdata['x'],
                                                   posdata['y'])

        self.data.update(
            spike_t=spike_t,
            spike_x=spike_x,
            spike_y=spike_y,
        )

        memoize_method.register_friend(position, self)
        self.position = position

    @staticmethod
    def interpolate_spikes(spike_t, t, x, y):
        """
        Find the locations of spikes in a spike train

        The arrays giving positions in which to interpolate may be masked or
        contain nans, in which case spikes occuring in the corresponding times
        will be discarded.

        Parameters
        ----------
        spike_t : array-like
            Array giving spike times.
        t, x, y : array-like
            Arrays giving the time, x-coordinate and y-coordinate of the
            position samples. If any of the arrays are masked or contain nans,
            spikes occuring in the corresponding time intervals will be masked.

        Returns
        -------
        spike_x, spike_y : ndarray
            Array of spike x and y coordinates.

        """
        xf = numpy.ma.array(x, dtype=numpy.float_)
        yf = numpy.ma.array(y, dtype=numpy.float_)
        xf = numpy.ma.filled(xf, fill_value=numpy.nan)
        yf = numpy.ma.filled(yf, fill_value=numpy.nan)

        spike_xf = numpy.interp(spike_t, t, xf)
        spike_yf = numpy.interp(spike_t, t, yf)
        mask = numpy.logical_or(numpy.isnan(spike_xf), numpy.isnan(spike_yf))

        spike_x = numpy.interp(spike_t, t, x)
        spike_y = numpy.interp(spike_t, t, y)
        spike_x = numpy.ma.masked_where(mask, spike_x)
        spike_y = numpy.ma.masked_where(mask, spike_y)
        return spike_x, spike_y

    @memoize_method
    def _spikemap(self, bins, range_):
        """
        Compute a map of the recorded spikes

        This part of the computation in `Cell.spikemap` is factored out
        to optimize memoization.

        """
        # We use self._occupancy to instantiate the BinnedSet, to avoid
        # creating unneccesary duplicate objects.
        occupancy = self._occupancy(bins, range_)
        bset = occupancy.bset

        data = self.data
        spike_x, spike_y = data['spike_x'], data['spike_y']
        spike_hist, __, __ = numpy.histogram2d(numpy.ma.compressed(spike_x),
                                               numpy.ma.compressed(spike_y),
                                               bins=(bset.xedges, bset.yedges),
                                               normed=False)
        return IntensityMap2D(spike_hist, bset)

    def spikemap(self, distribution_spikemap=False, bins=None, range_=None,
                 filter_size=None, filter_='gaussian', normalize_spikemap=True,
                 **kwargs):
        """
        Compute a map of the spatial spike distribution

        Parameters
        ----------
        distribution_spikemap : bool, optional
            If True, the spike map will be normalized to have sum 1, such
            that it can be interpreted as a spatial frequency distribution map
            for the spikes.
        bins, range_
            See `Cell.occupancy`.
        filter_size : scalar
            Characteristic length of the smoothing filter to apply to the
            spikemap. If None, the default specification, set at
            initialization and stored in `self.params`, is used.
        filter_
            See `IntensityMap2D.smooth` (..note:: defaults may differ).
        normalize_occupancy
            Passed as keyword 'normalize' to `IntensityMap2D.smooth`.
        **kwargs : dict, optional
            Keyword arguments are passed to `Cell.nspikes`.

        Returns
        -------
        IntensityMap2D
            Spike map.

        """
        if filter_size is None:
            filter_size = self.params['filter_size']

        spikemap = self._spikemap(bins, range_)
        if distribution_spikemap:
            spikemap /= self.nspikes(**kwargs)

        return spikemap.smooth(filter_size, filter_=filter_,
                               normalize=normalize_spikemap)

    def nspikes(self, **kwargs):
        """
        Count the number of spikes recorded from this cell

        Parameters
        ----------
        **kwargs : dict, optional
            Not in use.

        Returns
        -------
        integer
            Number of spikes.

        """
        return len(numpy.ma.compressed(self.data['spike_x']))

    def _occupancy(self, bins, range_, distribution_occupancy=False):
        """
        Compute a map of the recorded spikes

        This part of the computation in `Cell.occupancy` is factored out
        as a courtesy to `Cell.spikemap`.

        """
        params = self.params
        if bins is None:
            bins = params['bins']
        if range_ is None:
            range_ = params['range_']
        return self.position.occupancy(bins=bins, range_=range_,
                                       distribution=distribution_occupancy)

    def occupancy(self, distribution_occupancy=False, bins=None, range_=None,
                  filter_size=None, filter_='gaussian',
                  normalize_occupancy=True, **kwargs):
        """
        Compute a map of occupancy times

        Parameters
        ----------
        bins, range_
            Bin and range specification. See `Position.occupancy` for details.
            If None, the default specification, set at initialization and
            stored in `self.params`, is used.
        distribution_occupancy : bool, optional
            If True, the occupancy map will be normalized to have sum 1, such
            that it can be interpreted as a frequency distribution for the
            occupancy.
        filter_size : scalar
            Characteristic length of the smoothing filter to apply to the
            occupancy map. If None, the default specification, set at
            initialization and stored in `self.params`, is used.
        filter_
            See `IntensityMap2D.smooth` (..note:: defaults may differ).
        normalize_occupancy
            Passed as keyword 'normalize' to `IntensityMap2D.smooth`.
        **kwargs : dict, optional
            Not in use.

        Returns
        -------
        IntensityMap2D
            Occupancy map.

        """
        if filter_size is None:
            filter_size = self.params['filter_size']
        occupancy = self._occupancy(
            bins=bins, range_=range_,
            distribution_occupancy=distribution_occupancy)
        return occupancy.smooth(filter_size, filter_=filter_,
                                normalize=normalize_occupancy)

    def total_time(self, **kwargs):
        """
        Compute the total recording time

        Parameters
        ----------
        **kwargs : dict, optional
            Not in use.

        Returns
        -------
        scalar
            Total time.

        """
        return self.position.total_time()

    @memoize_method
    def ratemap(self, bins=None, range_=None, normalize_ratemean=False,
                filter_size=None, filter_mode='pre', filter_='gaussian',
                normalize_ratemap=True, **kwargs):
        """
        Compute the firing rate map of the cell

        Parameters
        ----------
        bins, range_
            See `Cell.occupancy`.
        normalize_ratemean : bool, optional
            If True, the firing rate is normalized such that the mean firing
            rate is 1.0.
        filter_size : scalar, optional
            Characteristic length of the smoothing filter used in the
            computation. If None, the default filter size, set at
            initialization and stored in `self.params`, is used.
        filter_mode : {'pre', 'post'}

            ``pre``
                The occupancy map and spike histogram are smoothed
                individually, and then divided to create the firing rate map.
            ``post``
                The histogram is divided by the occupancy to create a raw,
                unsmoothed firing rate map, which is then smoothed.
        filter_
            See `IntensityMap2D.smooth` (..note:: defaults may differ).
        normalize_ratemap
            Passed as keyword 'normalize' to `IntensityMap2D.smooth`.
        **kwargs: dict, optional
            Passed on to `Cell.total_time`.

        Returns
        -------
        IntensityMap2D
            Firing rate map.

        """
        if filter_size is None:
            filter_size = self.params['filter_size']

        if filter_mode == 'pre':
            prefilter_size = filter_size
            postfilter_size = 0.0
        elif filter_mode == 'post':
            prefilter_size = 0.0
            postfilter_size = filter_size
        else:
            raise ValueError("unknown filter mode {}"
                             .format(filter_mode))

        kwargs.update(
            bins=bins,
            range_=range_,
            filter_size=prefilter_size,
            filter_=filter_,
            distribution_occupancy=False,
            distribution_spikemap=False,
            normalize_occupancy=normalize_ratemap,
            normalize_spikemap=normalize_ratemap,
        )

        occupancy = self.occupancy(**kwargs)
        spikemap = self.spikemap(**kwargs)

        ratemap = spikemap / occupancy
        if normalize_ratemean:
            kwargs.update(normalize_ratemean=False)
            ratemap /= self.rate_mean(**kwargs)

        return ratemap.smooth(postfilter_size, filter_=filter_,
                              normalize=normalize_ratemap)

    # This is really a stochastic property and should ideally not be memoized,
    # but ain't nobody got time for that.
    @memoize_method
    def stability(self, **kwargs):
        """
        Compute a measure of the spatial stability of the cell firing pattern

        Returns
        -------
        scalar
            The spatial stability of the cell firing pattern.
        **kwargs : dict, optional
            Keyword arguments are passed to, `Cell.ratemap`,
            `Cell.occupancy`, `Cell.nspikes` and `Cell.total_time`.

        """
        kwargs.update(
            filter_size=2.0 * self.params['filter_size'],
            distribution_occupancy=False,
            normalize_ratemean=False,
        )
        rmap = self.ratemap(**kwargs)

        kwargs.update(filter_size=0.0)

        def _stability_terms(cell1, cell2):
            dev = cell1.ratemap(**kwargs) - cell2.ratemap(**kwargs)
            deviation = dev * dev / rmap

            invtime1 = 1.0 / cell1.occupancy(**kwargs)
            invtime2 = 1.0 / cell2.occupancy(**kwargs)
            invtime = invtime1 + invtime2

            return deviation.sum(), invtime.sum()

        def _interval_length(int_):
            return int_[1] - int_[0]

        def _intervals():
            split1, split2 = numpy.sort(numpy.random.random_sample(2))
            ints = ((0.0, split1), (split1, split2), (split2, 1.0))
            return sorted(ints, key=_interval_length, reverse=True)

        deviation_sum = 0.0
        invtime_sum = 0.0
        trials = 50
        for _ in range(trials):
            int1, int2, _ = _intervals()
            new_cell1 = self.subinterval(*int1)
            new_cell2 = self.subinterval(*int2)
            deviation, invtime = _stability_terms(new_cell1, new_cell2)
            deviation_sum += deviation
            invtime_sum += invtime

        mean_sfactor = deviation_sum / invtime_sum

        def _transform(x):
            return 1.0 / (1.0 + x)
            #return numpy.exp(-x)

        return _transform(mean_sfactor)

    def subinterval(self, start, stop):
        """
        Create a Cell object from the data recored in part of the session

        Parameters
        ----------
        start, stop : scalar in [0, 1]
            Start and stop times for the subinterval to use, given as fractions
            of the total length of the recording session.

        Returns
        -------
        Cell
            New Cell object based on the data recorded between
            ..math:`t_0 + (t_f - t_0) * start` and ..math:`t_0 + (t_f - t_0)
            * stop`.

        """
        data = self.data
        pos = self.position
        posdata = pos.data
        posparams = pos.params

        t, x, y = posdata['t'], posdata['x'], posdata['y']
        t_0, t_1 = t[(0, -1), ]
        length = t_1 - t_0

        t_start, t_stop = t_0 + length * start, t_0 + length * stop
        posindex = numpy.logical_and(t_start < t, t <= t_stop)
        new_t, new_x, new_y = t[posindex], x[posindex], y[posindex]
        new_pos = Position(new_t, new_x, new_y,
                           speed_window=posparams['speed_window'],
                           min_speed=posparams['min_speed'])

        spike_t = data['spike_t']
        spikeindex = numpy.logical_and(t_start < spike_t, spike_t <= t_stop)
        new_spike_t = data['spike_t'][spikeindex]

        kwargs = self.info
        kwargs.update(self.params)

        return Cell(position=new_pos, spike_t=new_spike_t, **kwargs)

    def resample_spikes(self, length=1.0, replace=True):
        """
        Create a Cell object by resampling spikes with replacement

        Parameters
        ----------
        length : float, optional
            Number of spikes in the resampled spike train, expressed as
            a fraction of the number of spikes in the original spike train.
        replace : bool, optional
            If True, the resampling is performed with replacement. If False,
            it is performed without replacement. Note that if False, length
            must be `<= 1.0`.

        Returns
        -------
        Cell
            New Cell object based on the data recorded between
            ..math:`t_0 + (t_f - t_0) * start` and ..math:`t_0 + (t_f - t_0)
            * stop`.

        """
        spike_t = self.data['spike_t']
        n = int(length * len(spike_t))
        new_spike_t = numpy.random.choice(spike_t, size=n, replace=replace)
        return self.new_spike_t(new_spike_t)

    def new_spike_t(self, new_spike_t):
        """
        Create a new Cell object with the same position data, parameters and
        info as this one, but new spiking times

        Parameters
        ----------
        new_spike_t : array-like
            Array with the new spike times

        Returns
        -------
        Cell
            New Cell object using the new spike times

        """
        kwargs = self.info
        kwargs.update(self.params)

        return Cell(position=self.position, spike_t=new_spike_t, **kwargs)

    def plot_spikes(self, axes=None, path=False, marker='o', alpha=0.25,
                    zorder=10, **kwargs):
        """
        Plot the spatial location of the recorded spikes

        The spikes can be added to an existing plot via the optional 'axes'
        argument.

        Parameters
        ----------
        axes : Axes or None, optional
            Axes instance to add the path to. If None, the most current Axes
            instance with `aspect='equal'` is grabbed or created.
        path : bool, optional
            If True, plot the path through the spikes using some hopefully
            sensible defaults. For better control, leave this False and use
            `Position.plot_path` instead.
        marker : matplotlib marker spec, optional
            Marker to use when drawing the spikes.
        alpha : scalar in [0, 1], optional
            The opacity of the markers.
        zorder : integer, optional
            Number determining the plotting order. Increase this if the spikes
            tend to be hidden behind other plotted features (e.g.  the path).
        **kwargs : dict, optional
            Additional keyword arguments passed are passed to `axes.plot`. Note
            in particular the keywords 'color', 'markersize' and 'label'.

        Returns
        -------
        list
            List containing the plotted Line2D instances

        """
        if axes is None:
            axes = pyplot.gca(aspect='equal')

        data = self.data

        h = []
        if path:
            pos = self.position
            plines = pos.plot_path(axes=axes)
            h += plines

        kw = dict(linestyle='None', marker=marker, alpha=alpha, zorder=zorder)
        kw.update(kwargs)

        h = axes.plot(data['spike_x'], data['spike_y'], **kw)

        range_ = self.params['range_']
        if range_ is not None:
            axes.set_xlim(range_[0])
            axes.set_ylim(range_[1])

        return h


class CellCollection(AlmostImmutable, MutableSequence):
    """
    Collect a number of Cell instances and provide methods for feature
    extraction, clustering, and plotting.

    This class is a sequence type, and individual cells can be accessed, added
    and removed by standard subscripting syntax.

    Parameters
    ----------
    cells : sequence
        Sequence of Cell instances.
    **kwargs : dict, optional
        Any additional keyword arguments are stored as a dict in the attribute
        `info`. They are not used for internal computations.

    """
    _cells = Memparams(list, '_cells')

    def __init__(self, cells, **kwargs):
        self._cells = cells
        self.info = kwargs

        for cell in self:
            memoize_method.register_friend(cell, self)

    # Implement abstract methods
    def __getitem__(self, index, *args, **kwargs):
        return self._cells.__getitem__(index, *args, **kwargs)

    def __setitem__(self, *args, **kwargs):
        return self._cells.__setitem__(*args, **kwargs)

    def __delitem__(self, *args, **kwargs):
        return self._cells.__delitem__(*args, **kwargs)

    def __len__(self, *args, **kwargs):
        return self._cells.__len__(*args, **kwargs)

    def insert(self, *args, **kwargs):
        return self._cells.insert(*args, **kwargs)

    # Override certain possibly very slow mixins
    def __iter__(self, *args, **kwargs):
        return self._cells.__iter__(*args, **kwargs)

    def __reversed__(self, *args, **kwargs):
        item = self._cells.__reversed__(*args, **kwargs)
        return type(self)(item, **self.info)

    def index(self, *args, **kwargs):
        return self._cells.index(*args, **kwargs)

    # Implement equality comparison
    def __eq__(self, other):
        return type(other) == type(self) and other._cells == self._cells

    def __ne__(self, other):
        return not self.__eq__(other)

    @staticmethod
    def _cell_list_from_session(session, bins, range_, position_kw, cell_kw):
        """
        Create a list of instantiated Cell objects from a single experimental
        session

        Contains the common part of the factory methods `from_session` and
        `from_multiple_sessions`.

        """
        pkw = dict(position_kw)
        for key in ('info', 'params'):
            if key in session:
                pkw.update(session[key])

        position = Position(session['t'], session['x'], session['y'], **pkw)

        cells = session['cells']
        clist = []
        for cell in cells:
            ckw = dict(cell_kw)
            for key in ('info', 'params'):
                if key in cell:
                    ckw.update(cell[key])

            ckw.update(position=position, spike_t=cell['spike_t'], bins=bins,
                       range_=range_)
            clist.append(Cell(**ckw))
        return clist

    @classmethod
    def from_session(cls, session, bins, range_=None, position_kw=None,
                     cell_kw=None, **kwargs):
        """
        Construct a CellCollection from a single experimental session

        Parameters
        ----------
        session : dict
            Dict containing the following fields:

            ``'t' : array-like``
                Array containing the times of the position samples.
            ``'x', 'y' : array-like``
                Arrays containing the x- and y-positions of the position
                samples.  Missing samples can be represented by nans or by
                using a masked array for at least one of the arrays.
            ``'cells': sequence``
                Sequence where each element is a dict containing the
                information for a particular cell in the following fields:

                ``'spike_t': array-like``
                    Array giving the times at which the cell spiked.
                ``'params': dict, optional``
                    Dict containing parameters to be passed as keyword
                    arguments to the Cell instantiation.
                ``'info': dict, optional``
                    Dict containing information about the cell. Passed as
                    keyword arguments to the Cell instantiation.
            ``'params': dict, optional``
                Dict containing parameters to be passed as keyword arguments to
                the Position instantiation.
            ``'info': dict, optional``
                Dict containing information about the session. Passed as
                keyword arguments to the Position instantiation.
        bins, range_ (range is optional)
            See `Cell`.
        position_kw : dict, optional
            Extra keyword arguments to `Position`. The keyword arguments from
            the 'params' and 'info' fields in the elements of
            `session['cells']` takes precedence in case of conflicts.
        cell_kw : dict, optional
            Extra keyword arguments to `Cell`. The keyword arguments from
            the 'params' and 'info' fields in `session` takes precedence in
            case of conflicts.
        **kwargs : dict, optional
            Any additional keyword arguments are stored as a dict in the
            attribute `info` on the CellCollection instance. They are not used
            for internal computations.

        """
        if position_kw is None:
            position_kw = {}
        if cell_kw is None:
            cell_kw = {}

        clist = cls._cell_list_from_session(session, bins, range_,
                                            position_kw, cell_kw)
        return cls(clist, **kwargs)

    @classmethod
    def from_multiple_sessions(cls, sessions, bins, range_=None,
                               position_kw=None, cell_kw=None, **kwargs):
        """
        Construct a CellCollection from multiple experimental sessions

        For each Cell instance `cell`, `cell.info['session']` will be set to
        the index of the session the cell was recorded in. This can be used to
        look up cells from the same session with the `CellCollection.lookup`
        method.

        Parameters
        ----------
        sessions : dict
            Dict containing the following fields:

            ``'sessions' : sequence``
                Sequence containing a dict for each experimental session. See
                `CellCollection.from_session` for an explanation of the format
                of each of the session dicts.
            ``'params': dict, optional``
                Dict containing parameters to be passed as keyword arguments to
                the CellCollection instantiation.
            ``'info': dict, optional``
                Dict containing information common to all sessions. Will be
                stored in the attribute `info` on the CellCollection instance.
        bins, range_ (range is optional)
            See `Cell`.
        position_kw : dict, optional
            Extra keyword arguments to `Position`. The keyword arguments from
            the 'params' and 'info' fields in `session` takes precedence in
            case of conflicts.
        cell_kw : dict, optional
            Extra keyword arguments to `Cell`. The keyword arguments from the
            'params' and 'info' fields in the elements of `session['cells']`
            takes precedence in case of conflicts.
        **kwargs : dict, optional
            Any additional keyword arguments are stored as a dict in the
            attribute `info` on the CellCollection instance. They are not used
            for internal computations.

        """
        if position_kw is None:
            position_kw = {}
        if cell_kw is None:
            cell_kw = {}

        clist = []
        for (i, session) in enumerate(sessions['sessions']):
            clist += cls._cell_list_from_session(session, bins, range_,
                                                 position_kw, cell_kw)

        for key in ('info', 'params'):
            if key in sessions:
                kwargs.update(sessions[key])

        return cls(clist, **kwargs)

    @classmethod
    def from_labels(cls, cells, labels, min_length=4, **kwargs):
        """
        Use a list of keys and corresponding labels to instantiate several
        CellCollection instances

        Parameters
        ----------
        cells : sequence of cells
            Sequence to pick cells for the CellCollections from.
        labels : sequence
            Sequence of labels with indices corresponding to the indices in
            `cells`: all cells with the same label become a collection. The
            special label `-1` denotes outliers.
        min_length : integer, optional
            The minimum number of cells in a collection. Potential collections
            with fewer cells than this are merged into the outliers.
        **kwargs : dict, optional
            Keyword arguments passed to the `CellCollection` constructor.

        Returns
        -------
        collections : list
            List of new CellCollection instances.
        outliers : CellCollection
            Collection of outlier cells.

        """
        tentative_collections = [[] for _ in range(max(labels) + 2)]
        for (cell, label) in zip(cells, labels):
            coll = tentative_collections[label].append(cell)
        collections = []
        outliers = CellCollection(tentative_collections.pop())
        for coll in tentative_collections:
            if len(coll) < min_length:
                outliers += coll
            else:
                collections.append(cls(coll, **kwargs))

        return collections, outliers

    def lookup(self, info):
        """
        Look up cells through their `info` attribute

        Parameters
        ----------
        info : dict
            Dict containing fields that should be matched in either `cell.info`
            or `cell.pos.info` in all returned Cell instances `cell`.

        Returns
        -------
        CellCollection
            Collection of Cell instances mathcing `info`.

        """
        clist = []
        for cell in self:
            for key, value in info.items():
                if key in cell.info:
                    cvalue = cell.info[key]
                    if cvalue == value:
                        clist.append(cell)
                        continue
                try:
                    posinfo = cell.pos.info
                except AttributeError:
                    pass
                else:
                    if key in posinfo:
                        cpvalue = posinfo[key]
                        if cpvalue == value:
                            clist.append(cell)

        return type(self)(clist, **self.info)

    def _mean_attribute(self, attr, **kwargs):
        """
        Compute the mean of an attribute of the cells in the CellCollection

        Parameters
        ----------
        attr : string
            Name of the Cell attribute to compute the mean over. It is assumed
            that for a Cell instance `cell`, `getattr(cell, attr)`  is
            a callable returning the desired object.
        **kwargs : dict, optional
            Keyword arguments are passed to `getattr(cell, attr)`.

        Returns
        -------
        mean :
            Mean of the attribute across the cells in the CellCollection.

        """
        attrsum = sum(getattr(cell, attr)(**kwargs) for cell in self)
        ninv = 1.0 / len(self)
        return ninv * attrsum

    def _mean_peak_attribute(self, attr, roll=0, **kwargs):
        """
        Compute the mean of an attribute of the peaks of the cells in the
        collection

        Here, the attributes are assumed to be arrays with a value for each of
        the peaks in the inner ring, such that the arrays must be rolled into
        maximal peak alignment before computing the mean (see `BaseCell.roll`
        for explanation of roll).

        Parameters
        ----------
        attr : string
            Name of the Cell attribute to compute the mean over. It is assumed
            that for a Cell instance `cell`, `getattr(cell, attr)`  is
            a callable returning the desired object.
        roll : integer, optional
            Global roll of the peak order of the cell. See `BaseCell.roll` for
            the full explanation of the `roll` parameter.  ..note:: The
            relative roll between cells necessary for for alignment of cell
            peaks is automatically handled by this function.  This parameter
            only controls an optional global roll, applied to all cells.
        **kwargs : dict, optional
            Keyword arguments are passed to `getattr(cell, attr)` and
            `BaseCell.roll`.

        Returns
        -------
        mean :
            Mean of the peak attribute across the cells in the CellCollection.

        """
        refcell = self[0]
        attrsum = 0.0
        for cell in self:
            cellroll = roll + cell.roll(refcell, **kwargs)
            attrsum += numpy.roll(getattr(cell, attr)(**kwargs), cellroll,
                                  axis=0)
        ninv = 1.0 / len(self)
        return ninv * attrsum

    def mean_firing_field(self, **kwargs):
        """
        Compute the mean firing field of cells in the collection

        Parameters
        ----------
        **kwargs : dict, optional
            Keyword arguments are passed to `cell.firing_field` for each cell.

        Returns
        -------
        ndarray
            The mean firing field covariance matrix.

        """
        return self._mean_attribute('firing_field', **kwargs)

    def mean_ellpars(self, cartesian_ellpars=False, **kwargs):
        """
        Compute the mean of the ellipse parameters of the cells in the
        CellCollection

        Parameters
        ----------
        cartesian_ellpars : bool, optional
            If True, the parameters are returned in cartesian coordinates.
            If False, they are returned in cartesian coordinates.
            In all cases, the means are computed in cartesian coordinates.
        **kwargs : dict, optional
            Keyword arguments are passed to `cell.ellpars` for each cell.

        Returns
        -------
        ndarray
            The mean ellipse parameters.

        """
        kwargs.update(cartesian_ellpars=True)
        ell_cart = self._mean_attribute('ellpars', **kwargs)
        if cartesian_ellpars:
            return ell_cart
        return numpy.array((numpy.sqrt(numpy.sum(ell_cart * ell_cart)),
                            numpy.arctan2(ell_cart[1], ell_cart[0])))

    def mean_grid_peaks(self, polar_peaks=False, **kwargs):
        """
        Compute the mean of the coordinates of the inner six peaks in the
        autocorrelogram of cells in the CellCollection

        Parameters
        ----------
        polar_peaks : bool, optional
            If True, the mean peak coordinates are returned as polar
            coordinates. If False, they are returned as cartesian coordinates.
            In all cases, the means are computed in cartesian coordinates.
        **kwargs : dict, optional
            Keyword arguments are passed to `cell.firing_field` for each cell.

        Returns
        -------
        ndarray
            Mean peak coordinates.

        """
        kwargs.update(polar_peaks=False)
        peaks = self._mean_peak_attribute('grid_peaks', **kwargs)
        if polar_peaks:
            px, py = peaks[:, 0], peaks[:, 1]
            radii = numpy.sqrt(px * px + py * py)
            angles = numpy.arctan2(py, px)
            peaks = numpy.column_stack((radii, angles))

        return peaks

    def stacked_ratemap(self, normalize='mean', threshold=None, **kwargs):
        """
        Compute the stacked firing rate map of cells in the collection

        The stacked firing rate map is the binwise average of the firing
        rate maps of all the cells.

        Parameters
        ----------
        normalize : {None, 'max', 'mean', 'std', 'zscore'}, optional
            Flag to choose the mode of rate map normalization before averaging.
            Possible values:

            ``None``
                No normalization si performed.
            ``'max'``
                The maximum value of each rate map is normalized to 1.0.
            ``'mean'``
                The mean firing rate of each cell is normalized to 1.0.
            ``'std'``
                The standard deviation of the firing rate of each cell is
                normalized to 1.0.
            ``'zscore'``
                The firing rate of each cell is normalized to its Z-score by
                subtracting the mean firing rate and dividing by the standard
                deviation.
        threshold : scalar or None, optional
            If not None, each firing rate will be transformed to a binary map
            with the value 1 in bins where the normalized firing rate exceeds
            the threshold value, and 0 otherwise.
        **kwargs : dict, optional
            Additional keyword arguments are passed to each invocation of
            `BaseCell.ratemap`, `BaseCell.rate_mean` and `BaseCell.rate_std`.

        Returns
        -------
        IntensityMap2D
            The stacked firing rate map.

        """
        if normalize is None:
            def _norm(cell):
                return cell.ratemap(**kwargs)
        elif normalize == 'max':
            def _norm(cell):
                ratemap = cell.ratemap(**kwargs)
                return ratemap / ratemap.max()
        elif normalize == 'mean':
            def _norm(cell):
                return cell.ratemap(**kwargs) / cell.rate_mean(**kwargs)
        elif normalize == 'std':
            def _norm(cell):
                return cell.ratemap(**kwargs) / cell.rate_std(**kwargs)
        elif normalize == 'zscore':
            def _norm(cell):
                return ((cell.ratemap(**kwargs) - cell.rate_mean(**kwargs))
                        / cell.rate_std(**kwargs))
        else:
            raise ValueError("unknown normalization: {}".format(normalize))

        if threshold is None:
            norm = _norm
        else:
            def norm(cell):
                ratemap = _norm(cell)
                return (ratemap > threshold).astype(numpy.float_)

        return IntensityMap2D.mean_map((norm(cell) for cell in self),
                                       ignore_missing=True)

    def distances(self, other=None, **kwargs):
        """
        Compute a distance matrix between the cells in CellCollection instances

        Parameters
        ----------
        other : None or iterable, optional
            If `other` is an iterable, the distance matrix between the cells in
            `cell` and the elements in `other`. If None, the distance matrix is
            computed between the cells in this instance.
        kwargs : dict, optional
            Additional keyword arguments to pass to `BaseCell.distance`.

        Returns
        -------
        DataFrame
            The distance matrix, indexed along rows by the Cell instances in
            `self`, and along columns by the Cell instance(s) in `other`. Both
            row and column order reflect the order of cells in the
            CellCollections.

        """
        if other is None:
            other = self

        distdict = {cell1: {cell2: cell1.distance(cell2, **kwargs)
                            for cell2 in other} for cell1 in self}

        # Order the distance matrix using `loc`
        return pandas.DataFrame(distdict).loc[self, other]

    def rolls(self, other=None, **kwargs):
        """
        Compute a roll matrix between the cells in CellCollection instances

        Parameters
        ----------
        other : None or iterable, optional
            If `other` is an iterable, the roll matrix between the cells in
            `cell` and the elements in `other`. If None, the roll matrix is
            computed between the cells in this instance.
        kwargs : dict, optional
            Additional keyword arguments to pass to `BaseCell.roll`.

        Returns
        -------
        DataFrame
            The roll matrix, indexed along rows by the Cell instances in
            `self`, and along columns by the Cell instances in `other`. Both
            row and column order reflect the order of cells in the
            CellCollections.

        """
        if other is None:
            other = self

        rolldict = {cell1: {cell2: cell1.roll(cell2, **kwargs)
                            for cell2 in other} for cell1 in self}

        # Order the roll matrix using `loc`
        return pandas.DataFrame(rolldict).transpose().loc[self, other]

    def features(self, index=FeatureNames.default_features, weights=None,
                 roll=0, **kwargs):
        """
        Compute a feature array of the cells

        The feature series comprising the array are computed, using the peak
        roll for each cell required for global consistency.

        Parameters
        ----------
        index, weights
            See `BaseCell.features`.
        roll : integer, optional
            Global roll of peak order of the cell. See `BaseCell.roll` for the
            full explanation of the `roll` parameter.  ..note:: The relative
            roll between cells necessary for for alignment of cell peaks is
            automatically handled by this function if necessary (that is, if
            `index` contains one or more names from `gridcell.grid_features`).
            This parameter only controls an optional global roll, applied to
            all cells.
        kwargs : dict, optional
            Keyword arguments are passed to `BaseCell.features` and
            `BaseCell.roll`.

        Returns
        -------
        DataFrame
            Feature array. The DataFrame row indices are the `Cell` instances,
            while the DataFrame columns contain the features and are labelled
            by the index from `BaseCell.features`. The row order reflects the
            order of the elements in the CellCollection.

        """
        relroll = pandas.Series({cell: 0 for cell in self})
        for ind in index:
            if ind in FeatureNames.peak_features:
                refcell = self[0]
                relroll = self.rolls(other=(refcell,), **kwargs)[refcell]
                break

        kwargs.update(index=index, weights=weights)
        featdict = {cell: cell.features(roll=(roll + relroll[cell]), **kwargs)
                    for cell in self}

        # Order the feature array using `loc`
        return pandas.DataFrame(featdict).transpose().loc[self]

    def dbscan(self, eps, min_samples, feat_kw=None, **kwargs):
        """
        Use the DBSCAN clustering algorithm to find modules

        Parameters
        ----------
        eps : scalar
            Maximum distance for points to be counted as neighbors.
        min_samples : integer
            Minimum number of neighbors for a point to be considered a core
            point.
        feat_kw : dict, optional
            Keyword arguments to pass to `CellCollection.features`.
        **kwargs : dict, optional
            Additional keyword arguments are passed on to `cluster.dbscan`.

        Returns
        -------
        labels : sequence
            Sequence of labels with indices corresponding to the indices in
            `self`: all cells with the same label belong to a module. The
            special label `-1` denotes outliers. This output can be fed
            directly to `CellCollection.from_labels` to instantiate new
            `CellCollection` instances based on the labels.

        """
        if feat_kw is None:
            feat_kw = {}

        features = self.features(**feat_kw)
        kwargs.update(eps=eps, min_samples=min_samples)
        return cluster.dbscan(features.values, **kwargs)[1]

    def mean_shift(self, feat_kw=None, cluster_all=False, **kwargs):
        """
        Use the mean shift clustering algorithm to find modules

        Parameters
        ----------
        feat_kw : dict, optional
            Keyword arguments to pass to `CellCollection.features`.
        cluster_all : bool, optional
            Whether to assign all cells to a cluster. See `cluster.mean_shift`
            for more details. ..note:: default value may be different here.
        **kwargs : dict, optional
            Additional keyword arguments are passed on to `cluster.mean_shift`.

        Returns
        -------
        labels : sequence
            Sequence of labels with indices corresponding to the indices in
            `self`: all cells with the same label belong to a module. The
            special label `-1` denotes outliers. This output can be fed
            directly to `CellCollection.from_labels` to instantiate new
            `CellCollection` instances based on the labels.

        """
        if feat_kw is None:
            feat_kw = {}

        features = self.features(**feat_kw)

        kwargs.update(cluster_all=cluster_all)
        return cluster.mean_shift(features.values, **kwargs)[1]

    def k_means(self, n_clusters, n_runs=1, feat_kw=None, **kwargs):
        """
        Use the K-means clustering algorithm to find modules

        Parameters
        ----------
        n_clusters : integer
            The number of clusters (and thus modules) to form.
        n_runs : integer, optional
            The number of times to run the K-means algorithm. Each run is
            initialized with a new random state, and the run ending at
            the lowest intertia criterion is used.
        feat_kw : dict, optional
            Keyword arguments to pass to `BaseCell.features`.
        **kwargs : dict, optional
            Additional keyword arguments are passed on to `cluster.k_means`.

        Returns
        -------
        labels : sequence
            Sequence of labels with indices corresponding to the indices in
            `self`: all cells with the same label belong to a module. The
            special label `-1` denotes outliers. This output can be fed
            directly to `CellCollection.from_labels` to instantiate new
            `CellCollection` instances based on the labels.

        """
        if feat_kw is None:
            feat_kw = {}

        features = self.features(**feat_kw)
        fvals = features.values

        __, labels, inertia = cluster.k_means(fvals, n_clusters, **kwargs)
        for _ in range(n_runs - 1):
            __, lbls, inrt = cluster.k_means(fvals, n_clusters, **kwargs)
            if inrt < inertia:
                inertia = inrt
                labels = lbls

        return labels

    def plot_features(self, index, feat_kw=None, axes=None, marker='o',
                      mean=True, mean_kw=None, **kwargs):
        """
        Plot a selection of features of the cells in the CellCollection

        The features can be added to an existing plot via the optional 'axes'
        argument. In this case, the features markers are added to the right of
        the current x limits.

        Parameters
        ----------
        index : sequence
            Index of features to plot. See `BaseCell.features` for explanation.
        feat_kw : dict, optional
            Keyword arguments to pass to `BaseCell.features`.
        axes : Axes, optional
            Axes instance to add the features to. If None, the current Axes
            instance is used if any, or a new one created.
        marker : valid matplotlib marker specification, optional
            Marker used to plot the features.
        mean : bool, optional
            If True, add a line showing the mean of each feature. By default
            this will be a gray line (color='0.50') with linewidth 0.5, but
            this can be overridden using the parameter `mean_kw`.
        mean_kw : dict, optional
            Keyword arguments to pass to `axes.plot` when plotting the mean.
        kwargs : dict, optional
            Additional keyword arguments to pass to `axes.plot`.  Note in
            particular the keywords 'markersize', 'color' and 'label'.

        Returns
        -------
        list
            List containing the plotted Line2D instances.

        """
        if axes is None:
            axes = pyplot.gca()

        if feat_kw is None:
            feat_kw = {}

        features = self.features(index=index, **feat_kw)

        # Start plotting from the current right end of the plot
        xlim = axes.get_xlim()
        right = xlim[1]
        xlocs = numpy.arange(right, right + len(features))

        kw = dict(linestyle='None', marker=marker)
        kw.update(kwargs)
        lines = []
        for ind in index:
            lines += axes.plot(xlocs, features[ind], **kw)

        if mean:
            mkw = dict(color='0.50', linewidth=0.5)
            if mean_kw is not None:
                mkw.update(mean_kw)
            for ind in index:
                m = numpy.empty_like(xlocs, dtype=numpy.float_)
                m.fill(features[ind].mean())
                lines += axes.plot(xlocs, m, **mkw)

        # Add ticklabels and rotate
        #add_ticks(axes.xaxis, xlocs, features.index)
        #pyplot.xticks(axes=axes, rotation='vertical')

        # Set limits so the plot is ready for another round
        axes.set_xlim((xlim[0], xlocs[-1] + 1.0))

        return lines

    def plot_ellpars(self, ellpars_kw=None, axes=None, keys=None, marker='o',
                     mean=None, mean_kw=None, **kwargs):
        """
        Plot the ellipse parameters for cells in the cell collection

        The parameters are visualized in a polar plot with coordinates as
        explained in `BaseCell.ellpars`.

        Parameters
        ----------
        ellpars_kw : dict, optional
            Keyword arguments to pass to `BaseCell.ellpars`.
        axes: Axes, optional
            Axes instance to add the ellipse parameters to. If None, the
            current Axes instance is used if any, or a new one created.
        marker : valid matplotlib marker specification.
            Marker to use plot the ellipse parameters as.
        mean : bool, optional
            If True, add a point showing the mean of the ellipse parameters.
            The mean is computed in the cartesian representation of the ellipse
            parameters. By default, the mean is plotted using a grey marker
            (color='0.50') of the same type as the ellipse parameters, but this
            can be overridden using the parameter `mean_kw`.
        mean_kw : dict, optional
            Keyword arguments to pass to `axes.plot` when plotting the mean
            parameters.
        kwargs : dict, optional
            Additional keyword arguments to to pass to `axes.plot`.  Note in
            particular the keywords 'markersize', 'color' and 'label'.

        Returns
        -------
        list
            List containing the plotted Line2D instances.

        """
        if axes is None:
            axes = pyplot.gca(projection='polar')

        epkw = {}
        if ellpars_kw is not None:
            epkw.update(ellpars_kw)
        epkw.update(cartesian_ellpars=False)

        ellpars = numpy.vstack([cell.ellpars(**epkw) for cell in self])

        kw = dict(linestyle='None', marker=marker)
        kw.update(kwargs)
        lines = axes.plot(ellpars[:, 1], ellpars[:, 0], **kw)

        if mean:
            mkw = dict(linestyle='None', marker=marker, color='0.50')
            if mean_kw is not None:
                mkw.update(mean_kw)
            mean_ellpars = self.mean_ellpars(**epkw)
            lines += axes.plot(mean_ellpars[1], mean_ellpars[0], **mkw)

        axes.set_ylim((0.0, 1.0))

        return lines

    def plot_grid_peaks(self, grid_peaks_kw=None, axes=None, marker='o',
                        mean=False, mean_kw=None, **kwargs):
        """
        Plot the peak locations for cells in the cell collection

        Parameters
        ----------
        grid_peaks_kw : dict, optional
            Keyword arguments to pass to `BaseCell.grid_peaks`.
        axes : Axes, optional
            Axes instance to add the peaks to. If None, the current Axes
            instance is used if any, or a new one created.
        marker : valid matplotlib marker specification.
            Marker to use plot the ellipse parameters as.
        mean : bool, optional
            If True, add a point showing the mean coordinates of the peaks.
            By default, the means are plotted using a grey marker
            (color='0.50') of the same type as the ellipse parameters, but this
            can be overridden using the parameter `mean_kw`.
        mean_kw : dict, optional
            Keyword arguments to pass to `axes.plot` when plotting the mean
            peak coordinates.
        kwargs : dict, optional
            Additional keyword arguments to to pass to `axes.plot`.  Note in
            particular the keywords 'markersize', 'color' and 'label'.

        Returns
        -------
        list
            List containing the plotted Line2D instances.

        """
        if axes is None:
            axes = pyplot.gca(projection='polar')

        gpkw = {}
        if grid_peaks_kw is not None:
            gpkw.update(grid_peaks_kw)
        gpkw.update(polar_peaks=True)

        peaks = numpy.vstack([cell.grid_peaks(**gpkw) for cell in self])

        kw = dict(linestyle='None', marker=marker)
        kw.update(kwargs)
        lines = axes.plot(peaks[:, 1], peaks[:, 0], **kw)

        if mean:
            mkw = dict(linestyle='None', marker=marker, color='0.50')
            if mean_kw is not None:
                mkw.update(mean_kw)
            mean_peaks = self.mean_grid_peaks(**gpkw)

            lines += axes.plot(mean_peaks[:, 1], mean_peaks[:, 0], **mkw)

        return lines

    def plot_stacked_ratemap(self, vmin=0.0, rate_kw=None, **kwargs):
        """
        Plot the stacked firing rate map of the cells in the collection

        This method is essentially a convenience wrapper around
        `self.stacked_ratemap().plot` -- the only difference is that it fixes
        the lower end of the colorbar to 0.0 by default (..note:: depending on
        the normalization used, the minimum value in the stacked rate map may
        be negative, making this default rather useless).

        Parameters
        ----------
        vmin : None or scalar.
            See `IntensityMap2D.plot`.
        rate_kw : dict or None, optional
            Optional keyword arguments to pass to
            `CellCollection.stacked_ratemap`.
        **kwargs : dict, optional
            Additional keyword arguments are passed to `IntensityMap2D.plot`.

        Returns
        -------
        See `IntensityMap2D.plot`.

        """
        if rate_kw is None:
            rate_kw = {}
        kwargs.update(vmin=vmin)
        return self.stacked_ratemap(**rate_kw).plot(**kwargs)

_PI_3 = _PI / 3.0
_COS_PI_3, _SIN_PI_3 = numpy.cos(_PI_3), numpy.sin(_PI_3)

REGULAR_GRID_PEAKS = numpy.array(
    [[1.0, 0.0],
     [_COS_PI_3, _SIN_PI_3],
     [-_COS_PI_3, _SIN_PI_3],
     [-1.0, 0.0],
     [-_COS_PI_3, -_SIN_PI_3],
     [_COS_PI_3, -_SIN_PI_3]])


class Module(CellCollection):
    """
    Collect a number of Cell instances belonging to the same grid cell module
    and provide methods for analyzing the grid phases.

    Parameters
    ----------
    See `CellCollection`.

    """
    def _template_bset(self, **kwargs):
        """
        Compute a BinnedSet2D for the template cell

        The BinnedSet2D is created by extending the BinnedSet2D underlying the
        firing rate maps of the cells in the module to three times the width in
        each dimension.

        Parameters
        ----------
        **kwargs : dict, optional
            Keyword arguments are passed to `BaseCell.ratemap`.

        Returns
        -------
        BinnedSet2D
            BinnedSet2D for the template cell.

        """
        bset = self[0].ratemap(**kwargs).bset
        extension = [(s, s) for s in bset.shape]
        return bset.extend(extension)

    @memoize_method
    def template(self, template_kw=None, **kwargs):
        """
        Construct a template cell for the module

        The template cell is created using the average of the peak coorindates
        of the cells in the module to define the grid pattern, and the average
        of the firing field covariance matrix of the cells in the module to
        define the firing field shape.

        Parameters
        ----------
        template_kw : dict or None, optional
            Keyword arguments passed to the instantiation of the
            TemplateGridCell.
        **kwargs : dict, optional
            Additional keyword arguments are passed to the underlying methods:
            `Module.mean_grid_peaks`, `Module.mean_firing_field`, and
            `BaseCell.ratemap`.

        Returns
        -------
        TemplateGridCell
            The template cell.

        """
        if template_kw is None:
            template_kw = {}
        return TemplateGridCell(self.mean_grid_peaks(**kwargs),
                                self.mean_firing_field(**kwargs),
                                self._template_bset(**kwargs), **template_kw)

    def window_vertices(self, window_type='voronoi', project_phases=False,
                        **kwargs):
        """
        Compute the vertices of the window of possible grid phases

        The window is defined as the central (Voronoi) unit cell in the grid
        pattern of the template cell of the module (see `Module.template`),
        that is, the set of points closer to the central firing field of this
        cell than to any other firing field in its firing pattern.

        Parameters
        ----------
        window_type : {'voronoi', 'rhomboid'}, optional
            If 'voronoi', the central Voronoi unit cell in the grid pattern of
            the template cell of the module is used as window. If 'rhomboid',
            a rhomboidal unit cell with four grid pattern nodes as vertices is
            used as the window.
        project_phases : bool, optional
            If True, the window is derived from a regular hexagonal grid with
            edge lengths 1.0, instead of from the module template cell.
        **kwargs : dict, optional
            Keyword arguments are passed to `Module.template` and
            `TemplateGridCell.grid_peaks`.

        Returns
        -------
        ndarray, shape (6, 2)
            Coordinates of the vertices of the phase window, sorted by angle
            with the x axis.

        """
        if project_phases:
            grid_peaks = REGULAR_GRID_PEAKS
        else:
            pkw = dict(kwargs)
            pkw.update(polar_peaks=False)
            grid_peaks = self.template(**kwargs).grid_peaks(**pkw)

        peak_pattern = numpy.vstack(((0.0, 0.0), grid_peaks))
        if window_type == 'voronoi':
            voronoi = spatial.Voronoi(peak_pattern)
            window = voronoi.vertices[
                voronoi.regions[voronoi.point_region[0]]]
        elif window_type == 'rhomboid':
            window = numpy.vstack((peak_pattern[:2],
                                   peak_pattern[1] + peak_pattern[2],
                                   peak_pattern[2]))
        else:
            raise ValueError("Unknown 'window_type': {}".format(window_type))

        return window

    def window(self, **kwargs):
        """
        Create a Window instance representing the window of possible grid
        phases

        See `self.window_vertices` for an explanation of what the window is all
        about.

        Parameters
        ----------
        **kwargs : dict, optional
            Keyword arguments are passed to `Module.window_vertices`.

        Returns
        -------
        Window
            Window instance representing the grid phase window.

        """
        return Window(self.window_vertices(**kwargs))

    @memoize_method
    def _phases(self, **kwargs):
        """
        Compute the grid phases of the cells in the module

        This part of the computation in `Module.phases` is factored out
        to optimize memoization.

        """
        template = self.template(**kwargs)
        return pandas.DataFrame(
            {cell: cell.phase(template, **kwargs) for cell in self},
            index=('phase_x', 'phase_y')).transpose().loc[self]

    def phases(self, project_phases=False, **kwargs):
        """
        Compute the grid phases of the cells in the module.

        The phase of each cell is defined with respect to the TemplateGridCell
        instance belonging to the module (see `Module.template`).

        Note that the phases returned from this method are the phases computed
        by `BaseCell.phase`. They are not necessarily wrapped into to any
        particular window of possible phases, as defined by
        `Module.window_vertices` and `Module.window`. For a window-aware phase
        pattern, use `Module.phase_pattern`.

        Parameters
        ----------
        project_phases : bool, optional
            If True, phases are projected onto a regular hexagonal grid with
            edge lengths 1.0.
        **kwargs : dict, optional
            Keyword arguments are passed to `BaseCell.phase`,
            `Module.template` and `TemplateGridCell.grid_peaks`.

        Returns
        -------
        DataFrame
            DataFrame containing the x- and y-components of the grid phase of
            each cell relative to the template cell. The rows in the DataFrame
            are indexed by the Cell objects.

        """
        phases = self._phases(**kwargs)
        if project_phases:
            pkw = dict(kwargs)
            pkw.update(polar_peaks=False, project_peaks=True)
            basis = self.template(**kwargs).grid_peaks(**pkw)[:2]
            new_basis = REGULAR_GRID_PEAKS[:2]
            coeffs = project_vectors(phases, basis)
            phases = pandas.DataFrame(coeffs.dot(new_basis),
                                      index=phases.index,
                                      columns=phases.columns)

        return phases

    @memoize_method
    def _pairwise_phases(self, **kwargs):
        """
        Compute the pairwise relative grid phases of the cells in the module

        This part of the computation in `Module.pairwise_phases` is factored
        out to optimize memoization.

        """
        pl = [(cell1, cell2) for cell1 in self for cell2 in self]
              #if cell2 is not cell1]
        return pandas.DataFrame(
            {pair: pair[1].phase(pair[0], **kwargs)
             for pair in pl},
            columns=('phase_x', 'phase_y')).transpose().loc[pl]

    def pairwise_phases(self, from_absolute=True, project_phases=False,
                        **kwargs):
        """
        Compute the pairwise relative grid phases of the cells in the module.

        The pairwise phases can be computed in two different ways: either from
        calling `BaseCell.phase` for each cell pair, or by computing the
        pairwise vector difference between the absolute phases from
        `Module.phases`.

        A full, antisymmetric matrix of relative phases is returned, so each
        cell pair is represented by two phases of opposite sign.

        Just like for the absolute phases, no wrapping is performed by this
        function. Use `Module.pairwise_phase_pattern` for window-aware, wrapped
        phases.

        ..note:: If `from_absolute` and `project_phases` are both False, there
        is no guarantee that the phases are well-defined with respect to
        a common window, and thus wrapping may not even make sense.

        Parameters
        ----------
        from_absolute : bool, optional
            If True, the pairwise relative phases are computed as the
            difference of the corresponding aboslute differences. If False,
            each pairwise relative phase is computed using `BaseCell.phase`.
        project_phases : bool, optional
            If True, phases are projected onto a regular hexagonal grid with
            edge lengths 1.0. The projection is based on the lattice defined by
            the module template cell.
        **kwargs : dict, optional
            Keyword arguments are passed to `BaseCell.phase`,
            `Module.template` and `TemplateGridCell.grid_peaks`.

        Returns
        -------
        DataFrame
            DataFrame containing the x- and y-components of the grid phase of
            each cell relative to each other cell. The DataFrame is indexed by
            pairs `(cell1, cell2)` for the relative phase of `cell2` with
            respect to `cell1`.

        """
        if from_absolute:
            kwargs.update(project_phases=project_phases)
            abs_phases = self.phases(**kwargs)
            pl = [(cell1, cell2) for cell1 in self for cell2 in self]
                  #if cell2 is not cell1]
            return pandas.DataFrame(
                {pair: abs_phases.loc[pair[1]] - abs_phases.loc[pair[0]]
                 for pair in pl}).transpose().loc[pl]

        pairwise_phases = self._pairwise_phases(**kwargs)
        if project_phases:
            pkw = dict(kwargs)
            pkw.update(polar_peaks=False, project_peaks=True)
            basis = self.template(**kwargs).grid_peaks(**pkw)[:2]
            new_basis = REGULAR_GRID_PEAKS[:2]
            coeffs = project_vectors(pairwise_phases, basis)
            pairwise_phases = pandas.DataFrame(
                coeffs.dot(new_basis), index=pairwise_phases.index,
                columns=pairwise_phases.columns)

        return pairwise_phases

    @memoize_method
    def phase_pattern(self, edge_correction='periodic', **kwargs):
        """
        Create a PointPattern instance of the phases of the cells in the module

        The phases as returned from `Module.phases` are wrapped into the window
        returned from `Module.window` if possible, before the PointPattern
        instance is created.

        Parameters
        ----------
        edge_correction : str {'stationary', 'finite', 'isotropic', 'periodic',
                               'plus'}, optional
            String to select the default edge handling to apply in computations
            involving the returned PointPattern instance. See the documentation
            for `PointPattern` for details.
        **kwargs : dict, optional
            Additional keyword arguments are passed to `Module.phases` and
            `Module.window`.

        Returns
        -------
        PointPattern
            PointPattern instance representning the phases.

        """
        window = self.window(**kwargs)
        # Avoid unneccesary spreading of keywords to memoized methods
        if 'window_type' in kwargs:
            del kwargs['window_type']

        phases = self.phases(**kwargs)
        phase_points = PointPattern.wrap_into(window, phases.values)
        return PointPattern(phase_points, window,
                            edge_correction=edge_correction)

    @memoize_method
    def pairwise_phase_pattern(self, full_window=False, sign='alternating',
                               edge_correction='finite', **kwargs):
        """
        Create a PointPattern instance of the pairwise phases of the cells in
        the module

        Before the PointPattern instance is created, the pairwise phases as
        returned from `Module.phases` are wrapped and reflected into half of
        a centered translation of the window returned from `Module.window`, cut
        along i diagonal. Using only half of the window ensures that the
        relative phase of each cell pair has a unique representation.

        Parameters
        ----------
        full_window : bool, optional
            If True, the pairwise phases take values in the same window as the
            absolute phases, leading to a sign ambiguity in the definition of
            the relative phase (see the parameter `sign`). If False, the window
            of the absolute phases is cut in half along a diagonal, such that
            each possible relative phase has an unambiguous representation.
        sign : {'alternating', 'random', 'both'}, optional
            If `'alternating'`, the sign ambiguity in the definition of the
            relative phase is resolved by making sure that each cell is placed
            first and last in the phase computations an equal number of times.
            If 'random', the sign is chosen for each cell pair. If `'both'`,
            both representations of each relative phase is included, giving two
            points per cell pair in the pattern.  If `full_window == False`,
            this parameter has no effect.
        edge_correction : str {'stationary', 'finite', 'isotropic', 'periodic',
                               'plus'}, optional
            String to select the default edge handling to apply in computations
            involving the returned PointPattern instance. See the documentation
            for `PointPattern` for details.
        **kwargs : dict, optional
            Additional keyword arguments are passed to `Module.pairwise_phases`
            and `Module.window`.

        Returns
        -------
        PointPattern
            PointPattern instance representning the pairwise phases.

        """
        # Avoid unneccesary spreading of keywords to memoized methods
        flag = False
        if 'from_absolute' in kwargs:
            from_absolute = kwargs.pop('from_absolute')
            flag = True

        window = self.window(**kwargs).centered()

        if 'window_type' in kwargs:
            del kwargs['window_type']

        if flag:
            kwargs.update(from_absolute=from_absolute)

        pairwise_phases = self.pairwise_phases(**kwargs)
        l = len(self)
        unique_indices = [(self[i], self[j])
                          for i in range(l - 1) for j in range(i + 1, l)]

        if not full_window:
            ppoints = PointPattern.wrap_into(
                window, pairwise_phases.loc[unique_indices].values)
            half_window = window.diagonal_cut()
            good_points = half_window.intersection(ppoints)
            bad_points = ppoints.difference(good_points)
            phase_points = ([(gp.x, gp.y) for gp in good_points] +
                            [(-bp.x, -bp.y) for bp in bad_points])
            window = half_window
        elif sign in ('random', 'alternating'):
            if sign == 'random':
                signs = numpy.random.randint(0, 2, size=len(unique_indices))
            else:
                signs = numpy.mod(numpy.arange(len(unique_indices)), 2)
            signs = 2 * signs - 1
            indices = [ui[::s] for (ui, s) in zip(unique_indices, signs)]
            phase_points = PointPattern.wrap_into(
                window, pairwise_phases.loc[indices].values)
        elif sign == 'both':
            indices = unique_indices + [ui[::-1] for ui in unique_indices]

            phase_points = PointPattern.wrap_into(
                window, pairwise_phases.loc[indices].values)
        else:
            raise ValueError("unknown sign: {}.".format(sign))

        return PointPattern(phase_points, window,
                            edge_correction=edge_correction)

    def phase_clusters(self, eps=0.075, min_samples=3, **kwargs):
        """
        Compute clusters of cells in phase space

        In order to do meaningful clustering that respects the periodic
        boundary conditions, the DBSCAN clustering algorithm is used.

        Parameters
        ----------
        eps, min_samples
            See `cluster.dbscan`. Note that clustering is performed on
            projected phases, so `eps` is dimensionless and should be less than
            1.0 to be meaningful.
        kwargs : dict, optional
            Additional keyword arguments are passed to `cluster.dbscan`.

        Returns
        -------
        labels : sequence
            Sequence of labels with indices corresponding to the indices in
            `self`: all cells with the same label belong to a cluster. The
            special label `-1` denotes outliers. This output can be fed
            directly to `CellCollection.from_labels` to instantiate new
            `CellCollection` instances based on the labels.

        """
        l = len(self)
        dmatrix = numpy.zeros((l, l))
        w = self.window(window_type='voronoi', project_phases=True).centered()
        ph = PointPattern.wrap_into(
            w, self.pairwise_phases(project_phases=True).values)
        for i in range(l):
            for j in range(l):
                p = ph[i * l + j]
                dmatrix[i, j] = numpy.sqrt(p.x * p.x + p.y * p.y)
        kwargs.update(eps=eps, min_samples=min_samples)
        return cluster.dbscan(dmatrix, metric='precomputed', **kwargs)[1]
