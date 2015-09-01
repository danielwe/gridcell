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
import numpy
import pandas
from scipy import signal, linalg, spatial
from sklearn import cluster
from matplotlib import pyplot
from collections import Mapping

from .utils import AlmostImmutable, gaussian, add_ticks, sensibly_divide
from .shapes import Ellipse
from .imaps import IntensityMap2D
from .pointpatterns import PointPattern
from .memoize.memoize import memoize_function, memoize_method


class Position(AlmostImmutable):
    """
    Represent positional data recorded over time

    """

    def __init__(self, t, x, y, speed_window=0.0, min_speed=0.0, **kwargs):
        """
        Initialize the Position instance

        :t: array-like, giving the times of position samples. Should be close
            to regularly spaced if speed_window != 0.0.
        :x: array-like, giving the x coordinates of position samples
        :y: array-like, giving the y coordinates of position samples
        :speed_window: length of the time interval over which to compute the
                       average speed at each sample. The length of the
                       averaging interval is rounded up to the nearest odd
                       number of samples to get a symmetric window. Must be
                       a non-negative number, given in the same unit as 't'.
                       Default is 0.0 (average speed computed over the shortest
                       possible interval -- a central difference around the
                       current sample).
        :min_speed: lower speed limit for valid data. Sampled positions are
                    only considered valid if the computed average speed at the
                    sample is larger than this. Must be a non-negative number,
                    given in the same unit as 'x' and 'y'. Default is 0.0 (all
                    data valid).
        :kwargs: none supported at the moment. In the future, support may be
                 added for a keyword 'info' containing information about the
                 transformation applied to get physical positions from raw
                 position data, if a use case can be found for having this
                 information available here.

        """
        if not min_speed >= 0.0:
            raise ValueError("'min_speed' must be a non-negative number")
        if not speed_window >= 0.0:
            raise ValueError("'speed_window' must be a non-negative number")
        self.speed_window = speed_window
        self.min_speed = min_speed

        self.speed, tweights, __ = self.speed_and_weights(t, x, y,
                                                          speed_window)

        speedmask = numpy.logical_or(self.speed < min_speed,
                                     numpy.ma.getmaskarray(self.speed))
        nanmask = numpy.logical_or(numpy.isnan(x), numpy.isnan(y))
        mask = numpy.logical_or(speedmask, nanmask)

        self.t = t
        self._x = x
        self._y = y
        self.x = numpy.ma.masked_where(mask, x)
        self.y = numpy.ma.masked_where(mask, y)
        self.tweights = numpy.ma.masked_where(mask, tweights)

    @staticmethod
    def speed_and_weights(t, x, y, speed_window):
        """
        Compute speed and time- and distance weights for position samples

        The computed speed is averaged over a certain time window around each
        sample. The time- and distance weights are arrays that assign a time
        interval and a distance interval to each sample.

        :t: array-like, giving the times of position samples. Should be close
            to regularly spaced if speed_window != 0.0.
        :x, y: array-like, giving the x and y coordinates of position samples.
               Nans or masked entries are treated as missing values, and
               affected entries in the returned speed and distance weight
               arrays will be nans.
        :speed_window: length of the time interval around each sample over
                       which to average the speed at each sample. The length of
                       the time interval is rounded up to the nearest odd
                       number of samples to get a symmetric window. Must be
                       a non-negative number, given in the same unit as 't'.
        :returns: array with speed at each position sample, and time- and
                  distance weight arrays. Missing values are masked.

        """
        tsteps = numpy.diff(t)
        tweights = 0.5 * numpy.hstack((tsteps[0], tsteps[:-1] + tsteps[1:],
                                       tsteps[-1]))

        x = numpy.ma.masked_where(numpy.isnan(x), x)
        y = numpy.ma.masked_where(numpy.isnan(y), y)

        xsteps = numpy.ma.diff(x)
        ysteps = numpy.ma.diff(y)
        dsteps = numpy.ma.sqrt(xsteps * xsteps + ysteps * ysteps)
        dweights = 0.5 * numpy.ma.hstack((dsteps[0], dsteps[:-1] + dsteps[1:],
                                          dsteps[-1]))

        dw_mask = numpy.ma.getmaskarray(dweights)
        dw_filled = numpy.ma.filled(dweights, fill_value=0.0)

        window_length = 2 * int(0.5 * speed_window / numpy.mean(tsteps)) + 1
        window_sequence = numpy.empty((window_length,))
        window_sequence.fill(1.0 / window_length)

        tweights_filt = sensibly_divide(
            signal.convolve(tweights, window_sequence, mode='same'),
            signal.convolve(numpy.ones_like(tweights), window_sequence,
                            mode='same'))
        dweights_filt = sensibly_divide(
            signal.convolve(dw_filled, window_sequence, mode='same'),
            signal.convolve((~dw_mask).astype(numpy.float_),
                            window_sequence, mode='same'), masked=True)

        speed = dweights_filt / tweights_filt

        return speed, tweights, dweights

    @memoize_method
    def timemap(self, bins, range_):
        """
        Compute a histogram showing the spatial distribution of time spent

        Only samples considered valid (average speed at sample higher than
        'self.min_speed') are included in the histogram. The histogram is
        returned as an IntensityMap2D instance.

        :bins: bin specification defining the bins to use in the histogram. The
               simplest formats are a scalar 'nbins' or a tuple '(nbins_x,
               nbins_y)', giving the number of bins of equal widths in each
               direction. For information about other valid formats, see the
               documentation for numpy.histogram2d().
        :range_: range specification giving the x and y values of the outermost
                 bin edges. The format is a tuple '((xmin, xmax), (ymin,
                 ymax))'. Samples outside this region will be discarded.
        :returns: IntensityMap2D instance containing the histogram

        """
        hist, xedges, yedges = numpy.histogram2d(
            numpy.ma.compressed(self.x),
            numpy.ma.compressed(self.y),
            bins=bins,
            range=range_,
            normed=False,
            weights=numpy.ma.compressed(self.tweights))

        return IntensityMap2D(hist, (xedges, yedges))

    def plot_path(self, axes=None, linewidth=0.5, color='0.5', alpha=0.5,
                  **kwargs):
        """
        Plot the path through the valid positions

        Parameters
        ----------
        axes : Axes, optional
            Axes instance to add the path to. If None (default), the current
            Axes instance is used if any, or a new one created.
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

        h = axes.plot(self.x, self.y, linewidth=linewidth, color=color,
                      alpha=alpha, **kwargs)
        return h

    def plot_samples(self, axes=None, marker='.', s=1.0, color=None,
                     alpha=0.5, **kwargs):
        """
        Make a scatter plot of the recorded positions without drawing a line

        The samples can be added to an existing plot via the optional 'axes'
        argument.

        :axes: Axes instance to add the path to. If None (default), the current
               Axes instance with equal aspect ratio is used if any, or a new
               one created.
        :marker: a valid matplotlib marker specification. Defaults to '.'
        :s: scalar or sequence giving the marker sizes. Defaults to 1.0
        :color: a valid matplotlib color specification, or a sequence of such,
                giving the color of the markers. If None (default), a grayscale
                color is computed based on the time weight associated with each
                sample, such that the sample that accounts for the longest time
                is black.
        :alpha: the opacity of the markers. Defaults to 0.5.
        :kwargs: additional keyword arguments passed on to axes.plot() for
                 specifying marker properties. Note especially the keyword
                 'label'.
        :returns: the plotted PathCollection instance

        """
        if axes is None:
            axes = pyplot.gca(aspect='equal')

        if color is None:
            color = 1.0 - (self.tweights / self.tweights.max())

        h = axes.scatter(self.x, self.y, marker=marker, s=s,
                         color=color, alpha=alpha, **kwargs)
        return h

    def plot_speed(self, axes=None, time_int=None, color='black',
                   min_speed=True, min_speed_kw=None, **kwargs):
        """
        Plot the speed versus time

        The path can be added to an existing plot via the optional 'axes'
        argument.

        :axes: Axes instance to add the speed curve to. If None (default), the
               current Axes instance is used if any, or a new one created.
        :time_int: sequence giving the start and end of the time interval to
                   plot the speed over. If None (default), the whole duration
                   of the measurement is used.
        :min_speed: if True, a horizontal line at self.min_speed (used to
                    filter valid positions) is added to the plot. By default,
                    a solid, red line with linewidth == 0.5 will be plotted,
                    but this can be overridden using the parameter
                    min_speed_kw. If false, no lines are plotted. Default is
                    true.
        :min_speed_kw: dict of keyword arguments to pass to the axes.plot()
                        method used to plot the min_speed line. default: None
                       (empty dict)
        :color: a valid matplotlib color specification giving the color to plot
                the speed curve with. Defaults to 'black'.
        :kwargs: additional keyword arguments passed on to axes.plot() for
                 specifying line properties. Note especially the keywords
                 'linestyle', 'linewidth' and 'label'.
        :returns: list containing the plotted Line2D instances

        """
        if axes is None:
            axes = pyplot.gca()

        if min_speed_kw is None:
            min_speed_kw = {}

        if time_int is not None:
            start_index = numpy.argmin(numpy.abs(self.t - time_int[0]))
            end_index = numpy.argmin(numpy.abs(self.t - time_int[1]))
            t = self.t[start_index:end_index]
            speed = self.speed[start_index:end_index]
        else:
            t, speed = self.t, self.speed
        lines = axes.plot(t, speed, color=color, **kwargs)

        if min_speed:
            min_line = axes.axhline(self.min_speed, linewidth=0.5, color='r',
                                    **min_speed_kw)
            lines.append(min_line)

        return lines


class BaseCell(AlmostImmutable):
    """
    Represent a cell for which only the spatial firing rate is known (no
    temporal information about position or spikes). Can be used for modeling
    prototyipcal/idealized grid cells, and also works as a base class for other
    Cell classes.

    """

    def __init__(self, firing_rate, threshold):
        """
        Initialize the BaseCell instance

        :firing_rate: an IntensityMap2D instance giving the spatial firing rate
                      map for the cell. This map is assumed to not require
                      further smoothing or other processing.
        :threshold: peak threshold. Used to separate peaks in the
                    autocorrelogram of the firing rate of the cell.

        """
        self.firing_rate = firing_rate
        self.bins = firing_rate.shape
        self.range_ = firing_rate.range_
        self.threshold = threshold

    def autocorrelate(self, mode='full', pearson=True, normalized=False):
        """
        Compute the autocorrelogram of the smoothed firing rate

        This is a wrapper around self.firing_rate.autocorrelate().

        :mode: string indicating the size of the output. See
               IntensityMap2D.autocorrelate() for details. Valid options:
               'full', 'valid', 'same'. Default is 'full'.
        :pearson: if True, the IntensityMap instances are normalized to mean
                  0.0 and variance 1.0 before correlating. The result of the
                  computation will then be the Pearson product-moment
                  correlation coefficient between displaced intensity arrays,
                  evaluated at each possible displacement. Default is True.
        :normalized: if True, any masked values or nans in the intensity
                     arrays, as well as values beyond the their edges, are
                     treated as missing values, and the correlogram is
                     renormalized for each cell to eliminate their influence.
                     Default is True.
        :returns: IntensityMap2D instance representing the autocorrelogram of
                  the firing rate

        """
        return self.firing_rate.autocorrelate(mode=mode, pearson=pearson,
                                              normalized=normalized)

    @staticmethod
    @memoize_function
    def detect_central_peaks(imap, threshold):
        """
        Identify the most central peaks in an IntensityMap

        This is essentially a wrapper around imap.peaks(threshold), but
        performs selection and sorting of the peaks. The selected peaks are the
        peak closest to the center, and the six peaks closest to it. The center
        peak is returned first, and the other six are sorted by the angle from
        the x axis to the line from the center peak to each of them.

        A label array identifying the regions surrounding each peak is also
        returned.

        :imap: IntensityMap instance to detect peaks in
        :threshold: lower intensity bound defining the regions in which peaks
                    are found.
        :returns:
            - numpy array 'peaks' of shape (7, 2), where (peaks[0][0],
              peaks[0][1]) are the x and y coordinates of the central peak,
              etc.
            - array of labels identifying the region surrounding each peak,
              such that labels == i is an index to the region surrounding the
              ith peak (found at index i - 1 in the returned array of peaks)

        """
        all_peaks, labels, __ = imap.peaks(threshold)

        # Find the peak closest to the center
        apx, apy = all_peaks[:, 0], all_peaks[:, 1]
        apr = apx * apx + apy * apy
        cindex = numpy.argmin(apr)

        # Find the seven peaks closest to the center (incl. the center peak)
        cpeak = all_peaks[cindex]
        cpx = apx - cpeak[0]
        cpy = apy - cpeak[1]
        cpr = cpx * cpx + cpy * cpy
        rsort = numpy.argsort(cpr)

        # Sort the peaks surrounding the center peak by angle
        cpx = cpx[rsort[1:7]]
        cpy = cpy[rsort[1:7]]
        angles = numpy.mod(numpy.arctan2(cpy, cpx), 2 * numpy.pi)
        asort = numpy.argsort(angles)
        sort = numpy.hstack((rsort[0], rsort[1:7][asort]))

        # Grab the sorted peaks, discard the radius information for now
        peaks = all_peaks[sort, :2]

        # Find corresponding labels
        new_labels = numpy.zeros_like(labels)
        for (i, s) in enumerate(sort):
            i1, s1 = i + 1, s + 1
            new_labels[labels == s1] = i1

        return peaks, new_labels

    def peaks(self, project=True, threshold=None):
        """
        Identify the coordinates of the six inner peaks in the autocorrelogram
        of the firing rate, sorted by angle with the positive x axis.

        :project: if True, the peaks from the autocorrelogram are projected
                  into the space of valid lattice vectors, such that each peak
                  is equal to the sum of its two neighbors. If False, the peaks
                  are returned without projection. Default is True.
        :threshold: lower intensity bound defining the regions in which peaks
                    are found. If None (default), the attribute self.threshold
                    is used.
        :returns: numpy array 'peaks' of shape (6, 2), where (peaks[0][0],
                  peaks[0][1]) are the x and y coordinates of the first peak,
                  etc.
        """
        if threshold is None:
            threshold = self.threshold

        acorr = self.autocorrelate()
        peaks, __ = self.detect_central_peaks(acorr, threshold)

        # Discard center peak
        peaks = peaks[1:]

        if project:
            # Project to true lattice vectors
            pmat = (1 / 6) * linalg.toeplitz([2, 1, -1, -2, -1, 1])
            peaks = pmat.dot(peaks)

        return peaks

    def peaks_polar(self, project=True, threshold=None):
        """
        Identify the polar coordinates of the six inner peaks in the
        autocorrelogram of the firing rate, sorted by angle with the positive
        x axis.

        :project: if True, the peaks from the autocorrelogram are projected
                  into the space of valid lattice vectors, such that each peak
                  is equal to the sum of its two neighbors. If False, the peaks
                  are returned without projection. Default is True.
        :threshold: lower intensity bound defining the regions in which peaks
                    are found. If None (default), the attribute self.threshold
                    is used.
        :returns: numpy array 'peaks' of shape (6, 2), where (peaks[0][0],
                  peaks[0][1]) is the r and theta coordinates of the first
                  peak, etc.

        """
        peaks = self.peaks(project=project, threshold=threshold)
        px, py = peaks[:, 0], peaks[:, 1]
        radii = numpy.sqrt(px * px + py * py)
        angles = numpy.mod(numpy.arctan2(py, px), 2 * numpy.pi)
        peaks_polar = numpy.column_stack((radii, angles))
        return peaks_polar

    @memoize_method
    def ellipse(self, project=True, threshold=None):
        """
        Fit an ellipse to the inner ring of six peaks around the center of the
        autocorrelogram

        :project: if True, the ellipse is fitted to the lattice-projected peaks
                  (see BaseCell.peaks()). If False, the it is fitted to the
                  unprojected peaks. Default is True.
        :threshold: lower intensity bound defining the regions in which peaks
                    are found. If None (default), the attribute self.threshold
                    is used.
        :returns: an Ellipse instance representing the fitted ellipse.

        """
        peaks = self.peaks(project=project, threshold=threshold)
        xscale, yscale = 0.5 * peaks[:, 0].ptp(), 0.5 * peaks[:, 1].ptp()
        f0 = numpy.sqrt(xscale * yscale)
        ellipse = Ellipse(fitpoints=peaks, f0=f0)
        return ellipse

    @memoize_method
    def scale(self, mode='geometric', project=True, threshold=None):
        """
        Calculate the grid scale of the cell

        :mode: flag to select which grid scale definition to use. Valid
               options:
            'geometric': the scale is defined as the geometric mean of the
                         semi-minor and semi-major axes of the ellipse fitted
                         to the inner ring of six peaks.
            'arithmetic': the scale is defined as the arithmetic mean of the
                          distances from the center to the each of the six
                          inner peaks.
               Default is 'geometric'.
        :project: if True, the scale is based on the lattice-projected peaks
                  (see BaseCell.peaks()). If False, it is based on the
                  unprojected peaks. Default is True.
        :threshold: lower intensity bound defining the regions in which peaks
                    are found. If None (default), the attribute self.threshold
                    is used.
        :returns: the grid scale

        """
        if mode == 'geometric':
            ellipse = self.ellipse(project=project, threshold=threshold)
            scale = numpy.sqrt(ellipse.a * ellipse.b)
        elif mode == 'arithmetic':
            radii = self.peaks_polar(project=project,
                                     threshold=threshold)[:, 0]
            scale = numpy.mean(radii)
        else:
            raise ValueError("unknown mode {}".format(mode))

        return scale

    @memoize_method
    def firing_field(self, threshold=None):
        """
        Compute a covariance matrix characterizing the average shape of the
        firing fields

        The estimate is based on the region surrounding the central peak in the
        autocorrelogram, such that the gaussian(x, cov=self.firing_field())
        approximates the shape of the firing fields.

        :threshold: lower intensity bound defining the regions in which peaks
                    are found. If None (default), the attribute self.threshold
                    is used.
        :returns: estimated firing field covariance matrix

        """
        if threshold is None:
            threshold = self.threshold

        acorr = self.autocorrelate()
        __, labels = self.detect_central_peaks(acorr, threshold)

        central_mask = ~(labels == 1)
        __, __, firing_field = acorr.fit_gaussian(mask=central_mask)

        return firing_field

    @memoize_method
    def firing_field_map(self, threshold=None):
        """
        Compute an IntensityMap2D instance showing the fitted firing field

        The intensity map is computed over the same region as the
        autocorrelogram returned from self.autocorrelate() and normalized to
        a maximal value of 1.0, and is thus useful for comparing the fitted
        field with the region it was fitted to.

        :threshold: lower intensity bound defining the regions in which peaks
                    are found. If None (default), the attribute self.threshold
                    is used.
        :returns: IntensityMap2D instance of the fitted firing field

        """
        acorr = self.autocorrelate()
        firing_field = self.firing_field(threshold=threshold)

        xcm, ycm = acorr.bset.cmesh
        xypoints = numpy.vstack((xcm.ravel(), ycm.ravel())).transpose()
        ffarr = gaussian(xypoints, cov=firing_field).reshape(acorr.shape)
        ffarr *= 1.0 / numpy.abs(ffarr).max()

        return IntensityMap2D(ffarr, acorr.bset)

    def correlate(self, other, mode='full', pearson=True, normalized=False):
        """
        Compute the cross-correlaogram of another cell's firing rate to this

        This is a wrapper around self.firing_rate.correlate().

        :other: another Cell instance.
        :mode: string indicating the size of the output. See
               IntensityMap2D.correlate() for details. Valid options: 'full',
               'valid', 'same'. Default is 'full'.
        :pearson: if True, the IntensityMap instances are normalized to mean
                  0.0 and variance 1.0 before correlating. The result of the
                  computation will then be the Pearson product-moment
                  correlation coefficient between displaced intensity arrays,
                  evaluated at each possible displacement. Default is True.
        :normalized: if True, any masked values or nans in the intensity
                     arrays, as well as values beyond the their edges, are
                     treated as missing values, and the correlogram is
                     renormalized for each cell to eliminate their influence.
                     Default is True.
        :returns: IntensityMap2D instance representing the cross-correlogram of
                  the firing rates

        """
        return self.firing_rate.correlate(other.firing_rate, mode=mode,
                                          pearson=pearson,
                                          normalized=normalized)

    @memoize_method
    def features(self, roll=0, lweight=1.0, extra=False):
        """
        Compute a series of features of this cell

        The purpose of the feature array is to embed the cell into
        a high-dimensional space with dimensionless axes, where the Euclidean
        distance between cells correspond to some concept of closeness. This is
        useful for clustering cells into modules.

        The present definition consists of the x and y coordinates of three
        unique inner peaks in the autocorrelogram, divided by the grid scale
        (parametrizing the orientation and shape of the grid), as well as the
        logarithm of the grid scale (parametrizing the size of the grid). The
        balance between the two is controlled by the parameter `sweight`.

        Parameters
        ----------
        roll : integer, optional
             Quantities related to individual peaks are listed in the order
             given by `numpy.roll(self.peaks(), roll, axis=0)`.
        lweight : scalar, optional
            The scaling factor deciding the relative weight between
            size-related and shape-related features. The size-related features
            are multiplied by this number.
        extra : bool, optional
            If True, the feature series is extended to also include the scale
            (no logarithm taken), polar coordinates for the peaks, as well as
            both cartesian and polar coordinates for the ellipse parameters.

        Returns
        -------
        Series
            Series containing the cell features. The keys are appropriately
            Latex-formatted strings..

        """
        scale = self.scale()
        logscale = numpy.log(scale)
        peaks = numpy.roll(self.peaks(), roll, axis=0)

        # Make sure that the differences between all features are
        # dimensionless, such that the distance is independent of units. Note
        # that (log(l_2) - log(l_1)) ** 2 = log(l_2 / l_1) ** 2, which is
        # dimensionless and thus OK, even though the dimension of log(l) is
        # not well-defined in a strict sense.
        feats = numpy.hstack((lweight * logscale, peaks[:3].ravel() / scale))
        featlabels = ('log_l', 'ax1', 'ay1', 'ax2', 'ay2', 'ax3', 'ay3')
        if extra:
            polar = numpy.roll(self.peaks_polar(), roll, axis=0)
            polar[:, 0] /= scale
            ellipse = self.ellipse()
            ecc, tilt_2 = ellipse.ecc, 2.0 * ellipse.tilt
            x_ell = ecc * numpy.cos(tilt_2)
            y_ell = ecc * numpy.sin(tilt_2)
            feats = numpy.hstack((feats, scale, polar[:3].ravel(), x_ell,
                                  y_ell, ecc, tilt_2))
            featlabels += ('l', 'al1', 'beta1', 'al2', 'beta2', 'al3',
                           'beta3', 'xell', 'yell', 'epsilon', '2theta')
        index = [features_index[label] for label in featlabels]
        return pandas.Series(feats, index=index)

    @memoize_method
    def roll(self, other):
        """
        Determine the peak roll necessary to align the peaks in this cell most
        cloesly with the ones in `other`

        Parameters
        ----------
        other : BaseCell
            Cell to align this one with.

        Returns
        -------
        roll : integer
            The peak roll to apply to `self` to align it with `other`.
            The peak roll is defined such that `numpy.roll(self.peaks(), roll,
            axis=0)` and `other.peaks()` give coordinates to the most closely
            corresponding peaks in `self` and `other`.

        """
        pi = numpy.pi
        pi_2 = 2.0 * pi
        sangles = self.peaks_polar()[:, 1]
        oangles = other.peaks_polar()[:, 1]

        roll = 0
        diff = numpy.mod(pi + sangles - oangles, pi_2) - pi
        delta = numpy.abs(numpy.sum(diff))
        for r in (-1, 1):
            diff = numpy.mod(pi + numpy.roll(sangles, r) - oangles, pi_2) - pi
            d = numpy.abs(numpy.sum(diff))
            if d < delta:
                delta = d
                roll = r
        return roll

    def distance(self, other, **kwargs):
        """
        Compute a distance between the grid patterns of grid cells

        This method defines a metric on the space of grid patterns from grid
        cells. The distance is defined as the Euclidean distance between the
        feature arrays of the cells, using the relative peak roll given by
        `BaseCell.roll`.

        Parameters
        ----------
        other : BaseCell
            BaseCell instance to measure distance to.
        kwargs : dict, optional
            Keyword arguments to pass to `BaseCell.features`.

        Returns
        -------
        scalar
            Distance between `self` and `other`.

        """
        if self is other:
            dist = 0.0
        else:
            roll = self.roll(other)

            ofeat = other.features(roll=0, **kwargs)
            dfeat = self.features(roll=roll, **kwargs) - ofeat
            dist = numpy.sqrt(numpy.sum(dfeat * dfeat))

        return dist

    @memoize_method
    def phase(self, other):
        """
        Find the grid phase of this cell relative to another cell

        :other: another Cell instance.
        :returns: tuple containing the x and y components of the relative grid
                  phase

        """
        corr = self.correlate(other)
        peaks, __ = self.detect_central_peaks(corr, self.threshold)

        return peaks[0]

    def plot_firing_rate(self, axes=None, cax=None, cmap=None, cbar_kw=None,
                         **kwargs):
        """
        Plot the spatial firing rate map of this cell

        This method is basically a wrapper around `self.firing_rate.plot`.

        Parameters
        ----------
        axes : Axes, optional
            Axes instance to add the firing rate to. If None (default), the
            current Axes instance is used if any, or a new one created.
        cax : Axes, optional
            Axes instance to add the colorbar to. If None (default), matplotlib
            automatically makes space for a colorbar on the right-hand side of
            the plot.
        cmap : Colormap or registered colormap name, optional
            Colormap to use for the plot. If None (default), the default
            colormap from rcParams is used.
            ..note:: The default map might be 'jet', and this is something you
            certainly DON'T WANT to use! If you're clueless, try 'YlGnBu_r' or
            'gray'.
        cbar_kw : dict, optional
            Keyword arguments to pass to `pyplot.colorbar`.
        **kwargs : dict, optional
            Additional keyword arguments pass to `IntensityMap2D.plot`.

        Returns
        -------
        Axes
            The axes instance containing the plot
        Colorbar
            The created `Colorbar` instance

        """
        axes, cbar = self.firing_rate.plot(axes=axes, cax=cax, vmin=0.0,
                                           cmap=cmap, cbar_kw=cbar_kw,
                                           **kwargs)
        return axes, cbar

    def plot_autocorrelogram(self, axes=None, cax=None, threshold=False,
                             peaks=False, ellipse=False, cmap=None,
                             cbar_kw=None, **kwargs):
        """
        Plot the autocorrelogram of the firing rate of this cell

        The correlogram can be added to an existing plot via the optional
        'axes' argument.

        This method is little but a wrapper around self.autocorrelate().plot().

        :axes: Axes instance to add the intensity map to. If None (default),
               a new Figure is created (this method never plots to the current
               Figure or Axes). In the latter case, equal aspect ration will be
               enforced on the newly created Axes instance.
        :cax: Axes instance to plot the colorbar into. If None (default),
              matplotlib automatically makes space for a colorbar on the
              right-hand side of the plot.
        :threshold: if True, mask values smaller than self.threshold from the
                    plot. Default is False.
        :peaks: if True, add the most central peak and the six peaks closest to
                it to the plot, using some (hopefully) sensible plotting
                defaults. If more control is required, leave this false and
                call self.plot_peaks() on the returned axes instance.
        :ellipse: if True, add an ellipse fitted through the six peaks closest
                  to the center peak to the plot, using some (hopefully)
                  sensible plotting defaults. If more control is required,
                  leave this false and call self.plot_ellipse() on the returned
                  axes instance.
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
        acorr = self.autocorrelate()

        if threshold:
            thres_val = self.threshold
        else:
            thres_val = None

        axes, cbar = acorr.plot(axes=axes, cax=cax, threshold=thres_val,
                                vmin=-1.0, vmax=1.0, cmap=cmap,
                                cbar_kw=cbar_kw, **kwargs)
        if peaks:
            self.plot_peaks(axes=axes)
        if ellipse:
            self.plot_ellipse(axes=axes)

        return axes, cbar

    def plot_peaks(self, axes=None, project=True, threshold=None, marker='o',
                   color='black', gridlines=True, gridlines_kw=None, **kwargs):
        """
        Plot the locations of the inner ring of peaks in the firing rate
        autocorrelogram

        The peaks can be added to an existing plot via the optional 'axes'
        argument. They are typically added to a plot of the autocorrelogram.

        :axes: Axes instance to add the path to. If None (default), the current
               Axes instance with equal aspect ratio is used if any, or a new
               one created.
        :project: if True, the lattice-projected peaks are plotted (see
                  BaseCell.peaks()). If False, the non-projected peaks are
                  plotted.
        :threshold: lower intensity bound defining the regions in which peaks
                    are found. If None (default), the attribute self.threshold
                    is used.
        :marker: a valid matplotlib marker specification. Defaults to 'o'
        :color: a valid matplotlib color specification, or a sequence of such,
                giving the color of the markers. Defaults to 'black'.
        :gridlines: if True, lines through the peaks showing the grid axes are
                    added to the plot. By default, dashed lines with color ==
                    '0.5' and opacity alpha == 0.5 are used, but this can be
                    overridden using the parameter gridlines_kw. If false, no
                    lines are plotted. Default is true.
        :gridlines_kw: dict of keyword arguments to pass to the axes.plot()
                       method used to plot gridlines. Default: None (empty
                       dict)
        :kwargs: additional keyword arguments passed on to the axes.plot()
                 method used to plot peaks. Note in particular the keyword
                 'markersize'.
        :returns: a list of the plotted Line2D instances

        """
        if axes is None:
            axes = pyplot.gca(aspect='equal')

        peaks = self.peaks(project=project, threshold=threshold)
        h = axes.plot(peaks[:, 0], peaks[:, 1], linestyle='None',
                      marker=marker, color=color, **kwargs)

        if gridlines:
            if gridlines_kw is None:
                gridlines_kw = {}
            acorr = self.autocorrelate()
            xedges = acorr.bset.xedges
            peaks_polar = self.peaks_polar(project=project)
            angles = peaks_polar[:3, 1]
            line_y = numpy.outer(xedges, numpy.tan(angles))
            h += axes.plot(xedges, line_y, linestyle='dashed', color='0.5',
                           alpha=0.5, **gridlines_kw)

        return h

    def plot_ellipse(self, axes=None, project=True, threshold=None,
                     linestyle='solid', linewidth=2.0, color='red',
                     smajaxis=True, **kwargs):
        """
        Plot an ellipse fitted through the inner ring of peaks in the firing
        rate autocorrelogram

        The ellipse can be added to an existing plot via the optional 'axes'
        argument. It is typically added to a plot of the autocorrelogram.

        This method is a simple wrapper around self.ellipse().plot()

        :axes: Axes instance to add the ellipse to. Passed through to the
               self.ellipse().plot() method.
        :project: if True, an ellipse fitted to the lattice-projected peaks is
                  plotted (see BaseCell.peaks()). If False, an ellipse fitted
                  to the unprojected peaks is plotted. Default is True.
        :threshold: lower intensity bound defining the regions in which peaks
                    are found. If None (default), the attribute self.threshold
                    is used.
        :linestyle,linewidth,color,smajaxis,kwargs:
            see Ellipse.plot() (note that defaults are different here).
        :returns: list containing the plotted objects: one pathces.Ellipse
                  instance, and a Line2D instance per plotted axis.

        """
        ell = self.ellipse(project=project, threshold=threshold)
        h = ell.plot(axes=axes, linestyle=linestyle, linewidth=linewidth,
                     color=color, smajaxis=smajaxis, **kwargs)
        return h

    def plot_correlogram(self, other, axes=None, cax=None, threshold=False,
                         cpeak=False, ellipse=False, cmap=None, cbar_kw=None,
                         **kwargs):
        """
        Plot the cross-correlogram of the firing rate of another cell and this
        cell

        The correlogram can be added to an existing plot via the optional
        'axes' argument.

        :other:another BaseCell instance
        :axes: Axes instance to add the intensity map to. If None (default),
               a new Figure is created (this method never plots to the current
               Figure or Axes). In the latter case, equal aspect ration will be
               enforced on the newly created Axes instance.
        :cax: Axes instance to plot the colorbar into. If None (default),
              matplotlib automatically makes space for a colorbar on the
              right-hand side of the plot.
        :threshold: if True, mask values smaller than self.threshold from the
                    plot. Default is False.
        :cpeak: if True, add the most central peak to the plot, using some
                (hopefully) sensible plotting defaults. If more control is
                required, call self.detect_central_peaks() on
                self.correlogram(other) to get peaks, and add them to the plot
                manually.
        :ellipse: if True, add an ellipse fitted through the six peaks closest
                  to the center peak to the plot, using some (hopefully)
                  sensible plotting defaults. If more control is required, call
                  self.detect_central_peaks() on self.correlogram(other) to get
                  the peaks, instantiate an Ellipse instance fitted to them,
                  and add use Ellipse.plot() to add it to the plot.
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
        corr = self.correlate(other)

        if threshold:
            thres_val = self.threshold
        else:
            thres_val = None

        axes, cbar = corr.plot(axes=axes, cax=cax, threshold=thres_val,
                               vmin=-1.0, vmax=1.0, cmap=cmap, cbar_kw=cbar_kw,
                               **kwargs)
        if cpeak or ellipse:
            pks, __ = self.detect_central_peaks(corr, self.threshold)
            if cpeak:
                axes.plot(pks[0, 0], pks[0, 1], linestyle='None', marker='o',
                          color='black')
                # Add a cross for comparison
                axes.axvline(0.0, linestyle='dashed', color='0.5', alpha=0.5)
                axes.axhline(0.0, linestyle='dashed', color='0.5', alpha=0.5)
            if ellipse:
                ell = Ellipse(fitpoints=pks)
                ell.plot(axes=axes, majaxis=True, linewidth=2.0, color='red')

        return axes, cbar

    def plot_firing_field(self, axes=None, cax=None, threshold=False,
                          cmap=None, cbar_kw=None, **kwargs):
        """
        Plot the shape of the firing field

        A Gaussian function with peak in the center and covariance matrix
        from self.firing_field() is plotted on axes identical to those used in
        self.plot_autocorrelogram().

        :axes: Axes instance to add the intensity map to. If None (default),
               a new Figure is created (this method never plots to the current
               Figure or Axes). In the latter case, equal aspect ration will be
               enforced on the newly created Axes instance.
        :cax: Axes instance to plot the colorbar into. If None (default),
              matplotlib automatically makes space for a colorbar on the
              right-hand side of the plot.
        :threshold: if True, mask values smaller than self.threshold from the
                    plot. Default is False.
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
        ffield = self.firing_field_map()

        if threshold:
            thres_val = self.threshold
        else:
            thres_val = None

        axes, cbar = ffield.plot(axes=axes, cax=cax, threshold=thres_val,
                                 cmap=cmap, cbar_kw=cbar_kw, **kwargs)

        return axes, cbar


class TemplateGridCell(BaseCell):
    """
    Represent an idealized grid cell defined only using the lattice vectors and
    the firing field covariance matrix. Can for example be used to represent
    the average cell from a module.

    """

    def __init__(self, peaks, firing_field, bset, threshold):
        """
        Initialize the TemplateGridCell instance

        :peaks: array-like of shape (6, 2), containing the six primary lattice
                vectors of the grid (equivalent to the coordinates of the six
                inner peaks in the autocorrelogram of the cell).
        :firing_field: array-like of shape (2, 2), giving the covariance
                       matrix characterizing the firing field shape. The
                       gaussian(x, cov=firing_field) gives the shape of the
                       firing fields of the cell.
        :bset: a BinnedSet2D instance defining the region over
               which this cells firing rate should be defined.
        :threshold: peak threshold. Used to separate peaks in the
                    autocorrelogram of the firing rate of the cell.

        """
        self._peaks = peaks
        self._firing_field = firing_field
        firing_rate = self.construct_firing_rate(peaks, firing_field, bset)
        BaseCell.__init__(self, firing_rate, threshold=threshold)

    @staticmethod
    def construct_firing_rate(peaks, firing_field, bset):
        """
        Create an IntensityMap2D instance of an idealized firing rate based on
        grid lattice vectors and the firing field covariance matrix

        :peaks: array-like of shape (6, 2), containing the six primary lattice
                vectors of the grid (equivalent to the coordinates of the six
                inner peaks in the autocorrelogram of the cell).
        :firing_field: array-like of shape (2, 2), giving the covariance
                       matrix characterizing the firing field shape. The
                       gaussian(x, cov=firing_field) gives the shape of the
                       firing fields of the cell.
        :bset: a BinnedSet2D instance defining the region over
               which this cells firing rate should be defined.
        :returns: IntensityMap2D instance of the firing rate

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

        layers = int(numpy.ceil(numpy.sqrt(diagsq / mldsq.min()))) + 1

        pattern = numpy.vstack(((0.0, 0.0), peaks))
        all_peaks = pattern.copy()
        for i in range(layers):
            for l in range(i):
                k = i - l
                disp = k * lattice + l * lattice_r1
                if (k - l) % 3 == 0:
                    for d in disp:
                        all_peaks = numpy.vstack((all_peaks, pattern + d))
                else:
                    for d in disp:
                        all_peaks = numpy.vstack((all_peaks, d))

        # Compute the firing rate
        firing_rate = 0.0
        for peak in all_peaks:
            firing_rate += gaussian(xypoints, mean=peak, cov=firing_field)

        # Normalize to a peak rate of 1 Hz, mostly for aesthetics
        firing_rate *= 1.0 / numpy.amax(firing_rate)

        # Reshape to the shape of xcm and ycm
        firing_rate = firing_rate.reshape(xcm.shape)

        return IntensityMap2D(firing_rate, bset)

    def peaks(self, project=True, **kwargs):
        """
        Return the coordinates of the six inner peaks

        :project: if True, the peaks from the autocorrelogram are projected
                  into the space of valid lattice vectors, such that each peak
                  is equal to the sum of its two neighbors. If False, the peaks
                  are returned without projection. Default is True.
        :kwargs: None supported at the moment. Included to absorb 'threshold',
                 which is typically passed by calling functions.
        :returns: numpy array 'peaks' of shape (6, 2), where (peaks[0][0],
                  peaks[0][1]) is the x and y coordinates of the first peak,
                  etc.
        """
        peaks = self._peaks

        if project:
            # Project to true lattice vectors
            pmat = (1 / 6) * linalg.toeplitz([2, 1, -1, -2, -1, 1])
            peaks = pmat.dot(peaks)

        return peaks

    def firing_field(self, **kwargs):
        """
        Return the firing field covariance matrix

        :kwargs: None supported at the moment. Included to absorb 'threshold',
                 which is typically passed by calling functions.
        :returns: firing field covariance matrix

        """
        return self._firing_field


class Cell(BaseCell):
    """
    Represent a cell based on recorded spike train and position, and provide
    methods for computing and plotting firing rate maps, correlations etc.

    """

    def __init__(self, spike_t, pos, bins, range_, filter_size,
                 threshold, **kwargs):
        """
        Initialize the Cell instance

        :spike_t: array-like, giving the times at which spikes were detected
                  from this cell.
        :pos: Position instance representing the spatial movement over time
              of the animal.
        :bins: bin specification defining the bins to use in the firing rate
               map. The simplest formats are a scalar 'nbins' or a tuple
               '(nbins_x, nbins_y)', giving the number of bins of equal widths
               in each direction. For information about other valid formats,
               see the documentation for numpy.histogram2d().
        :range_: range specification giving the x and y values of the outermost
                 bin edges. The format is a tuple '((xmin, xmax), (ymin,
                 ymax))'. Samples outside this region will be discarded.
        :threshold: peak threshold. Used to separate peaks in the
                    autocorrelogram of the firing rate of the cell.
        :filter_size: characteristic smoothing width to use when computing
                      firing rates. Carries the same units as the coordinates
                      in pos. If None, no smoothing is applied.
        :kwargs: none supported at the moment. In the future, support may be
                 added for a keyword 'info' containing information about the
                 transformation applied to get physical positions from raw
                 position data, if a use case can be found for having this
                 information available here.

        """
        self.spike_t = spike_t
        self.pos = pos
        self.filter_size = filter_size

        spike_x, spike_y = self.interpolate_spikes(spike_t, pos.t, pos.x,
                                                   pos.y)
        self.spike_x, self.spike_y = spike_x, spike_y

        timemap = pos.timemap(bins=bins, range_=range_)
        firing_rate = self.compute_firing_rate(spike_x, spike_y, timemap,
                                               filter_size)
        BaseCell.__init__(self, firing_rate, threshold=threshold)

    @staticmethod
    def interpolate_spikes(spike_t, t, x, y):
        """
        Find the locations of spikes in a spike train

        The arrays giving positions in which to interpolate may be masked or
        contain nans, in which case spikes occuring in the corresponding times
        will be discarded.

        :spike_t: array-like, giving the times at which spikes were detected
        :t: array-like, giving the times of position samples
        :x: array-like, possibly masked, giving the x coordinates of position
            samples
        :y: array-like, possibly masked, giving the y coordinates of position
            samples
        :returns: array of spike x coordinates, array of spike y coordinates.
                  The arrays are masked if the input x and y arrays are masked.

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

    @staticmethod
    def compute_firing_rate(spike_x, spike_y, timemap, filter_size,
                            smoothing_mode='pre'):
        """
        Compute the firing rate map from spikes and positional data

        :spike_x: array-like, possibly masked, x coordinates of spikes
        :spike_y: array_like, possibly masked, y coordinates of spikes
        :timemap: IntensityMap2D instance giving the spatial distribution of
                  time spent
        :filter_size: if not None, the firing rate map is smoothed using the
                      IntensityMap2D.smoothed() method, with this parameter as
                      filter size.
        :smoothing_mode: flag to select between two available smoothing
                         methods. Possible values:
            'pre': the histogram of spikes and the timemap are smoothed
                   individually before they are divided by each other to create
                   the firing rate map.
            'post': the histogram of spikes is divided by the timemap to create
                    an unsmoothed firing rate map, which is then smoothed.
        :returns: IntensityMap2D instance giving the firing rate map, and if
                  filter_size is not None, another InteisytMap2D giving the
                  smoothed firing rate map.

        """
        bset = timemap.bset
        spike_hist, __, __ = numpy.histogram2d(numpy.ma.compressed(spike_x),
                                               numpy.ma.compressed(spike_y),
                                               bins=(bset.xedges, bset.yedges),
                                               normed=False)
        spikemap = IntensityMap2D(spike_hist, bset)

        if filter_size is not None:
            if smoothing_mode == 'pre':
                firing_rate = (spikemap.smoothed(filter_size) /
                               timemap.smoothed(filter_size))
            elif smoothing_mode == 'post':
                firing_rate = spikemap / timemap
                firing_rate = firing_rate.smoothed(filter_size)
            else:
                raise ValueError("unknown smoothing mode {}"
                                 .format(smoothing_mode))
        else:
            firing_rate = spikemap / timemap

        return firing_rate

    def plot_spikes(self, axes=None, path=False, marker='o', alpha=0.25,
                    zorder=10, **kwargs):
        """
        Plot the spatial location of the recorded spikes

        The spikes can be added to an existing plot via the optional 'axes'
        argument.

        :axes: Axes instance to add the spikes to. If None (default), the
               current Axes instance with equal aspect ratio is used if any, or
               a new one created.
        :path: if True, plot the path through the spikes using some hopefully
               sensible defaults. For better control, leave this False and use
               self.pos.plot_path() instead.
        :marker: a valid matplotlib marker specification. Defaults to 'o'
        :alpha: the opacity of the markers. Defaults to 0.5.
        :zorder: number determining the plotting order. Increase this if the
                 spikes tend to be hidden behind other plotted features (e.g.
                 the path).
        :kwargs: additional keyword arguments passed on to axes.plot() for
                 specifying marker properties. Note especially the keywords
                 'color', 'markersize' and 'label'.
        :returns: list containing the plotted Line2D instance

        """
        if axes is None:
            axes = pyplot.gca(aspect='equal')

        h = []
        if path:
            plines = self.pos.plot_path(axes=axes)
            h += plines

        h = axes.plot(self.spike_x, self.spike_y, linestyle='None',
                      marker=marker, alpha=alpha, zorder=zorder, **kwargs)
        return h


class CellCollection(AlmostImmutable, Mapping):
    """
    Represent a collection of Cell instances (presumably from the same animal),
    and provide methods for general batch operations such as computing the
    distance matrices between cells, identifying modules using clustering
    algorithms, and create collective plots. This class subclasses Mapping to
    work as a read-only dict, and individual cell instances are looked up by
    using the cell label as key.

    """

    def __init__(self, cells, **kwargs):
        """
        Initialize the CellCollection instance

        :cells: mapping of cell labels to Cell instances
        :kwargs: none supported at the moment

        """
        self._cells = cells

    def __getitem__(self, key):
        return self._cells.__getitem__(key)

    def __iter__(self):
        return self._cells.__iter__()

    def __len__(self):
        return self._cells.__len__()

    @classmethod
    def from_session(cls, session, bins, range_, thresholds, **kwargs):
        """
        Construct a CellCollection from a single experimental session

        :session: session mapping containing at least the following fields:
            't': array-like, giving the times of position samples. Should be
                 close to regularly spaced if the kwarg speed_window is
                 provided.
            'x': array-like, giving the x coordinates of position samples
            'y': array-like, giving the y coordinates of position samples
            'spike_ts': mapping of cell labels to spike time arrays giving the
                        times at which spikes were detected from the cell.
        :bins: bin specification defining the bins to use in the firing rate
               maps. The simplest formats are a scalar 'nbins' or a tuple
               '(nbins_x, nbins_y)', giving the number of bins of equal widths
               in each direction. For information about other valid formats,
               see the documentation for numpy.histogram2d().
        :range_: range specification giving the x and y values of the outermost
                 bin edges. The format is a tuple '((xmin, xmax), (ymin,
                 ymax))'. Samples outside this region will be discarded.
        :thresholds: mapping of cell labels to peak thresholds. Used to
                     separate peaks in the autocorrelogram of the firing rate
                     of the cell. A good idea may be to use a defaultdict to
                     supply a default value, and only set explicit thresholds
                     other than the default when necessary.
        :kwargs: Passed through to the Position and Cell constructors. Note:
                 'speed_window', 'min_speed' (Position), 'filter_size' (Cell).

        """
        t, x, y, spike_ts = (session[skey]
                             for skey in ('t', 'x', 'y', 'spike_ts'))
        pos = Position(t, x, y, **kwargs)

        cells = {str(ckey): Cell(spike_t, pos, bins, range_,
                                 threshold=thresholds[ckey], **kwargs)
                 for (ckey, spike_t) in spike_ts.items()}
        return cls(cells)

    @classmethod
    def from_multiple_sessions(cls, sessions, bins, range_, thresholds,
                               **kwargs):
        """
        Construct a CellCollection from multiple experimental sessions

        :sessions: mapping of session labels to session mappings, or sequence
                   of session mappings. Each session mapping should contain at
                   least the following fields:
            't': array-like, giving the times of position samples. Should be
                 close to regularly spaced if the kwarg speed_window is
                 provided.
            'x': array-like, giving the x coordinates of position samples
            'y': array-like, giving the y coordinates of position samples
            'spike_ts': mapping of cell labels to spike time arrays giving the
                        times at which spikes were detected from the cell.
        :bins: bin specification defining the bins to use in the firing rate
               maps. The simplest formats are a scalar 'nbins' or a tuple
               '(nbins_x, nbins_y)', giving the number of bins of equal widths
               in each direction. For information about other valid formats,
               see the documentation for numpy.histogram2d().
        :range_: range specification giving the x and y values of the outermost
                 bin edges. The format is a tuple '((xmin, xmax), (ymin,
                 ymax))'. Samples outside this region will be discarded.
        :thresholds: nested mapping that maps session labels to mappings of
                     cell labels to peak thresholds. If 'sessions' is
                     a sequence, this should also be a sequence. A good idea
                     may be to use a defaultdict for each session to supply
                     a default value, and only set explicit thresholds other
                     than the default when necessary.
        :kwargs: Passed through to the Position and Cell constructors. Note:
                 'speed_window', 'min_speed' (Position), 'filter_size',
                 'threshold' (Cell).

        """
        all_cells = {}
        try:
            items = sessions.items()
        except AttributeError:
            # We didn't get a mapping. Assume the next best thing: a sequence
            items = enumerate(sessions)

        for (sskey, session) in items:
            t, x, y, spike_ts = (session[skey]
                                 for skey in ('t', 'x', 'y', 'spike_ts'))
            pos = Position(t, x, y, **kwargs)
            cells = {str(sskey) + '-' + str(ckey):
                     Cell(spike_t, pos, bins, range_,
                          threshold=thresholds[sskey][ckey], **kwargs)
                     for (ckey, spike_t) in spike_ts.items()}
            all_cells.update(cells)
        return cls(all_cells)

    def lookup(self, keys):
        """
        Look up cells using a sequence of keys

        The returned keys and cells are sorted by the keys.

        :keys: sequence of keys to look up. If None, all cells are looked
               selected.
        :returns: sorted list of keys, list of cells.

        """
        if keys is None:
            keys, cells = zip(*sorted(self.items()))
        else:
            keys = sorted(keys)
            cells = [self[key] for key in keys]

        return keys, cells

    @memoize_method
    def _mean_attribute(self, attr, keys=None):
        """
        Compute the mean of an attribute of the cells in the collection

        :attr: the Cell attribute to compute the mean over. Assumed to be
               a callable returning the requested values.
        :keys: sequence of cell keys to select cells to compute the mean peak
               locations from. If None, all cells are included.
        :returns: attribute mean

        """
        __, cells = self.lookup(keys)
        attrsum = sum(getattr(cell, attr)() for cell in cells)
        ninv = 1.0 / len(cells)
        return ninv * attrsum

    @memoize_method
    def _mean_peak_attribute(self, attr, keys=None):
        """
        Compute the mean of an attribute of the peaks of the cells in the
        collection

        Here, the attributes are assumed to be arrays with a value for each of
        the peaks in the inner ring, such that the arrays must be rolled into
        maximal peak alignment before computing the mean (see BaseCell.roll()
        for explanation of roll).

        :attr: the Cell attribute to compute the mean over. Assumed to be
               a callable returning the requested values in the form of an
               array with entries for each peak.
        :keys: sequence of cell keys to select cells to compute the mean peak
               locations from. If None, all cells are included.
        :returns: attribute mean

        """
        __, cells = self.lookup(keys)
        refcell = cells[0]

        attrsum = 0.0
        for cell in cells:
            roll = cell.roll(refcell)
            attrsum += numpy.roll(getattr(cell, attr)(), roll, axis=0)
        ninv = 1.0 / len(cells)
        return ninv * attrsum

    def mean_firing_field(self, keys=None):
        """
        Compute the mean firing field of cells in the collection

        :keys: sequence of cell keys to select cells to compute the mean firing
               field from. If None, all cells are included.
        :returns: the mean firing field covariance matrix

        """
        return self._mean_attribute('firing_field', keys=keys)

    def mean_peaks(self, keys=None):
        """
        Compute the mean locations of the inner six peaks of cells in the
        collection

        :keys: sequence of cell keys to select cells to compute the mean peak
               locations from. If None, all cells are included.
        :returns: numpy array of mean peak locations

        """
        return self._mean_peak_attribute('peaks', keys=keys)

    def mean_peaks_polar(self, keys=None):
        """
        Compute the mean polar coordinates of the inner six peaks of cells in
        the collection

        :keys: sequence of cell keys to select cells to compute the mean peak
               polar coordinates from. If None, all cells are included.
        :returns: numpy array of mean peak polar coordinates

        """
        return self._mean_peak_attribute('peaks_polar', keys=keys)

    @memoize_method
    def stacked_firing_rate(self, keys=None, normalize=None, threshold=None):
        """
        Compute the stacked firing rate map of cells in the collection

        The stacked firing rate map is defined as the average of the firing
        rates of all the cells.

        :keys: sequence of cell keys to select cells to compute the stacked
               firing rate from. If None, all cells are included.
        :normalize: string to select how to normalize the rate maps before
                    stacking. Possible values:
            None: no normalization is performed
            'max': the maximum value of each rate map will be normalized to 1.0
            'mean': the mean of the rate maps will be normalized to 1.0
            'std': the standard deviation of the rate maps will be normalized
                   to 1.0
            'zscore': the rate maps are replaced with the correpsonding Z-score
                      maps: for each rate map, its mean is be subtracted and
                      the result is be divided by the standard deviation.
        'threshold': if not None, each firing rate will be transformed to
                     a binary variable with the value 1 in bins where the
                     normalized firing rate exceeds `threshold`, and
                     0 otherwise.
        :returns: IntensityMap2D instance containing the stacked firing rate.

        """
        __, cells = self.lookup(keys)

        if normalize is None:
            def _norm(imap):
                return imap
        elif normalize == 'max':
            def _norm(imap):
                return imap / imap.max()
        elif normalize == 'mean':
            def _norm(imap):
                return imap / imap.mean()
        elif normalize == 'std':
            def _norm(imap):
                return imap / imap.std()
        elif normalize == 'zscore':
            def _norm(imap):
                return (imap - imap.mean()) / imap.std()
        else:
            raise ValueError("unknown normalization: {}".format(normalize))

        if threshold is None:
            norm = _norm
        else:
            def norm(imap):
                imap = _norm(imap)
                return (imap > threshold).astype(numpy.float_)

        return IntensityMap2D.mean_map((norm(cell.firing_rate)
                                        for cell in cells),
                                       ignore_missing=True)

    @memoize_method
    def distances(self, keys1=None, keys2=None, **kwargs):
        """
        Compute a distance matrix between cells

        Parameters
        ----------
        keys1, keys2 : sequence, optional
            Sequences of cell keys to select cells to compute the distance
            matrix between. If `None`, all cells are included.
        kwargs : dict, optional
            Keyword arguments to pass to `BaseCell.features`.

        Returns
        -------
        DataFrame
            The distance matrix, indexed along rows and columns by the cell
            keys.

        """
        keys1, cells1 = self.lookup(keys1)
        keys2, cells2 = self.lookup(keys2)

        distdict = {}
        for (key1, cell1) in zip(keys1, cells1):
            dists = [cell1.distance(cell2, **kwargs)
                     for cell2 in cells2]
            distdict[key1] = pandas.Series(dists, index=keys2)

        distmatrix = pandas.DataFrame(distdict).transpose()

        return distmatrix

    @memoize_method
    def features(self, keys=None, **kwargs):
        """
        Compute a feature array of the cells

        The feature series comprising the array are computed with the peak roll
        for each cell required for global consistency.

        Parameters
        ----------
        keys : sequence, optional
            Sequence of cell keys to select cells to compute the feature array
            for. If `None`, all cells are included.
        kwargs : dict, optional
            Keyword arguments to pass to `BaseCell.features`.

        Returns
        -------
        DataFrame
            Feature array. The DataFrame row indices are the cell keys, while
            the DataFrame columns contain the features and are labelled by
            appropriate Latex-formatted strings.

        """
        keys, cells = self.lookup(keys)
        refcell = cells[0]

        featdict = {}
        for (key, cell) in zip(keys, cells):
            roll = cell.roll(refcell)
            featdict[key] = cell.features(roll=roll, **kwargs)

        features = pandas.DataFrame(featdict).transpose()

        return features

    def dbscan(self, eps, min_samples, keys=None, features_kw=None,
               mod_kw=None, **kwargs):
        """
        Use the DBSCAN clustering algorithm to find modules

        Parameters
        ----------
        eps : scalar
            Maximum distance for points to be counted as neighbors.
        min_samples : integer
            Minimum number of neighbors for a point to be considered a core
            point.
        keys : sequence, optional
            Sequence of cell keys to select cells to search for modules among.
            If `None`, all cells are included.
        features_kw : dict, optional
            Keyword arguments to pass to `BaseCell.features`.
        mod_kw : dict, optional
            Keyword arguments to pass to the factory method
            `CellCollection.modules_from_labels`.
        **kwargs : dict, optional
            Keyword arguments passed on to `cluster.dbscan()`.

        Returns
        -------
        list
            List of `Module` instances containing the detected modules.
        CellCollection
            CellCollection containing any outlier cells.

        """
        if features_kw is None:
            features_kw = {}

        features = self.features(keys=keys, **features_kw)
        keys, feature_arr = features.index, features.values
        labels = cluster.dbscan(feature_arr, eps=eps,
                                min_samples=min_samples, **kwargs)[1]

        if mod_kw is None:
            mod_kw = {}

        return self.modules_from_labels(keys, labels, **mod_kw)

    def mean_shift(self, keys=None, features_kw=None, mod_kw=None, **kwargs):
        """
        Use the mean shift clustering algorithm to find modules

        Parameters
        ----------
        keys : sequence, optional
            Sequence of cell keys to select cells to search for modules among.
            If `None`, all cells are included.
        features_kw : dict, optional
            Keyword arguments to pass to `BaseCell.features`.
        mod_kw : dict, optional
            Keyword arguments to pass to the factory method
            `CellCollection.modules_from_labels`.
        **kwargs : dict, optional
            Keyword arguments passed on to `cluster.mean_shift()`.

        Returns
        -------
        list
            List of `Module` instances containing the detected modules.
        CellCollection
            CellCollection containing any outlier cells.

        """
        if features_kw is None:
            features_kw = {}

        features = self.features(keys=keys, **features_kw)
        keys, feature_arr = features.index, features.values

        labels = cluster.mean_shift(feature_arr, cluster_all=False,
                                    **kwargs)[1]

        if mod_kw is None:
            mod_kw = {}

        return self.modules_from_labels(keys, labels, **mod_kw)

    def k_means(self, n_clusters, n_runs=1, keys=None, features_kw=None,
                mod_kw=None, **kwargs):
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
        keys : sequence, optional
            Sequence of cell keys to select cells to search for modules among.
            If `None`, all cells are included.
        features_kw : dict, optional
            Keyword arguments to pass to `BaseCell.features`.
        mod_kw : dict, optional
            Keyword arguments to pass to the factory method
            `CellCollection.modules_from_labels`.
        **kwargs : dict, optional
            Keyword arguments passed on to `cluster.k_means()`.

        Returns
        -------
        list
            List of `Module` instances containing the detected modules.
        CellCollection
            CellCollection containing any outlier cells.

        """
        if features_kw is None:
            features_kw = {}

        features = self.features(keys=keys, **features_kw)
        keys, feature_arr = features.index, features.values

        __, labels, inertia = cluster.k_means(feature_arr, n_clusters,
                                              **kwargs)
        for __ in range(n_runs - 1):
            __, lbls, inrt = cluster.k_means(feature_arr, n_clusters, **kwargs)
            if inrt < inertia:
                inertia = inrt
                labels = lbls

        if mod_kw is None:
            mod_kw = {}

        return self.modules_from_labels(keys, labels, **mod_kw)

    def modules_from_labels(self, keys, labels, min_length=4, **kwargs):
        """
        Use a list of keys and corresponding labels to instantiate Module
        instances

        Parameters
        ----------
        keys : sequence
            Sequence of cell keys to create modules from.
        labels : sequence
            Sequence of labels corresponding to the `keys`.
            All cells with the same label become a module. The special label
            `-1` denotes outliers.
        min_length : integer, optional
            The minimum number of cells in a module. Tentative modules with
            fewer cells than this are merged into the outliers.
        kwargs : dict, optional
            Keyword arguments passed to the `Module` constructor.
        :mod_kw: dict of keyword arguments to pass to the Module constructor.
        :returns: sequence of module instances, sorted by mean grid scale;
                  Cellcollection instance containing any outliers.

        """
        tentative_modules_ = {}
        outliers_ = {}
        for (key, label) in zip(keys, labels):
            if label == -1:
                outliers_[key] = self[key]
            else:
                try:
                    tentative_modules_[label][key] = self[key]
                except KeyError:
                    tentative_modules_[label] = {key: self[key]}
        modules_ = []
        for mod in tentative_modules_.values():
            if len(mod) < min_length:
                outliers_.update(mod)
            else:
                modules_.append(mod)
        outliers = CellCollection(outliers_)
        modules = [Module(mod, **kwargs) for mod in modules_]
        modules.sort(key=(lambda mod: mod.template.scale()))

        return modules, outliers

    def plot_scales(self, axes=None, keys=None, marker='o', mean=True,
                    mean_kw=None, **kwargs):
        """
        Plot the grid scales of cells in the cell collection

        The scales can be added to an existing plot via the optional 'axes'
        argument. In this case, the scale markers are added to the right of the
        current x limits.

        :axes: Axes instance to add the scales to. If None (default), the
               current Axes instance is used if any, or a new one created.
        :keys: sequence of cell keys to select cells to plot scales for. If
               None (default), all cells are included.
        :marker: a valid matplotlib marker specification. Defaults to 'o'
        :mean: if True, add a line showing the mean of the plotted scales. By
               default, a gray (color == 0.5) thin (linewidth == 0.5) line is
               used, but this can be overridden using the parameter mean_kw.
        :mean_kw: dict of keyword arguments to pass to the axes.plot() method
                  used to plot the mean. Default: None (empty dict)
        :kwargs: additional keyword arguments passed on to the axes.plot()
                 method used to plot scales. Note in particular the keywords
                 'markersize', 'color' and 'label'.
        :returns: a list of the plotted Line2D instances

        """
        if axes is None:
            axes = pyplot.gca()

        if mean_kw is None:
            mean_kw = {}

        features = self.features(keys=keys, extra=True)
        scales = features[features_index['l']]

        # Start plotting from the current right end of the plot
        xlim = axes.get_xlim()
        right = xlim[1]
        xlocs = numpy.arange(right, right + len(scales))

        lines = axes.plot(xlocs, scales, linestyle='None', marker=marker,
                          **kwargs)

        if mean:
            mscale = numpy.empty_like(xlocs)
            mscale.fill(scales.mean())
            lines += axes.plot(xlocs, mscale, linewidth=0.5, color='0.50',
                               **mean_kw)

        # Add ticklabels and rotate
        add_ticks(axes.xaxis, xlocs, scales.index)
        pyplot.xticks(axes=axes, rotation='vertical')

        # Set limits so the plot is ready for another round
        axes.set_xlim((xlim[0], xlocs[-1] + 1.0))

        # This does not belong here anymore, but don't forget the syntax!
        #fig.subplots_adjust(bottom=0.2)

        return lines

    def plot_angles(self, axes=None, keys=None, marker='o', mean=True,
                    mean_kw=None, **kwargs):
        """
        Plot the grid angles of cells in the cell collection

        The angles can be added to an existing plot via the optional 'axes'
        argument. In this case, the angle markers are added to the right of the
        current x limits.

        :axes: Axes instance to add the angles to. If None (default), the
               current Axes instance is used if any, or a new one created.
        :keys: sequence of cell keys to select cells to plot angles for. If
               None (default), all cells are included.
        :marker: a valid matplotlib marker specification. Defaults to 'o'
        :mean: if True, add lines showing the means of the plotted angles. By
               default, a gray (color == 0.5) thin (linewidth == 0.5) line is
               used, but this can be overridden using the parameter mean_kw.
        :mean_kw: dict of keyword arguments to pass to the axes.plot() method
                  used to plot the mean. Default: None (empty dict)
        :kwargs: additional keyword arguments passed on to the axes.plot()
                 method used to plot angles. Note in particular the keywords
                 'markersize', 'color' and 'label'.
        :returns: a list of the plotted Line2D instances

        """
        if axes is None:
            axes = pyplot.gca()

        features = self.features(keys=keys, extra=True)
        index = [features_index[label]
                 for label in ('beta1', 'beta2', 'beta3')]
        angles = numpy.rad2deg(features[index])

        # Start plotting from the current right end of the plot
        xlim = axes.get_xlim()
        right = xlim[1]
        xlocs = numpy.arange(right, right + len(angles))

        # Create a threefold tile of xlocs separated by a nan. This is a trick
        # to stack three lines on top of each other as part of the same Line2D
        # instance, without connecting the end of one with the beginning of the
        # next.
        xlocs_nan = numpy.hstack((xlocs, numpy.nan))
        xlocs3 = numpy.tile(xlocs_nan, 3)
        angles_flat = numpy.hstack((angles[index[0]], numpy.nan,
                                    angles[index[1]], numpy.nan,
                                    angles[index[2]], numpy.nan))

        # Plot the angles
        lines = axes.plot(xlocs3, angles_flat, linestyle='None', marker=marker,
                          **kwargs)

        if mean:
            if mean_kw is None:
                mean_kw = {}

            mangles = angles.mean()
            ma1 = numpy.empty_like(xlocs)
            ma2 = numpy.empty_like(xlocs)
            ma3 = numpy.empty_like(xlocs)
            ma1.fill(mangles[index[0]])
            ma2.fill(mangles[index[1]])
            ma3.fill(mangles[index[2]])
            mangles_flat = numpy.hstack((ma1, numpy.nan,
                                         ma2, numpy.nan,
                                         ma3, numpy.nan))
            lines += axes.plot(xlocs3, mangles_flat, linewidth=0.5,
                               color='0.50', **mean_kw)

        # Add ticklabels and rotate
        add_ticks(axes.xaxis, xlocs, angles.index)
        pyplot.xticks(axes=axes, rotation='vertical')

        # Set limits so the plot is ready for another round
        axes.set_xlim((xlim[0], xlocs[-1] + 1.0))

        # This does not belong here anymore, but don't forget the syntax!
        #fig.subplots_adjust(bottom=0.2)

        return lines

    def plot_ellpars(self, axes=None, keys=None, marker='o', mean=None,
                     mean_kw=None, **kwargs):
        """
        Plot the ellipse parameters for cells in the cell collection

        The parameters are visualized in a polar plot with eccentricity as the
        radius and twice the ellipse tilt as the angle. The tilt is doubled
        because ellipse tilts are equivalent modulo pi radians, whereas polar
        coordinates are equivalent modulo 2 * pi radians.

        Parameters
        ----------
        axes: Axes, optional
            Axes instance to add the ellipse parameters to. If None (default),
            the current Axes instance is used if any, or a new one created.
        keys : sequence, optional
            Sequence of cell keys to select cells to plot ellipse parameters
            for. If None, all cells are included.
        marker : valid matplotlib marker specification.
            Marker to use plot the ellipse parameters as.
        mean : bool, optional
            If True, add a point showing the mean of the ellipse parameters.
            The mean is computed in the cartesian coordinate representation of
            the ellpars as they are plotted by this method.
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

        features = self.features(keys=keys, extra=True)
        index = [features_index[label] for label in ('epsilon', '2theta')]
        ellpars = features[index]

        lines = axes.plot(ellpars[index[1]], ellpars[index[0]],
                          linestyle='None', marker='o', **kwargs)

        if mean:
            if mean_kw is None:
                mean_kw = {}
            index = [features_index[label] for label in ('xell', 'yell')]
            cartesian_ellpars = features[index]
            mean_cartesian_ellpars = cartesian_ellpars.mean()
            mean_xell = mean_cartesian_ellpars[index[0]]
            mean_yell = mean_cartesian_ellpars[index[1]]
            mean_tilt_2 = numpy.arctan2(mean_yell, mean_xell)
            mean_ecc = numpy.sqrt(mean_xell * mean_xell +
                                  mean_yell * mean_yell)

            lines += axes.plot(mean_tilt_2, mean_ecc, linestyle='None',
                               marker='o', color='0.50', **mean_kw)

        axes.set_ylim((0.0, 1.0))

        return lines

    def plot_peaks(self, axes=None, keys=None, marker='o', mean=False,
                   mean_kw=None, **kwargs):
        """
        Plot the peak locations for cells in the cell collection

        Parameters
        ----------
        axes: Axes, optional
            Axes instance to add the peaks to. If None (default), the current
            Axes instance is used if any, or a new one created.
        keys : sequence, optional
            Sequence of cell keys to select cells to plot peaks for. If None,
            all cells are included.
        marker : valid matplotlib marker specification.
            Marker to use plot the ellipse parameters as.
        mean : bool, optional
            If True, add points showing the means of the plotted peaks.
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

        peaks = numpy.vstack([cell.peaks_polar() for cell in self.values()])

        lines = axes.plot(peaks[:, 1], peaks[:, 0], linestyle='None',
                          marker='o', **kwargs)

        if mean:
            if mean_kw is None:
                mean_kw = {}
            mean_peaks = numpy.mean([cell.peaks() for cell in self.values()],
                                    axis=0)
            mean_peakx = mean_peaks[:, 0]
            mean_peaky = mean_peaks[:, 1]
            mean_beta = numpy.arctan2(mean_peaky, mean_peakx)
            mean_l_al = numpy.sqrt(mean_peakx * mean_peakx +
                                   mean_peaky * mean_peaky)

            lines += axes.plot(mean_beta, mean_l_al, linestyle='None',
                               marker='o', color='0.50', **mean_kw)

        return lines

    def plot_stacked_firing_rate(self, axes=None, keys=None, normalize=None,
                                 cax=None, cmap=None, cbar_kw=None, **kwargs):
        """
        Plot the stacked firing rate map for cells in the collection

        The stacked firing rate can be added to an existing plot via the
        optional 'axes' argument.

        This method is just a wrapper around self.stacked_firing_rate.plot().

        :axes: Axes instance to add the intensity map to. If None (default),
               a new Figure is created (this method never plots to the current
               Figure or Axes). In the latter case, equal aspect ration will be
               enforced on the newly created Axes instance.
        :keys: sequence of cell keys to select cells to plot the stacked firing
               rate of. If None (default), all cells are included.
        :normalize: string to select how to normalize the rate maps before
                    stacking. See self.stacked_firing_rate for details. Note
                    that this method does not support the optional `threshold`
                    keyword to self.stacked_firing_rate -- if required, use
                    self.stacked_firing_rate.plot instead.
        :cax: Axes instance to plot the colorbar into. If None (default),
              matplotlib automatically makes space for a colorbar on the
              right-hand side of the plot.
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
        sfiring_rate = self.stacked_firing_rate(keys=keys, normalize=normalize)
        axes, cbar = sfiring_rate.plot(axes=axes, cax=cax, cmap=cmap,
                                       cbar_kw=cbar_kw, **kwargs)
        return axes, cbar


class Module(CellCollection):
    """
    Represent a module of grid cells and provide methods for analyzing the grid
    phases

    """

    def __init__(self, cells, threshold=0.25, **kwargs):
        """
        Initialize the Module instance

        :cells: a mapping of cell labels to Cell instances belonging to
                a single module.
        :threshold: peak threshold for the TemplateGridCell instance
                    constructed by the module.
        :kwargs: passed through to the CellCollection constructor

        """
        CellCollection.__init__(self, cells, **kwargs)

        # Compute a large BinnedSet2D for the ideal cell
        bset = self.values()[0].firing_rate.bset
        new_bset = bset
        for __ in range(3):
            new_bset = new_bset.correlate(bset, mode='full')

        # Construct the ideal cell and save it to the instance
        self.template = TemplateGridCell(self.mean_peaks(),
                                         self.mean_firing_field(),
                                         new_bset, threshold=threshold)

    @memoize_method
    def phases(self, keys=None):
        """
        Compute the grid phase of cells in the module.

        The phase is defined with respect to the TemplateGridCell instance
        belonging to the module (self.template).

        :keys: sequence of cell keys to select cells to compute the phase for.
               If None, all cells are included.
        :returns: dict containing the phases

        """
        keys, cells = self.lookup(keys)
        phases = {key: cell.phase(self.template)
                  for (key, cell) in zip(keys, cells)}

        return phases

    @memoize_method
    def phase_pattern(self, edge_correction='periodic', keys=None):
        """
        Create a PointPattern instance of the phases of the cells in the module

        Parameters
        ----------
        edge_correction : str {'stationary', 'finite', 'isotropic', 'periodic',
                               'plus'}, optional
            String to select the default edge handling to apply in computations
            involving the returned PointPattern instance. See the documentation
            for `PointPattern` for details.
        keys : sequence
            Keys to select cells from which to include the phase in the
            PointPattern. If None, all cells are included.

        Returns
        -------
        PointPattern
            PointPattern instance representning the phases.

        """
        phases = self.phases(keys=keys).values()
        peak_pattern = numpy.vstack(((0.0, 0.0), self.template.peaks()))
        window = spatial.Voronoi(peak_pattern).vertices
        angles = numpy.arctan2(window[:, 1], window[:, 0])
        sort_ind = numpy.argsort(angles)
        window = window[sort_ind]
        return PointPattern(phases, window, edge_correction=edge_correction)

    def plot_phases(self, axes=None, keys=None, periodic=False, **kwargs):
        """
        Plot the absolute grid phases of the cells in the module

        The phases can be added to an existing plot via the optional 'axes'
        argument.

        :axes: Axes instance to add the phases to. Passed through to the
               PointPattern.plot_pattern() method.
        :keys: sequence of cell keys to select cells to include in the
               phase pattern. If None, all cells are included.
        :periodic: if True, the plot is peridically extended along the grid
                   lattice vectors.
        :kwargs: additional keyword arguments passed on to the
                 PointPattern.plot_pattern() method. Note in particular the
                 keywords 'window', 'window_kw' and 'plus_kw'.
        :returns: a list of the plotted objects, as returned from
                  PointPattern.plot_pattern()

        """
        phase_pattern = self.phase_pattern(keys=keys)
        return phase_pattern.plot_pattern(axes=axes, periodic=periodic,
                                          **kwargs)

features_index = {
    'log_l': r"$\log{l}$",
    'ax1': r"$a_{x,1} / l$",
    'ay1': r"$a_{y,1} / l$",
    'ax2': r"$a_{x,2} / l$",
    'ay2': r"$a_{y,2} / l$",
    'ax3': r"$a_{x,3} / l$",
    'ay3': r"$a_{y,3} / l$",
    'l': r"$l$",
    'al1': r"$l_1 / l$",
    'beta1': r"$\beta_1$",
    'al2': r"$l_2 / l$",
    'beta2': r"$\beta_2$",
    'al3': r"$l_3 / l$",
    'beta3': r"$\beta_3$",
    'xell': r"$\epsilon \cos 2 \theta$",
    'yell': r"$\epsilon \sin 2 \theta$",
    'epsilon': r"$\epsilon$",
    '2theta': r"$2 \theta$",
}
