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
from .pointpatterns import PointPattern, PointPatternCollection
from .external.memoize import memoize_function, memoize_method


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

    def plot_path(self, axes=None, linewidth=0.5, color='0.5', **kwargs):
        """
        Plot the path through the valid positions

        The path can be added to an existing plot via the optional 'axes'
        argument.

        :axes: Axes instance to add the path to. If None (default), the current
               Axes instance with equal aspect ratio is used if any, or a new
               one created.
        :linewidth: number giving the width of the plotted path. Defaults: 0.5
        :color: a valid matplotlib color specification giving the color to plot
                the path with. Defaults to '0.5', a moderate gray.
        :kwargs: additional keyword arguments passed on to axes.plot() for
                 specifying line properties. Note especially the keywords
                 'linestyle' and 'label'.
        :returns: the plotted Line2D instance

        """
        if axes is None:
            axes = pyplot.gca(aspect='equal')

        h = axes.plot(self.x, self.y, linewidth=linewidth, color=color,
                      **kwargs)
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

    def autocorrelate(self, mode='full', pearson=True, normalized=True):
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
        radii = px * px + py * py
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
            radii = self.peaks_polar(project=project, threshold=threshold)
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

    def correlate(self, other, mode='full', pearson=True, normalized=True):
        """
        Compute the cross-correlaogram of another cell's firing rate to this

        This is a wrapper around self.firing_rate.correlate().

        :other: another Cell instance.
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
        :returns: IntensityMap2D instance representing the cross-correlogram of
                  the firing rates

        """
        return self.firing_rate.correlate(other.firing_rate, mode=mode,
                                          pearson=pearson,
                                          normalized=normalized)

    def features(self, roll=0):
        """
        Compute a series of features of this cell

        The purpose of the feature series is to embed the cell into
        a high-dimensional space where the euclidean distance between cells
        correspond to some concept of closeness. This is useful for clustering
        cells into modules.

        The present definition consists of the x and y coordinates of each of
        the six inner peaks in the autocorrelogram, scaled by the factor log(r)
        / r, where r = self.scale().

        :roll: quantities related to individual peaks are listed in
               counterclockwise order in the feature array. The parameter
               'roll' decides which peak to start at: self.peaks()[roll] is
               used first, and self.peaks()[roll - 1] last (this can be thought
               of as replacing self.peaks() with numpy.roll(self.peaks(), roll,
               axis=0)).
        :returns: one-dimensional array of features

        """
        scale = self.scale()
        peaks = numpy.roll(self.peaks(), roll, axis=0)
        return peaks.ravel() * numpy.log(scale) / scale

    @memoize_method
    def distance(self, other):
        """
        Compute a distance between the grid patterns of grid cells

        This method defines a metric on the space of grid patterns from grid
        cells. The distance is defined as the Euclidean distance between the
        feature arrays of the cells, using the relative peak roll that
        minimizes this distance.

        :other: Cell instance to measure distance to.
        :returns: distance between cells, and the peak roll applied to this
                  cell to obtain it. The peak roll  is defined such that
                  numpy.roll(self.peaks, roll, axis=0) and other.peaks() give
                  coordinates to the most closely corresponding peaks in self
                  and other

        """
        if self is other:
            distance = 0.0
            roll = 0

        else:
            ofeat = other.features(roll=0)
            dfeat = self.features(roll=0) - ofeat
            distance = numpy.sum(dfeat * dfeat)
            roll = 0

            # To make sure that the most closesly corresponding peaks are used,
            # the metric is computed as the minimum of three different relative
            # peak orderings.
            for r in (-1, 1):
                dfeat = self.features(roll=r) - ofeat
                dist = numpy.sum(dfeat * dfeat)
                if dist < distance:
                    distance = dist
                    roll = r

        return distance, roll

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

        The firing rate can be added to an existing plot via the optional
        'axes' argument.

        This method is just a wrapper around self.firing_rate.plot().

        :axes: Axes instance to add the intensity map to. If None (default),
               a new Figure is created (this method never plots to the current
               Figure or Axes). In the latter case, equal aspect ration will be
               enforced on the newly created Axes instance.
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
        axes, cbar = self.firing_rate.plot(axes=axes, cax=cax, cmap=cmap,
                                           cbar_kw=cbar_kw, **kwargs)
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
                         peaks=False, ellipse=False, cmap=None, cbar_kw=None,
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
        :peaks: if True, add the most central peak and the six peaks closest to
                it to the plot, using some (hopefully) sensible plotting
                defaults. If more control is required, call
                self.detect_central_peaks() on self.correlogram(other) to get
                the peaks, and add them to the plot manually.
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
        if peaks or ellipse:
            pks, __ = self.detect_central_peaks(corr, self.threshold)
            if peaks:
                axes.plot(pks[:, 0], pks[:, 1], linestyle='None', marker='o',
                          color='black', **kwargs)
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
                                 vmin=-1.0, vmax=1.0, cmap=cmap,
                                 cbar_kw=cbar_kw, **kwargs)

        return axes, cbar


class IdealGridCell(BaseCell):
    """
    Represent an idealized grid cell defined only using the lattice vectors and
    the firing field covariance matrix. Can for example be used to represent
    the average cell from a module.

    """

    def __init__(self, peaks, firing_field, bset, threshold):
        """
        Initialize the IdealGridCell instance

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

    def plot_spikes(self, axes=None, path=False, marker='o', alpha=0.5,
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
        maximal peak alignment before computing the mean (see Cell.distance()
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
            __, roll = cell.distance(refcell)
            attrsum += numpy.roll(getattr(cell, attr)(), roll, axis=0)
        ninv = 1.0 / len(cells)
        return ninv * attrsum

    def mean_scale(self, keys=None):
        """
        Compute the mean scale of cells in the collection

        :keys: sequence of cell keys to select cells to compute the mean scale
               from. If None, all cells are included.
        :returns: the mean scale

        """
        return self._mean_attribute('scale', keys=keys)

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

    def mean_ellpars(self, mode='direct', keys=None):
        """
        Compute the mean tilt and eccentricity of the ellipses fitted to the
        inner six peaks of cells in the collection

        :mode: string to select how to compute the means. Possible values:
            'direct': the returned tilt is the mean of the tilts from each
                      cell, and the returned eccentricity is the mean of the
                      eccentricities from each cell
            'euclidean': the parameters are mapped into the plane using (ecc,
                         2 * tilt) as polar coordinates (r, theta), and the
                         mean of the corresponding cartesian coordinates is
                         computed, before converting back to the mean
                         eccentricity and tilt
        :keys: sequence of cell keys to select cells to compute the mean
               ellipse parameters from. If None, all cells are included.
        :returns: mean tilt, mean eccentricity

        """
        __, cells = self.lookup(keys)
        ninv = 1.0 / len(cells)
        if mode == 'direct':
            tiltsum, eccsum = 0.0, 0.0
            for cell in cells:
                ell = cell.ellipse()
                tiltsum += ell.tilt
                eccsum += ell.ecc
            mean_tilt, mean_ecc = ninv * tiltsum, ninv * eccsum
        elif mode == 'euclidean':
            xsum, ysum = 0.0, 0.0
            for cell in cells:
                ell = cell.ellipse()
                tilt2, ecc = 2.0 * ell.tilt, ell.ecc
                x, y = ecc * numpy.cos(tilt2), ecc * numpy.sin(tilt2)
                xsum += x
                ysum += y
            xmean, ymean = ninv * xsum, ninv * ysum
            mean_tilt = .5 * numpy.arctan2(ymean, xmean)
            mean_ecc = numpy.sqrt(xmean * xmean + ymean * ymean)
        else:
            raise ValueError("unknown mode: {}".format(mode))

        return mean_tilt, mean_ecc

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

        return sum(norm(cell.firing_rate) for cell in cells) / len(cells)

    @memoize_method
    def distances(self, keys1=None, keys2=None):
        """
        Compute a distance matrix between cells

        :keys1, keys2: sequences of cell keys to select cells to compute the
                       distance matrix between. If None, all cells are
                       included.
        :returns: DataFrame containing the distance matrix, DataFrame
                  containing the roll matrix. Both DataFrames are indexed along
                  rows and columns by the cell keys (keys). The roll matrix is
                  organized such that rollmatrix[key1][key2] gives the roll to
                  apply to self[key2] to align it with self[key1] (see
                  BaseCell.distance() for the full explanation of roll).

        """
        keys1, cells1 = self.lookup(keys1)
        keys2, cells2 = self.lookup(keys2)

        distdict, rolldict = {}, {}
        for (key1, cell1) in zip(keys1, cells1):
            dist, roll = zip(*[cell1.distance(cell2) for cell2 in cells2])
            distdict[key1] = pandas.Series(dist, index=keys2)
            rolldict[key1] = pandas.Series(roll, index=keys2)

        distmatrix = pandas.DataFrame(distdict).transpose()
        rollmatrix = pandas.DataFrame(rolldict).transpose()

        return distmatrix, rollmatrix

    @memoize_method
    def features(self, keys=None):
        """
        Compute a feature array of the cells

        The feature series comprising the array are computed with the roll
        required for consistency.

        :keys: sequence of cell keys to select cells to compute the feature
               array for. If None, all cells are included.
        :returns: DataFrame containing the feature array. The DataFrame row
                  indices are the cell keys, while the DataFrame columns
                  contain the features.

        """
        keys, cells = self.lookup(keys)
        refcell = cells[0]

        featdict = {}
        for (key, cell) in zip(keys, cells):
            __, roll = cell.distance(refcell)
            featdict[key] = cell.features(roll=roll)

        features = pandas.DataFrame(featdict).transpose()

        return features

    def dbscan(self, eps, min_samples, keys=None, mod_kw=None, **kwargs):
        """
        Use the DBSCAN clustering algorithm to find modules in the collection
        of cells

        'eps': maximum distance for points to be counted as neighbors
        'min_samples': minimum number of neighbors for a point to be considered
                       a core point
        :keys: sequence of cell keys to select cells to search for modules
               among. If None (default), all cells are included.
        :mod_kw: dict of keyword arguments to pass to the Module constructor.
        :kwargs: passed through to cluster.dbscan()
        :returns: list of Module instances, and a CellCollection instance
                  containing any outliers

        """
        features = self.features(keys=keys)
        keys, feature_arr = features.index, features.values
        labels = cluster.dbscan(feature_arr, eps=eps,
                                min_samples=min_samples)[1]

        return self.modules_from_labels(keys, labels, mod_kw=mod_kw)

    def mean_shift(self, keys=None, mod_kw=None, **kwargs):
        """
        Use the mean shift clustering algorithm to find modules in the
        collection of cells

        :keys: sequence of cell keys to select cells to search for modules
               among. If None (default), all cells are included.
        :mod_kw: dict of keyword arguments to pass to the Module constructor.
        :kwargs: passed through to cluster.mean_shift()
        :returns: list of Module instances, and a CellCollection instance
                  containing any outliers

        """
        features = self.features(keys=keys)
        keys, feature_arr = features.index, features.values

        labels = cluster.mean_shift(feature_arr, cluster_all=False,
                                    **kwargs)[1]

        return self.modules_from_labels(keys, labels, mod_kw=mod_kw)

    def k_means(self, n_clusters, keys=None, mod_kw=None, **kwargs):
        """
        Use the K-means clustering algorithm to find modules in the collection
        of cells

        :n_clusters: the number of clusters (and thus modules) to form
        :keys: sequence of cell keys to select cells to search for modules
               among. If None (default), all cells are included.
        :mod_kw: dict of keyword arguments to pass to the Module constructor.
        :kwargs: passed through to cluster.k_means()
        :returns: list of Module instances, and a CellCollection instance
                   containing any outliers

        """
        features = self.features(keys=keys)
        keys, feature_arr = features.index, features.values

        labels = cluster.k_means(feature_arr, n_clusters, **kwargs)[1]

        return self.modules_from_labels(keys, labels, mod_kw=mod_kw)

    def modules_from_labels(self, keys, labels, mod_kw=None):
        """
        Use a list of keys and corresponding labels to instantiate Module
        instances

        :keys: sequence of cell keys that have been grouped into modules
        :labels: sequence of labels corresponding to the cells given by 'keys';
                 cells with the same label are grouped into the same module.
                 The label -1 denotes outliers.
        :mod_kw: dict of keyword arguments to pass to the Module constructor.
        :returns: sequence of module instances, sorted by mean grid scale;
                  Cellcollection instance containing any outliers.

        """
        if mod_kw is None:
            mod_kw = {}

        modules_ = {}
        outliers_ = {}
        for (key, label) in zip(keys, labels):
            if label == -1:
                outliers_[key] = self[key]
            else:
                try:
                    modules_[label][key] = self[key]
                except KeyError:
                    modules_[label] = {key: self[key]}

        outliers = CellCollection(outliers_)
        modules = [Module(mod, **mod_kw) for mod in modules_.values()]
        modules.sort(key=(lambda mod: mod.mean_scale()))

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

        keys, cells = self.lookup(keys)
        scales = [cell.scale() for cell in cells]

        # Start plotting from the current right end of the plot
        xlim = axes.get_xlim()
        right = xlim[1]
        xlocs = numpy.arange(right, right + len(cells))

        lines = axes.plot(xlocs, scales, linestyle='None', marker=marker,
                          **kwargs)

        if mean:
            mscale = numpy.empty_like(xlocs)
            mscale.fill(self.mean_scale(keys=keys))
            lines += axes.plot(xlocs, mscale, linewidth=0.5, color='0.50',
                               **mean_kw)

        # Add ticklabels and rotate
        add_ticks(axes.xaxis, xlocs, keys)
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

        if mean_kw is None:
            mean_kw = {}

        keys, cells = self.lookup(keys)
        refcell = cells[0]

        alpha, beta, gamma = [], [], []
        for cell in cells:
            __, roll = cell.distance(refcell)
            angles = numpy.rad2deg(numpy.roll(cell.peaks_polar()[:3, 1], roll))
            alpha.append(angles[0])
            beta.append(angles[1])
            gamma.append(angles[2])

        # Start plotting from the current right end of the plot
        xlim = axes.get_xlim()
        right = xlim[1]
        xlocs = numpy.arange(right, right + len(cells))

        # Create a threefold tile of xlocs separated by a nan. This is a trick
        # to stack three lines on top of each other as part of the same Line2D
        # instance, without connecting the end of one with the beginning of the
        # next.
        xlocs_nan = numpy.hstack((xlocs, numpy.nan))
        xlocs3 = numpy.tile(xlocs_nan, 3)
        abg = numpy.hstack((alpha, numpy.nan,
                            beta, numpy.nan,
                            gamma, numpy.nan))

        # Plot the angles
        lines = axes.plot(xlocs3, abg, linestyle='None', marker=marker,
                          **kwargs)

        if mean:
            malpha, mbeta, mgamma = numpy.rad2deg(
                self.mean_peaks_polar(keys=keys)[:3, 1])
            ma = numpy.empty_like(xlocs)
            mb = numpy.empty_like(xlocs)
            mg = numpy.empty_like(xlocs)
            ma.fill(malpha)
            mb.fill(mbeta)
            mg.fill(mgamma)
            mabg = numpy.hstack((ma, numpy.nan, mb, numpy.nan, mg, numpy.nan))
            lines += axes.plot(xlocs3, mabg, linewidth=0.5, color='0.50',
                               **mean_kw)

        # Add ticklabels and rotate
        add_ticks(axes.xaxis, xlocs, keys)
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
        because ellipse tilts are degenerate modulo pi radians.

        The ellipse parameters can be added to an existing plot via the
        optional 'axes' argument.

        :axes: Axes instance to add the ellipse parameters to. If None
               (default), the current Axes instance is used if any, or a new
               one created.
        :keys: sequence of cell keys to select cells to plot ellipse parameters
               for. If None (default), all cells are included.
        :marker: a valid matplotlib marker specification. Defaults to 'o'.
        :mean: select the kind of mean of ellipse parameters to add to the
               plot. Possible values:
            :direct: the arithmetic means of the eccentricity and tilt is used
                     for the mean point
            :euclidean: the mean cartesian coordinates of the ellipse parameter
                        points in the plane is used for the mean point
            :None: no mean point is plotted
               By default, a gray (color == 0.5) marker of type 'o' is used,
               but this can be overridden using the parameter mean_kw.
        :mean_kw: dict of keyword arguments to pass to the axes.plot() method
                  used to plot the mean ellipse. Default: None (empty dict)
        :kwargs: additional keyword arguments passed on to the axes.plot()
                 method. Note in particular the keywords 'markersize', 'color'
                 and 'label'.
        :returns: a list of the plotted Line2D instances

        """
        if axes is None:
            axes = pyplot.gca(projection='polar')

        __, cells = self.lookup(keys)

        tilt2, ecc = [], []
        for cell in cells:
            tilt2.append(2 * cell.ellipse().tilt)
            ecc.append(cell.ellipse().ecc)

        lines = axes.plot(tilt2, ecc, linestyle='None', marker='o', **kwargs)

        if mean is not None:
            if mean_kw is None:
                mean_kw = {}

            mtilt, mecc = self.mean_ellpars(mode=mean, keys=keys)
            lines += axes.plot(2.0 * mtilt, mecc, linestyle='None', marker='o',
                               color='0.50', **mean_kw)

        axes.set_ylim((0.0, 1.0))

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
        :threshold: peak threshold for the IdealGridCell instance constructed
                    by the module.
        :kwargs: passed through to the CellCollection constructor

        """
        CellCollection.__init__(self, cells, **kwargs)

        # Compute a large BinnedSet2D for the ideal cell
        bset = self.values()[0].firing_rate.bset
        new_bset = bset
        for __ in range(3):
            new_bset = new_bset.correlate(bset, mode='full')

        # Construct the ideal cell and save it to the instance
        self.idealcell = IdealGridCell(self.mean_peaks(),
                                       self.mean_firing_field(), new_bset,
                                       threshold=threshold)

    @memoize_method
    def phases(self, keys=None):
        """
        Compute the grid phase of cells in the module.

        The phase is defined with respect to the IdealGridCell instance
        belonging to the module (self.idealcell).

        :keys: sequence of cell keys to select cells to compute the phase for.
               If None, all cells are included.
        :returns: dict containing the phases

        """
        keys, cells = self.lookup(keys)
        phases = {key: cell.phase(self.idealcell)
                  for (key, cell) in zip(keys, cells)}

        return phases

    @memoize_method
    def phase_pattern(self, keys=None):
        """
        Create a PointPattern instance of the phases of the cells in the module

        :keys: sequence of cell keys to select cells for which to include the
               phase in the PointPattern. If None, all cells are included.
        :returns: PointPattern instance

        """
        phases = self.phases(keys=keys).values()
        peak_pattern = numpy.vstack(((0.0, 0.0), self.idealcell.peaks()))
        window = spatial.Voronoi(peak_pattern).vertices
        angles = numpy.arctan2(window[:, 1], window[:, 0])
        sort_ind = numpy.argsort(angles)
        window = window[sort_ind]
        return PointPattern(phases, window)

    @memoize_method
    def simulate_phase_patterns(self, process='binomial', nsims=1000,
                                keys=None):
        """
        Simulate a number of point processes in the same window, and of the
        same intensity, as the phase pattern for this module

        :process: string specifying the process to simulate. Possible values:
                  'binomial', 'poisson'
        :nsims: the number of patterns to generate from the process
        :keys: sequence of cell keys to select cells whose phases make up the
               PointPattern instance that underlies the simulation. If None,
               all cells are included.
        :returns: a PointPatternCollection instance containing the simulated
                  processes

        """
        phase_pattern = self.phase_pattern(keys=keys)
        window = phase_pattern.window
        intensity = phase_pattern.intensity(mode='standard')
        return PointPatternCollection.from_simulation(window, intensity,
                                                      process=process,
                                                      nsims=nsims)

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

    def plot_phase_k(self, axes=None, keys=None, edge_correction='periodic',
                     **kwargs):
        """
        Plot the empirical K-function for the grid phase point pattern

        The K-function can be added to an existing plot via the optional 'axes'
        argument.

        :axes: Axes instance to add the K-function to. Passed through to the
               PointPattern.plot_kfunction() method.
        :keys: sequence of cell keys to select cells to include in the
               phase pattern. If None, all cells are included.
        :edge_correction: flag to select the edge_correction to use in the
                          K-function. See the documentation for
                          PointPattern.kfunction() for details.
        :kwargs: additional keyword arguments passed on to the
                 PointPattern.plot_kfunction() method. Note in particular the
                 keywords 'csr' and 'csr_kw'.
        :returns: a list of the plotted objects, as returned from
                  PointPattern.plot_kfunction()

        """
        phase_pattern = self.phase_pattern(keys=keys)
        return phase_pattern.plot_kfunction(axes=axes,
                                            edge_correction=edge_correction,
                                            **kwargs)

    def plot_phase_l(self, axes=None, keys=None, edge_correction='periodic',
                     **kwargs):
        """
        Plot the empirical L-function for the grid phase point pattern

        The L-function can be added to an existing plot via the optional 'axes'
        argument.

        :axes: Axes instance to add the L-function to. Passed through to the
               PointPattern.plot_lfunction() method.
        :keys: sequence of cell keys to select cells to include in the
               phase pattern. If None, all cells are included.
        :edge_correction: flag to select the edge_correction to use in the
                          L-function. See the documentation for
                          PointPattern.lfunction() for details.
        :kwargs: additional keyword arguments passed on to the
                 PointPattern.plot_lfunction() method. Note in particular the
                 keywords 'csr' and 'csr_kw'.
        :returns: a list of the plotted objects, as returned from
                  PointPattern.plot_lfunction()

        """
        phase_pattern = self.phase_pattern(keys=keys)
        return phase_pattern.plot_lfunction(axes=axes,
                                            edge_correction=edge_correction,
                                            **kwargs)

    def plot_phase_kenvelope(self, axes=None, process='binomial', nsims=1000,
                             keys=None, edge_correction='periodic', low=0.025,
                             high=0.975, **kwargs):
        """
        Plot an envelope of empirical K-function values

        The envelope is based on simulated patterns using
        Module.simulate_phase_patterns().

        The envelope can be added to an existing plot via the optional 'axes'
        argument.

        :axes: Axes instance to add the envelope to. If None (default), the
               current Axes instance is used if any, or a new one created.
        :process: string specifying the process to simulate. Possible values:
                  'binomial', 'poisson'
        :nsims: the number of patterns to generate from the process
        :keys: sequence of cell keys to select cells whose phases make up the
               PointPattern instance that underlies the simulation. If None,
               all cells are included.
        :edge_correction: flag to select the handling of edges. See the
                          documentation for PointPattern.kfunction() for
                          details.
        :low: percentile defining the lower edge of the envelope
        :high: percentile defining the higher edge of the envelope
        :kwargs: additional keyword arguments passed on to the
                 PointPatternCollection.plot_kenvelope() method. Note in
                 particular the keywords 'alpha', 'edgecolor', 'facecolor' and
                 'label'.
        :returns: the PolyCollection instance filling the envelope.

        """
        sims = self.simulate_phase_patterns(process=process, nsims=nsims,
                                            keys=keys)
        return sims.plot_kenvelope(axes=axes, edge_correction=edge_correction,
                                   low=low, high=high, **kwargs)

    def plot_phase_kmean(self, axes=None, process='binomial', nsims=1000,
                         keys=None, edge_correction='periodic', **kwargs):
        """
        Plot the mean of empirical K-function values

        The mean is based on simulated patterns using
        Module.simulate_phase_patterns().

        The mean can be added to an existing plot via the optional 'axes'
        argument.

        :axes: Axes instance to add the mean to. If None (default), the current
               Axes instance is used if any, or a new one created.
        :process: string specifying the process to simulate. Possible values:
                  'binomial', 'poisson'
        :nsims: the number of patterns to generate from the process
        :keys: sequence of cell keys to select cells whose phases make up the
               PointPattern instance that underlies the simulation. If None,
               all cells are included.
        :edge_correction: flag to select the handling of edges. See the
                          documentation for PointPattern.kfunction() for
                          details.
        :kwargs: additional keyword arguments passed on to the
                 PointPatternCollection.plot_kmean() method. Note in particular
                 the keywords 'linewidth', 'linestyle', 'color', and 'label'.
        :returns: list containing the Line2D of the plotted mean.

        """
        sims = self.simulate_phase_patterns(process=process, nsims=nsims,
                                            keys=keys)
        return sims.plot_kmean(axes=axes, edge_correction=edge_correction,
                               **kwargs)

    def plot_phase_lenvelope(self, axes=None, process='binomial', nsims=1000,
                             keys=None, edge_correction='periodic', low=0.025,
                             high=0.975, **kwargs):
        """
        Plot an envelope of empirical L-function values

        The envelope is based on simulated patterns using
        Module.simulate_phase_patterns().

        The envelope can be added to an existing plot via the optional 'axes'
        argument.

        :axes: Axes instance to add the envelope to. If None (default), the
               current Axes instance is used if any, or a new one created.
        :process: string specifying the process to simulate. Possible values:
                  'binomial', 'poisson'
        :nsims: the number of patterns to generate from the process
        :keys: sequence of cell keys to select cells whose phases make up the
               PointPattern instance that underlies the simulation. If None,
               all cells are included.
        :edge_correction: flag to select the handling of edges. See the
                          documentation for PointPattern.kfunction() for
                          details.
        :low: percentile defining the lower edge of the envelope
        :high: percentile defining the higher edge of the envelope
        :kwargs: additional keyword arguments passed on to the
                 PointPatternCollection.plot_lenvelope() method. Note in
                 particular the keywords 'alpha', 'edgecolor', 'facecolor' and
                 'label'.
        :returns: the PolyCollection instance filling the envelope.

        """
        sims = self.simulate_phase_patterns(process=process, nsims=nsims,
                                            keys=keys)
        return sims.plot_lenvelope(axes=axes, edge_correction=edge_correction,
                                   low=low, high=high, **kwargs)

    def plot_phase_lmean(self, axes=None, process='binomial', nsims=1000,
                         keys=None, edge_correction='periodic', **kwargs):
        """
        Plot the mean of empirical L-function values

        The mean is based on simulated patterns using
        Module.simulate_phase_patterns().

        The mean can be added to an existing plot via the optional 'axes'
        argument.

        :axes: Axes instance to add the mean to. If None (default), the current
               Axes instance is used if any, or a new one created.
        :process: string specifying the process to simulate. Possible values:
                  'binomial', 'poisson'
        :nsims: the number of patterns to generate from the process
        :keys: sequence of cell keys to select cells whose phases make up the
               PointPattern instance that underlies the simulation. If None,
               all cells are included.
        :edge_correction: flag to select the handling of edges. See the
                          documentation for PointPattern.kfunction() for
                          details.
        :kwargs: additional keyword arguments passed on to the
                 PointPatternCollection.plot_lmean() method. Note in particular
                 the keywords 'linewidth', 'linestyle', 'color', and 'label'.
        :returns: list containing the Line2D of the plotted mean.

        """
        sims = self.simulate_phase_patterns(process=process, nsims=nsims,
                                            keys=keys)
        return sims.plot_lmean(axes=axes, edge_correction=edge_correction,
                               **kwargs)
