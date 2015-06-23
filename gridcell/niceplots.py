#!/usr/bin/env python

"""File: niceplots.py
Module defining convenience functions for easily creating nicely formatted
plots of gridcell stuff

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
import seaborn
from matplotlib import pyplot, patches, ticker


def pos_and_spikes(cell, alpha_path=0.5, alpha_spikes=0.25, length_unit='cm',
                   palette=None):
    """
    Convenience function to create nice position-and-spike plots

    Parameters
    ----------
    cell : Cell
        Cell to plot position and spikes for.
    alpha_path : scalar in [0.0, 1.0], optional
        The opacity of the path.
    alpha_spikes : scalar in [0.0, 1.0], optional
        The opacity of the spike markers.
    length_unit : string, optional
        The length unit to add to the axis labels.
    palette : sequence, optional
        Color palette to use for the position and spikes: the first color is
        used for the position, the second for the spikes.

    Returns
    -------
    list
        List of artists added to the plot.

    """
    if palette is None:
        artists = cell.pos.plot_path(alpha=alpha_path)
        palette = (None, None)
    else:
        artists = cell.pos.plot_path(alpha=alpha_path, color=palette[0])

    range_ = cell.range_
    axes = artists[0].get_axes()
    artists += cell.plot_spikes(axes=axes, alpha=alpha_spikes,
                                color=palette[1])
    axes.set(xlim=range_[0], ylim=range_[1],
             xticks=range_[0], yticks=range_[1])
    axes.set(xlabel=r"$x \ / \ \mathrm{{{}}}$".format(length_unit),
             ylabel=r"$y \ / \ \mathrm{{{}}}$".format(length_unit))

    return artists


def firing_rate(cell, cmap='YlGnBu_r', length_unit='cm', rate_unit='Hz'):
    """
    Convenience function to plot nice firing rate maps

    Parameters
    ----------
    cell : BaseCell
        Cell to plot firing rate map for.
    cmap : Colormap or registered colormap name, optional
        Colormap to use for the plot.
    length_unit : string, optional
        The length unit to add to the axis labels.
    rate_unit : string, optional
        The frequency unit to add to the colorbar label.

    Returns
    -------
        The return signature is equivalent to that of
        `BaseCell.plot_firing_rate`.

    """
    axes, cbar = cell.plot_firing_rate(cmap=cmap, edgecolor='face')

    cbar.solids.set(edgecolor='face')
    cbar.set_label(r"$f \ / \ \mathrm{{{}}}$".format(rate_unit))
    cbar.locator = ticker.MaxNLocator(nbins=2)
    cbar.update_ticks()

    range_ = cell.range_
    axes.set(xticks=range_[0], yticks=range_[1])
    axes.set(xlabel=r"$x \ / \ \mathrm{{{}}}$".format(length_unit),
             ylabel=r"$y \ / \ \mathrm{{{}}}$".format(length_unit))
    return axes, cbar


def template_firing_rate(module, cmap='YlGnBu_r', length_unit='cm', box=True,
                         window=False, palette=None):
    """
    Convenience function to plot nice template cell firing rate maps

    Parameters
    ----------
    module : Module
        Module for which to plot the template cell firing rate.
    cmap : Colormap or registered colormap name, optional
        Colormap to use for the plot.
    length_unit : string, optional
        The length unit to add to the axis labels.
    box : bool, optional
        If True, a box representning the environment where the true recordings
        were done is added to the plot.
    window : bool, optional
        If True, a polygon representning the window of possible phases is added
        to the plot.
    palette : sequence, optional
        Color palette to use for the box and window: the first color is used
        for the box, the second for the window.

    Returns
    -------
        The return signature is equivalent to that of
        `BaseCell.plot_firing_rate`.

    """
    cell = module.template
    axes, cbar = cell.plot_firing_rate(cmap=cmap, edgecolor='face')

    cbar.solids.set(edgecolor='face')
    cbar.set_label(r"$f \ / \ f_\mathrm{{max}}$")
    cbar.locator = ticker.MaxNLocator(nbins=2)
    cbar.update_ticks()

    range_ = module.values()[0].range_
    axes.set(xticks=range_[0], yticks=range_[1])
    axes.set(xlabel=r"$x \ / \ \mathrm{{{}}}$".format(length_unit),
             ylabel=r"$y \ / \ \mathrm{{{}}}$".format(length_unit))

    if palette is None:
        palette = (None, None)

    if box:
        xy = (range_[0][0], range_[1][0])
        width = range_[0][1] - range_[0][0]
        height = range_[1][1] - range_[1][0]
        rect = patches.Rectangle(xy=xy, width=width, height=height,
                                 fill=False, color=palette[0],
                                 linewidth=2.0)
        axes.add_patch(rect)
    if window:
        pattern = module.phase_pattern()
        windowpatch = pattern.window.patch(fill=False, color=palette[1],
                                           linewidth=2.0)
        axes.add_patch(windowpatch)

    return axes, cbar


def stacked_firing_rate(sfiring_rate, cmap='YlGnBu_r', length_unit='cm'):
    """
    Convenience function to plot nice stacked cell firing rate maps

    Parameters
    ----------
    sfiring_rate : IntensityMap2D
        Stacked firing rate to plot
    cmap : Colormap or registered colormap name, optional
        Colormap to use for the plot.
    length_unit : string, optional
        The length unit to add to the axis labels.

    Returns
    -------
        The return signature is equivalent to that of
        `IntensityMap2D.plot`.

    """
    vmin = min(0.0, sfiring_rate.min())
    axes, cbar = sfiring_rate.plot(cmap=cmap, vmin=vmin, edgecolor='face')

    cbar.solids.set(edgecolor='face')
    cbar.set_ticks([0.0])

    range_ = sfiring_rate.range_
    axes.set(xticks=range_[0], yticks=range_[1])
    axes.set(xlabel=r"$x \ / \ \mathrm{{{}}}$".format(length_unit),
             ylabel=r"$y \ / \ \mathrm{{{}}}$".format(length_unit))

    return axes, cbar


def acorr(cell, cmap='coolwarm', length_unit='cm', peaks=True, ellipse=True,
          palette=None):
    """
    Convenience function to plot nice firing rate autocorrelograms

    Parameters
    ----------
    cell : BaseCell
        Cell to plot autocorrelogram for.
    cmap : Colormap or registered colormap name, optional
        Colormap to use for the plot.
    length_unit : string, optional
        The length unit to add to the axis labels.
    peaks : bool, optional
        If True, the inner ring of peaks are plotted with markers.
    ellipse : bool, optional
        If True, the ellipse through the inner ring of peaks is drawn.
    palette : sequence, optional
        Color palette to use for peaks and ellipse: the first color is used for
        peak markers, the second for the ellipse.

    Returns
    -------
        The return signature is equivalent to that of
        `BaseCell.plot_autocorrelogram`.

    """
    axes, cbar = cell.plot_autocorrelogram(cmap=cmap, edgecolor='face',
                                           cbar_kw={'ticks': [-1, 0, 1]})

    if palette is None:
        palette = (None, None)

    if peaks:
        cell.plot_peaks(axes=axes, color=palette[0])
    if ellipse:
        cell.plot_ellipse(axes=axes, color=palette[1])

    cbar.solids.set(edgecolor='face')
    cbar.set_label(r"$\rho$")

    range_ = cell.range_
    axes.set(xticks=range_[0], yticks=range_[1])
    axes.set(xlabel=r"$x \ / \ \mathrm{{{}}}$".format(length_unit),
             ylabel=r"$y \ / \ \mathrm{{{}}}$".format(length_unit))
    return axes, cbar


def ccorr(cell1, cell2, cmap='coolwarm', length_unit='cm', cpeak=True):
    """
    Convenience function to plot nice firing rate cross-correlograms

    Parameters
    ----------
    cell1, cell2 : BaseCell
        Cell to plot firing rate map for.
    cmap : Colormap or registered colormap name, optional
        Colormap to use for the plot.
    length_unit : string, optional
        The length unit to add to the axis labels.
    cpeak : bool, optional
        If True, the peak closest to the center is plotted with a marker.

    Returns
    -------
        The return signature is equivalent to that of
        `BaseCell.plot_correlogram`.

    """
    axes, cbar = cell1.plot_correlogram(cell2, cmap=cmap, cpeak=cpeak,
                                        edgecolor='face',
                                        cbar_kw={'ticks': [-1, 0, 1]})

    cbar.solids.set(edgecolor='face')
    cbar.set_label(r"$\rho$")

    range_ = cell1.range_
    axes.set(xticks=range_[0], yticks=range_[1])
    axes.set(xlabel=r"$x \ / \ \mathrm{{{}}}$".format(length_unit),
             ylabel=r"$y \ / \ \mathrm{{{}}}$".format(length_unit))
    return axes, cbar


def phase_pattern(module, periodic=True, length_unit='cm', palette=None):
    """
    Convenience function to plot nice phase patterns

    Parameters
    ----------
    module : Module
        Module to plot phase pattern from.
    length_unit : string, optional
        The length unit to add to the axis labels.
    periodic : bool, optional
        If True, the periodic extension is included in the plot.
    palette : sequence, optional
        Color palette to use for the artists: the first color is used for the
        window, the second for the points in the pattern, and the third for the
        periodic extension (if any).

    Returns
    -------
        The return signature is equivalent to that of
        `Module.plot_phases`.

    """
    if palette is None:
        palette = (None,) * 3

    artists = module.plot_phases(window=True, periodic=periodic,
                                 window_kw={'color': palette[0]},
                                 periodic_kw={'color': palette[2]},
                                 color=palette[1])

    range_ = module.values()[0].range_
    axes = artists[0].get_axes()
    axes.set(xticks=range_[0], yticks=range_[1])
    axes.set(xlabel=r"$x \ / \ \mathrm{{{}}}$".format(length_unit),
             ylabel=r"$y \ / \ \mathrm{{{}}}$".format(length_unit))
    return artists


def kfunction(pattern, nsims=1000, length_unit='cm', csr=True, interval=False,
              palette=None):
    """
    Convenience function to plot nice K functions

    Parameters
    ----------
    pattern : PointPattern
        Point pattern to plot a K function estimator from.
    nsims : integer, optional
        Number of simulated patterns to compute mean and envelope from. If 0,
        no mean or envelope is plotted.
    length_unit : string, optional
        The length unit to add to the axis labels.
    csr : bool, optional
        If True, the theoretical CSR line is added to the plot.
    interval : bool, optional
        If True, the standard interval for evaluating the L test statistic is
        marked on the plot.
    palette : sequence, optional
        Color palette to use for the artists: the first color is used for the
        estimator, the second for CSR (if any), the third for the envelope (if
        any), the fourth for the mean (if any), and the fifth for the interval
        markers (if any).


    Returns
    -------
    list
        List of artists added to the plot.

    """
    if palette is None:
        palette = (None,) * 5

    axes = pyplot.gca()
    if nsims > 0:
        sims = pattern.simulate(nsims=nsims)
        artists = [sims.plot_kenvelope(axes=axes, color=palette[2])]
        artists += sims.plot_kmean(axes=axes, label='Mean', color=palette[3])
    else:
        artists = []
    artists += pattern.plot_kfunction(axes=axes, csr=csr, color=palette[0],
                                      csr_kw={'label': 'CSR',
                                              'color': palette[1]},
                                      label='Estimator', zorder=10)
    if interval:
        interval = pattern.lstatistic_interval()
        axes.axvline(interval[0], color=palette[4])
        axes.axvline(interval[1], color=palette[4])

    axes.set(xlabel=r"$r \ / \ \mathrm{{{}}}$".format(length_unit),
             ylabel=r"$K(r) \ / \ \mathrm{{{}}}^2$".format(length_unit))

    return artists


def lfunction(pattern, nsims=1000, length_unit='cm', csr=True, interval=False,
              palette=None):
    """
    Convenience function to plot nice L functions

    Parameters
    ----------
    pattern : PointPattern
        Point pattern to plot an L function estimator from.
    nsims : integer, optional
        Number of simulated patterns to compute mean and envelope from. If 0,
        no mean or envelope is plotted.
    length_unit : string, optional
        The length unit to add to the axis labels.
    csr : bool, optional
        If True, the theoretical CSR line is added to the plot.
    interval : bool, optional
        If True, the standard interval for evaluating the L test statistic is
        marked on the plot.
    palette : sequence, optional
        Color palette to use for the artists: the first color is used for the
        estimator, the second for CSR (if any), the third for the envelope (if
        any), the fourth for the mean (if any), and the fifth for the interval
        markers (if any).

    Returns
    -------
    list
        List of artists added to the plot.

    """
    axes = pyplot.gca()
    if nsims > 0:
        sims = pattern.simulate(nsims=nsims)
        artists = [sims.plot_lenvelope(axes=axes, color=palette[2])]
        artists += sims.plot_lmean(axes=axes, label='Mean', color=palette[3])
    else:
        artists = []
    artists += pattern.plot_lfunction(axes=axes, csr=csr, color=palette[0],
                                      csr_kw={'label': 'CSR',
                                              'color': palette[1]},
                                      label='Estimator', zorder=10)
    if interval:
        interval = pattern.lstatistic_interval()
        axes.axvline(interval[0], color=palette[4])
        axes.axvline(interval[1], color=palette[4])

    axes.set(xlabel=r"$r \ / \ \mathrm{{{}}}$".format(length_unit),
             ylabel=r"$L(r) \ / \ \mathrm{{{}}}$".format(length_unit))

    return artists


def pair_corr(pattern, nsims=1000, length_unit='cm', csr=True, interval=False,
              palette=None):
    """
    Convenience function to plot nice pair correlation functions

    Parameters
    ----------
    pattern : PointPattern
        Point pattern to plot a pair correlation function estimator from.
    nsims : integer, optional
        Number of simulated patterns to compute mean and envelope from. If 0,
        no mean or envelope is plotted.
    length_unit : string, optional
        The length unit to add to the axis labels.
    csr : bool, optional
        If True, the theoretical CSR line is added to the plot.
    interval : bool, optional
        If True, the standard interval for evaluating the L test statistic is
        marked on the plot.
    palette : sequence, optional
        Color palette to use for the artists: the first color is used for the
        estimator, the second for CSR (if any), the third for the envelope (if
        any), the fourth for the mean (if any), and the fifth for the interval
        markers (if any).

    Returns
    -------
    list
        List of artists added to the plot.

    """
    axes = pyplot.gca()
    if nsims > 0:
        sims = pattern.simulate(nsims=nsims)
        artists = [sims.plot_pair_corr_envelope(axes=axes, color=palette[2])]
        artists += sims.plot_pair_corr_mean(axes=axes, label='Mean',
                                            color=palette[3])
    else:
        artists = []
    artists += pattern.plot_pair_corr_function(axes=axes, csr=csr,
                                               color=palette[0],
                                               csr_kw={'label': 'CSR',
                                                       'color': palette[1]},
                                               label='Estimator', zorder=10)
    if interval:
        interval = pattern.lstatistic_interval()
        axes.axvline(interval[0], color=palette[4])
        axes.axvline(interval[1], color=palette[4])

    axes.set(xlabel=r"$r \ / \ \mathrm{{{}}}$".format(length_unit),
             ylabel=r"$g(r)$")

    return artists


def ldistplot(pattern, nsims=1000, length_unit='cm', palette=None):
    """
    Convenience function to plot nice L statistic distribution plots

    Parameters
    ----------
    pattern : PointPattern
        Point pattern to plot a the L statistic estimator from.
    nsims : integer, optional
        Number of simulated patterns to compute distribution from.
    length_unit : string, optional
        The length unit to add to the axis labels.
    palette : sequence, optional
        Color palette to use: the first color is used for histogram/KDE, the
        second for the pattern statistic.

    Returns
    -------
        The return signature is equivalent to that of
        `seaborn.distplot`.

    """
    sims = pattern.simulate(nsims=nsims)

    if palette is None:
        palette = (None, None)

    axes = seaborn.distplot(sims.lstatistics(), color=palette[0],
                            hist_kws={'histtype': 'stepfilled',
                                      'label': 'Simulations'})
    axes.axvline(pattern.lstatistic(), color=palette[1], label='Pattern')
    axes.set(xlabel=r"$\tau \ / \ \mathrm{{{}}}$".format(length_unit),
             ylabel=r"$f(\tau) \ / \ \mathrm{{{}}}^{{-1}}$"
             .format(length_unit))

    return axes
