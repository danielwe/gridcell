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
from .utils import distplot
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
        palette = (None, None)

    pos = cell.position
    if palette[0] is None:
        artists = pos.plot_path(alpha=alpha_path)
    else:
        artists = pos.plot_path(alpha=alpha_path, color=palette[0])

    axes = artists[0].get_axes()
    if palette[1] is None:
        artists += cell.plot_spikes(axes=axes, alpha=alpha_spikes)
    else:
        artists += cell.plot_spikes(axes=axes, alpha=alpha_spikes,
                                    color=palette[1])

    range_ = cell.params['range_']
    axes.set(xlim=range_[0], ylim=range_[1],
             xticks=range_[0], yticks=range_[1])
    axes.set(xlabel=r"$x \ / \ \mathrm{{{}}}$".format(length_unit),
             ylabel=r"$y \ / \ \mathrm{{{}}}$".format(length_unit))

    return artists


def ratemap(cell, cmap='YlGnBu_r', length_unit='cm', rate_unit='Hz'):
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
    See `BaseCell.plot_ratemap`.

    """
    axes, cbar = cell.plot_ratemap(cmap=cmap,
                                   edgecolor='face',
                                   )

    cbar.solids.set(edgecolor='face')
    cbar.set_label(r"$f \ / \ \mathrm{{{}}}$".format(rate_unit))
    cbar.locator = ticker.MaxNLocator(nbins=2)
    cbar.update_ticks()

    range_ = cell.params['range_']
    axes.set(xticks=range_[0], yticks=range_[1])
    axes.set(xlabel=r"$x \ / \ \mathrm{{{}}}$".format(length_unit),
             ylabel=r"$y \ / \ \mathrm{{{}}}$".format(length_unit))
    return axes, cbar


def template_ratemap(module, cmap='YlGnBu_r', length_unit='cm', box=True,
                     window_type=None, palette=None):
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
    window_type : None or string, optional
        If not None, a polygon representning the window of possible phases is
        added to the plot. The value of this variable is passed to
        `Module.window` as the keyword `window_type`. See this method for
        possible values.
    palette : sequence, optional
        Color palette to use for the box and window: the first color is used
        for the box, the second for the window.

    Returns
    -------
    See `TemplateGridCell.plot_ratemap`

    """
    cell = module.template()
    axes, cbar = cell.plot_ratemap(cmap=cmap,
                                   edgecolor='face'
                                   )

    cbar.solids.set(edgecolor='face')
    cbar.set_label(r"$f \ / \ f_\mathrm{{max}}$")
    cbar.locator = ticker.MaxNLocator(nbins=2)
    cbar.update_ticks()

    range_ = next(iter(module)).params['range_']
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
    if window_type is not None:
        windowpatch = module.window(window_type=window_type).patch(
            fill=False, color=palette[1], linewidth=2.0)
        axes.add_patch(windowpatch)

    return axes, cbar


def imap(imap, vmin=None, vmax=None, cmap='YlGnBu_r', length_unit='cm'):
    """
    Convenience base function to plot any IntensityMap2D instance

    Parameters
    ----------
    ratemap : IntensityMap2D
        Stacked firing rate to plot
    cmap : Colormap or registered colormap name, optional
        Colormap to use for the plot.
    length_unit : string, optional
        The length unit to add to the axis labels.

    Returns
    -------
    See `IntensityMap2D.plot`.

    """
    axes, cbar = imap.plot(cmap=cmap, vmin=vmin, vmax=vmax, edgecolor='face')

    cbar.solids.set(edgecolor='face')

    range_ = imap.range_
    axes.set(xticks=range_[0], yticks=range_[1])
    axes.set(xlabel=r"$x \ / \ \mathrm{{{}}}$".format(length_unit),
             ylabel=r"$y \ / \ \mathrm{{{}}}$".format(length_unit))

    return axes, cbar


def acorr(cell, cmap='coolwarm', length_unit='cm', threshold=False,
          grid_peaks=False, grid_ellipse=False, palette=None):
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
    threshold : bool, optional
        See `BaseCell.plot_acorr`.
    grid_peaks : bool, optional
        If True, the inner ring of peaks are plotted with markers.
    grid_ellipse : bool, optional
        If True, the ellipse through the inner ring of peaks is drawn.
    palette : sequence, optional
        Color palette to use for grid peaks and ellipse: the first color is
        used for peak markers, the second for the ellipse.

    Returns
    -------
    See `BaseCell.plot_acorr`.

    """
    axes, cbar = cell.plot_acorr(cmap=cmap,
                                 edgecolor='face',
                                 threshold=threshold,
                                 cbar_kw={'ticks': [-1, 0, 1]})

    if palette is None:
        palette = (None, None)

    if grid_peaks:
        cell.plot_grid_peaks(axes=axes, markersize=8, color=palette[0])
    if grid_ellipse:
        cell.plot_grid_ellipse(axes=axes, color=palette[1])

    cbar.solids.set(edgecolor='face')
    cbar.set_label(r"$r$")

    range_ = cell.params['range_']
    axes.set(xticks=range_[0], yticks=range_[1])
    axes.set(xlabel=r"$x \ / \ \mathrm{{{}}}$".format(length_unit),
             ylabel=r"$y \ / \ \mathrm{{{}}}$".format(length_unit))
    return axes, cbar


def corr(cell1, cell2, cmap='coolwarm', length_unit='cm', center_peak=False):
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
    center_peak : bool, optional
        If True, the peak closest to the center is marked.

    Returns
    -------
    See `BaseCell.plot_corr`.

    """
    axes, cbar = cell1.plot_corr(cell2, cmap=cmap, center_peak=center_peak,
                                 edgecolor='face',
                                 cbar_kw={'ticks': [-1, 0, 1]})

    cbar.solids.set(edgecolor='face')
    cbar.set_label(r"$\rho$")

    range_ = cell1.params['range_']
    axes.set(xticks=range_[0], yticks=range_[1])
    axes.set(xlabel=r"$x \ / \ \mathrm{{{}}}$".format(length_unit),
             ylabel=r"$y \ / \ \mathrm{{{}}}$".format(length_unit))
    return axes, cbar


def phase_pattern(module, window_type='voronoi', project_phases=False,
                  periodic_levels=4, length_unit='cm', palette=None):
    """
    Convenience function to plot nice phase patterns

    Parameters
    ----------
    module : Module
        Module to plot phase pattern from.
    window_type : string, optional
        The type of window to use. See `Module.window_vertices` for possible
        values.
    project_phases : bool, optional
        If True, the phases are projected to a regular hexagonal grid.
    periodic_levels : bool, optional
        Add this many levels of periodic extension of the phase pattern to the
        plot.
    length_unit : string, optional
        The length unit to add to the axis labels. This is ignored if
        project_phases is True.
    palette : sequence, optional
        Color palette to use for the artists: the first color is used for the
        window, the second for the points in the pattern, and the third for the
        periodic extension (if any).

    Returns
    -------
    See `Module.plot_phases`.

    """
    if palette is None:
        palette = (None,) * 3

    phase_pattern = module.phase_pattern(window_type=window_type,
                                         project_phases=project_phases)
    artists = phase_pattern.plot_pattern(window=True,
                                         periodic_levels=periodic_levels,
                                         window_kw={'color': palette[0]},
                                         periodic_kw={'color': palette[2]},
                                         color=palette[1])

    axes = artists[0].get_axes()
    if project_phases:
        range_ = ((-1.0, 1.0), (-1.0, 1.0))
        axes.set(xlabel=r"$\delta_x$",
                 ylabel=r"$\delta_y$")
    else:
        range_ = next(iter(module)).params['range_']
        axes.set(xlabel=r"$\delta_x \ / \ \mathrm{{{}}}$".format(length_unit),
                 ylabel=r"$\delta_y \ / \ \mathrm{{{}}}$".format(length_unit))
    axes.set(xticks=range_[0], yticks=range_[1])
    return artists


def pairwise_phase_pattern(module, window_type='voronoi', from_absolute=True,
                           project_phases=False, full_window=False,
                           sign='regular', length_unit='cm',
                           palette=None):
    """
    Convenience function to plot nice pairwise phase patterns

    Parameters
    ----------
    module : Module
        Module to plot phase pattern from.
    window_type : string, optional
        The type of window to use. See `Module.window_vertices` for possible
        values.
    from_absolute : bool, optional
        If True, the pairwise phases are computed from the absolute phases. If
        False, they are computed directly from each pair of cells.
    project_phases : bool, optional
        If True, the phases are projected to a regular hexagonal grid.
    full_window : bool, optional
        If True, the full window used for absolute phases is also used for
        relative phases.
    sign : string, optional
        Flag to select how to resolve the sin ambiguity when `full_window ==
        True`.
    length_unit : string, optional
        The length unit to add to the axis labels. This is ignored if
        project_phases is True.
    palette : sequence, optional
        Color palette to use for the artists: the first color is used for the
        window, the second for the points in the pattern, and the third for the
        periodic extension (if any).

    Returns
    -------
    See `Module.plot_phases`.

    """
    if palette is None:
        palette = (None,) * 3

    phase_pattern = module.pairwise_phase_pattern(
        window_type=window_type,
        from_absolute=from_absolute,
        project_phases=project_phases,
        full_window=full_window,
        sign=sign)
    artists = phase_pattern.plot_pattern(window=True,
                                         window_kw={'color': palette[0]},
                                         periodic_kw={'color': palette[2]},
                                         color=palette[1])

    axes = artists[0].get_axes()
    if project_phases:
        range_ = ((-1.0, 1.0), (-1.0, 1.0))
        axes.set(xlabel=r"$\delta_x$",
                 ylabel=r"$\delta_y$")
    else:
        range_ = next(iter(module)).params['range_']
        axes.set(xlabel=r"$\delta_x \ / \ \mathrm{{{}}}$".format(length_unit),
                 ylabel=r"$\delta_y \ / \ \mathrm{{{}}}$".format(length_unit))
    axes.set(xticks=range_[0], yticks=range_[1])
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


def ldistplot(pattern, nsims=1000, weight_function=None, length_unit='cm',
              palette=None):
    """
    Convenience function to plot nice L statistic distribution plots

    Parameters
    ----------
    pattern : PointPattern
        Point pattern to plot a the L statistic estimator from.
    weight_function : callable
        Weight function to use for the L statistic estimator
    nsims : integer, optional
        Number of simulated patterns to compute distribution from.
    length_unit : string, optional
        The length unit to add to the axis labels.
    palette : sequence, optional
        Color palette to use: the first color is used for histogram/KDE, the
        second for the pattern statistic.

    Returns
    -------
    See `seaborn.distplot`.

    """
    sims = pattern.simulate(nsims=nsims)

    if palette is None:
        palette = (None, None)

    lstat = sims.lstatistics(weight_function=weight_function)

    axes = distplot(lstat, color=palette[0],
                    hist_kws={'histtype': 'stepfilled',
                              'label': 'Simulations'})
    axes.axvline(pattern.lstatistic(), color=palette[1], label='Pattern')
    axes.set(xlabel=r"$\tau \ / \ \mathrm{{{}}}$".format(length_unit),
             ylabel=r"$f(\tau) \ / \ \mathrm{{{}}}^{{-1}}$"
             .format(length_unit))

    return axes
