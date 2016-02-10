#!/usr/bin/env python

"""File: formats.py
Module containing convenience functions for formatting raw data in a way that
the gridcell module understands, and transforming positions from raw to
physical coordinates.

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
from collections import deque, Mapping
import numpy
from scipy import optimize, signal
from .utils import sensibly_divide
from .cells import Position


def filter_positions(t, x, y, window_time, cutoff_freq, window='hann'):
    """
    Filter a series of position samples

    The positions are interpreted by `gridcell.Position` to find missing values
    and splits between consecutive streaks of samples, before, they are filter
    using `filter_`.

    Parameters
    ----------
    t : array-like
        Time points of the samples in the signal.
    x, y : array-like
        Position samples. See `gridcell.Position` for details about how missing
        values and splits between consecutive streaks of regular samples are
        handled.
    window_time, cutoff_freq, window
        See `filter_`.

    Returns
    -------
    xf, yf : array-like, masked
        Filtered position samples. Missing values are masked.

    """
    pos = Position(t, x, y)
    data = pos.data
    tsplit, xsplit, ysplit = data['tsplit'], data['xsplit'], data['ysplit']
    xsplitf, ysplitf = [], []
    for tt, xx, yy in zip(tsplit, xsplit, ysplit):
        xsplitf.append(filter_(tt, xx, window_time, cutoff_freq, window))
        ysplitf.append(filter_(tt, yy, window_time, cutoff_freq, window))
    return numpy.ma.hstack(xsplitf), numpy.ma.hstack(ysplitf)


def filter_(t, x, window_time, cutoff_freq, window):
    """
    Filter a signal using a FIR low-pass filter

    Parameters
    ----------
    t : array-like
        Time points of the samples in the signal. Only used to extract the
        sampling frequency.
    x : array-like
        Signal to filter. If given as a masked array, the masked entries
        will be handled using normalized convolution. Boundary conditions
        are handled by treating samples beyond the boundary the same as
        masked samples.
    window_time : non-negative scalar
        Length of the filter kernel, given in the same unit as `t`. The
        actual kernel will be rounded to an odd number of samples.
    cutoff_freq : non-negative scalar
        Cutoff frequency of the filter. Must be between 0.0 and half the
        sampling frequency, and given in the reciprocal unit of `t`.
    window : string
        Windowing function to use when constructing the filter kernel. See
        `signal.firwin` for valid options.

    Returns
    -------
    xf : ndarray
        Filtered signal. Locations where only masked values contributed to
        the filtered value are masked.

    """
    if window_time < 0.0:
        raise ValueError("'window_time' must be a non-negative number")
    elif window_time == 0.0:
        return x
    fs = 1.0 / numpy.mean(numpy.diff(t))
    wl = 2 * int(0.5 * window_time * fs) + 1
    kernel = signal.firwin(wl, [cutoff_freq], pass_zero=True,
                           nyq=(0.5 * fs), window='hann')
    mask = numpy.ma.getmaskarray(x)
    xf = numpy.ma.filled(x, fill_value=0.0)
    xn = signal.convolve(xf, kernel, mode='same')
    xd = signal.convolve((~mask).astype(numpy.int_), kernel, mode='same')
    return sensibly_divide(xn, xd, masked=True)


def sessions(positions, spike_times, param_dicts=None, info_dicts=None):
    """
    Identify separate sessions in multiple position and spike time recordings

    Parameters
    ----------
    positions : sequence
        Sequence containing the position samples corresponding to each cell.
        Each element should be a sequence similar to `(t, x, y)` or `(t, x, y,
        x2, y2)`, where `x` and `y` are arrays containing the position samples,
        `t` is an array containing the corresponding sample times, and `x2`and
        `y2` are optional arrays containing and extra set of position samples.
        Both standards may be present simultaneously.
    spike_ts : sequence
        Sequence containing the spike times corresponding to each cell.
        Each element should be an array of spike times.
    param_dicts : sequence or None, optional
        Sequence containing a dict of parameters for each cell. If None, an
        empty dict will be assigned to each cell.
    info_dicts : sequence or None, optional
        Sequence containing an info dict for each cell. If None, an empty info
        dict will be assigned to each cell.

    Returns
    -------
    dict
        Dict containing the data from the sessions, formatted for compliance
        with `CellCollection.from_multiple_sessions`.

    """
    session_indices = _equal_items(positions)
    if param_dicts is None:
        param_dicts = [{} for _ in spike_times]
    if info_dicts is None:
        info_dicts = [{} for _ in spike_times]

    sessions = {'sessions': [], 'params': {}, 'info': {}}
    for ind in session_indices:
        pos = positions[ind[0]]
        spike_ts = [spike_times[i] for i in ind]
        st_indices = _equal_items(spike_ts)
        cells = [{'spike_t': spike_times[ind[sti[0]]],
                  'params': param_dicts[ind[sti[0]]],
                  'info': info_dicts[ind[sti[0]]]}
                 for sti in st_indices]
        if len(pos) > 3:
            t, x, y, x2, y2 = pos
            sessions['sessions'].append({'t': t, 'x': x, 'y': y, 'x2': x2,
                                         'y2': y2, 'cells': cells,
                                         'params': {}, 'info': {}})
        else:
            t, x, y = pos
            sessions['sessions'].append({'t': t, 'x': x, 'y': y,
                                         'cells': cells, 'params': {},
                                         'info': {}})

    return sessions


def transform_sessions(sessions, global_=False, **kwargs):
    """
    Apply scaling, rotation and translation to the positions from several
    sessions

    Parameters
    ----------
    sessions : dict
        Dict containing the data from the sessions, formatted for compliance
        with `CellCollection.from_multiple_sessions`.
    global_ : bool, optional
        If True, the transformation is applied to the union of all position
        samples from all sessions. If False, the transformation is applied
        separately to the position samples from each session.
    **kwargs
        Keyword arguments are passed to `transform`.

    Returns
    -------
    dict
        A dict similar to `sessions`, but with positions transformed.

    """
    new_sessions = dict(sessions)

    if global_:
        xarrs, yarrs, lengths = [], [], [], [], []
        for session in sessions['sessions']:
            xarrs.append(session['x'])
            yarrs.append(session['y'])
            lengths.append(session['t'].size)
        xall, yall = numpy.hstack(xarrs), numpy.hstack(yarrs)

        xall, yall, info = transform(xall, yall, **kwargs)

        new_sessions['sessions'] = []
        new_sessions['info'] = _update_info(sessions['info'], info)
        for (l, session) in zip(lengths, sessions['sessions']):
            x, y = xall[:l], yall[:l]
            xall, yall = xall[l:], yall[l:]

            new_session = dict(session)
            if 'x2' in session:
                if 'y2' not in session:
                    raise ValueError("'x2' and 'y2' must either both be given "
                                     "or both left out")
                x2, y2 = _transform(session['x2'], session['y2'], info)
                new_session.update(
                    x=x, y=y, x2=x2, y2=y2,
                    info=_update_info(new_session['info'], info))
            else:
                new_session.update(
                    x=x, y=y,
                    info=_update_info(new_session['info'], info))
            new_sessions['sessions'].append(new_session)
    else:
        new_sessions['sessions'] = [transform_session(session, **kwargs)
                                    for session in sessions['sessions']]

    return new_sessions


def transform_session(session, **kwargs):
    """
    Apply scaling, rotation and translation to the positions in a session

    This is a convenience wrapper around `transform` that can be applied to
    a formatted session dict.

    Parameters
    ----------
    session : dict
        A dict containing the data from a recording session, formatted for
        compliance with `CellCollection.from_session`.
    **kwargs
        Keyword arguments are passed to `transform`.

    Returns
    -------
    dict
        A dict similar to `session`, but with positions transformed.

    """
    x, y = session['x'], session['y']
    new_session = dict(session)
    if 'x2' in session:
        if 'y2' not in session:
            raise ValueError("'x2' and 'y2' must either both be given or both "
                             "left out")
        x2, y2 = session['x2'], session['y2']
        x, y, x2, y2, info = transform(x, y, x2=x2, y2=y2, **kwargs)
        new_session.update(x=x, y=y, x2=x2, y2=y2,
                           info=_update_info(new_session['info'], info))
    else:
        x, y, info = transform(x, y, **kwargs)
        new_session.update(x=x, y=y,
                           info=_update_info(new_session['info'], info))
    return new_session


def transform(x, y, x2=None, y2=None, range_=None, translate=False,
              rotate=False):
    """
    Scale, rotate and translate a set of coordinates

    Parameters
    ----------
    x, y : array-like
        Arrays containing the coordinates.
    x2, y2 : array-like, optional
        Arrays containing the coordinates of a second set of position
        samples. These arrays will be transformed, but will not affect the
        transform parameters.
    range : ((xmin, xmax), (ymin, ymax)), optional
        Range specification giving the x and y values of the edges within which
        to fit the transformed coordinates. The coordinates will be rescaled
        to fit, and to the extent possible fill, this range. The scaling is
        applied to the displacement of the coordinates from the centroid of
        this range, and not on the raw coordinates. If None, no scaling is
        performed.
    translate : bool, optional
        If True, the coordinates are translated such that the centroid of their
        smallest axis-aligned rectangular bounding box is placed at the
        centroid of the box defined by `range_`. This is done before scaling,
        and minimizes the amount of scaling needed. If `range_` is None, the
        centroid is translated to the origin of the coordinate system.
    rotate : bool, optional
        If True, the coordinates are rotated around the centroid of their
        smallest rectangular bounding box, such that this box is aligned with
        the coordinate axes after the rotation. This is done before translation
        and scaling.

    Returns
    -------
    x, y : ndarray
        Arrays containing the transformed coordinates.
    x2, y2 : ndarray
        Arrays containing the transformed coordinates of the second set of
        samples. Only returned if `x2` and `y2` are not `None`.
    info : dict
        Information about the transformation. Contains the following key-value
        pairs:

        ``'rotation_angle': scalar``
            The rotation angle in radians.
        ``'rotation_anchor': (xanchor, yanchor)``
            Anchor point for the rotation.
        ``'translation_vector': (xdisp, ydisp)``
            The x and y translation displacement. If `range_` is not None, this
            value is scaled along with the samples, such that it refers to the
            same units as `range_`.
        ``'scaling_factor': scalar``
            Scaling factor.
        ``'scaling_anchor': (xanchor, yanchor)``
            Anchor point for the scaling transformation.

    """
    info = {'rotation_angle': 0.0, 'rotation_anchor': (0.0, 0.0),
            'translation_vector': (0.0, 0.0),
            'scaling_factor': 1.0, 'scaling_anchor': (0.0, 0.0)}

    if range is None:
        xo, yo = 0.0, 0.0
    else:
        xo, yo = _midpoint(range_[0]), _midpoint(range_[1])

    if rotate:
        angle = -_tilt(x, y)
        x, y = _rotate(x, y, angle)
        # Find new (wrong) and old (correct) centroid
        new_xc, new_yc = _midpoint(x), _midpoint(y)
        old_xc, old_yc = _rotate(new_xc, new_yc, -angle)
        # Reset centroid
        x, y = x - new_xc + old_xc, y - new_yc + old_yc
    else:
        angle = 0.0
        old_xc, old_yc = _midpoint(x), _midpoint(y)
    info['rotation_angle'] = angle
    info['rotation_anchor'] = (old_xc, old_yc)

    if translate:
        xc, yc = _midpoint(x), _midpoint(y)
        xtrans, ytrans = xo - xc, yo - yc

        # Beware! Using += would modify the original arrays if rotate=False!
        x, y = x + xtrans, y + ytrans
    else:
        xtrans, ytrans = 0.0, 0.0
    info['translation_vector'] = (xtrans, ytrans)

    if range_ is not None:
        dxo, dyo = x - xo, y - yo

        # The following is only safe as long as we know that (xo, yo) is within
        # range_. Here, we know this due to the present definition of (xo, yo).
        # In other cases, scaling might not always succeed, and an error should
        # be raised.
        candidate_factors = numpy.array(
            [(range_[0][0] - xo) / numpy.nanmin(dxo),
             (range_[1][0] - yo) / numpy.nanmin(dyo),
             (range_[0][1] - xo) / numpy.nanmax(dxo),
             (range_[1][1] - yo) / numpy.nanmax(dyo)],
        )
        scaling_factor = numpy.amin(
            candidate_factors[candidate_factors >= 0.0])

        dxo *= scaling_factor
        dyo *= scaling_factor
        x, y = xo + dxo, yo + dyo
    else:
        scaling_factor = 1.0
    info['scaling_factor'] = scaling_factor
    info['scaling_anchor'] = (xo, yo)

    if x2 is not None:
        if y2 is None:
            raise ValueError("'x2' and 'y2' must either both be given or "
                             "both left out")
        x2, y2 = _transform(x2, y2, info)
        return x, y, x2, y2, info
    return x, y, info


def _transform(x, y, info):
    """
    Scale, rotate and translate a set of coordinates using preset
    parameters

    Parameters
    ----------
    x, y : array-like
        Arrays containing the coordinates.
    info : dict
        Information about the transformation. Contains the following key-value
        pairs:

        ``'rotation_angle': scalar``
            The rotation angle in radians.
        ``'rotation_anchor': (xanchor, yanchor)``
            Anchor point for the rotation.
        ``'translation_vector': (xdisp, ydisp)``
            The x and y translation displacement. If `range_` is not None, this
            value is scaled along with the samples, such that it refers to the
            same units as `range_`.
        ``'scaling_factor': scalar``
            Scaling factor.
        ``'scaling_anchor': (xanchor, yanchor)``
            Anchor point for the scaling transformation.

    Returns
    -------
    x, y : ndarray
        Arrays containing the transformed coordinates.

    """
    angle = info.get('rotation_angle', 0.0)
    xc, yc = info.get('rotation_anchor', (0.0, 0.0))
    xtrans, ytrans = info.get('translation_vector', (0.0, 0.0))
    scaling_factor = info.get('scaling_factor', 1.0)
    xo, yo = info.get('scaling_anchor', (0.0, 0.0))

    dxc, dyc = x - xc, y - yc
    dxc, dyc = _rotate(dxc, dyc, angle)
    x, y = xc + dxc, yc + dyc

    # Beware! Using += could modify the original arrays
    x, y = x + xtrans, y + ytrans

    dxo, dyo = x - xo, y - yo
    dxo *= scaling_factor
    dyo *= scaling_factor
    x, y = xo + dxo, yo + dyo

    return x, y


def _equal_items(container):
    """
    Find equal items in a container

    Parameters
    ----------
    container : sequence or mapping
        Container to find equal items in. If a sequence, the elements will be
        compared. If a mapping, the values will be compared. Equals are
        identified based on equality, not identity. Array-valued elements are
        supported.

    Returns
    -------
    list
        A list where each element is a list of indices or keys to equal items
        in `container`.

    """
    if isinstance(container, Mapping):
        index = deque(container.keys())
    else:
        index = deque(range(len(container)))

    equal_indices = []
    while index:
        index1 = index.popleft()
        value = container[index1]
        equals = [index1]
        new_index = deque(index)
        for index2 in index:
            if numpy.all(container[index2] == value):
                equals.append(index2)
                new_index.remove(index2)
        index = new_index
        equal_indices.append(equals)
    return equal_indices


def _unique_items(container):
    """
    Remove duplicate items from a sequence or mapping

    Parameters
    ----------
    container : sequence or mapping
        Container to remove duplicates from. If a sequence, the elements will
        be compared. If a mapping, the values will be compared. Duplicates are
        removed based on equality, not identity. Array-valued elements are
        supported.

    Returns
    -------
    sequence or mapping
        Container of the same type as `container`, containing the unique
        items from `container`.

    """
    equal_indices = _equal_items(container)
    if isinstance(container, Mapping):
        new_container = type(container)({index[0]: container[index[0]]
                                         for index in equal_indices})
    else:
        new_container = type(container)([container[index[0]]
                                         for index in equal_indices])
    return new_container


def _tilt(x, y):
    """
    Determine the tilt of the smallest rectangular bounding box that will fit
    around the set of position samples

    Parameters
    ----------
    x, y : array-like
        Arrays containing the position samples.

    Returns
    -------
    scalar
        Tilt angle

    """
    def _bbox_area(tilt):
        rot_x, rot_y = _rotate(x, y, -tilt)
        dx, dy = (numpy.ptp(rot_x[~numpy.isnan(rot_x)]),
                  numpy.ptp(rot_y[~numpy.isnan(rot_y)]))
        return dx * dy

    pi_4 = .25 * numpy.pi
    tilt = optimize.minimize_scalar(_bbox_area, bounds=(-pi_4, pi_4),
                                    method='bounded').x

    return tilt


def _rotate(x, y, angle):
    """
    Rotate a set of position samples by a specified angle around the origin

    Parameters
    ----------
    x, y : array-like
        Arrays containing the position samples.
    angle : scalar
        Angle to rotate, in radians. The samples are rotated in the positive
        right-handed angular direction around the origin of the coordinate
        system.

    Returns
    -------
    x, y : ndarray
        Arrays containing the rotated position samples.

    """
    rot_x = x * numpy.cos(angle) + y * numpy.sin(angle)
    rot_y = y * numpy.cos(angle) - x * numpy.sin(angle)
    return rot_x, rot_y


def _midpoint(x):
    """
    Determine the value of the midpoint between the minimum and maximum value
    in an array

    Parameters
    ----------
    x : array-like

    Returns
    -------
    scalar
        Midpoint between the minimum and maximum in `x`.

    """
    return 0.5 * (numpy.nanmin(x) + numpy.nanmax(x))


def _update_info(info1, info2):
    """
    Update info dict, combining transform info to get the correct info for the
    composite transform

    The transformation is somewhat overparametrized in order to isolate the
    rotation, translation and scaling steps from each other. We therefore adopt
    the following convention: we keep the rotation anchor from the first
    transformation (`info1`) and the scaling anchor from the second
    transformation (`info2`).

    Parameters
    ----------
    info1, info2 : dict
        Info dicts, possibly containing fields such as those returned from
        `transform`.

    Returns
    -------
    info : dict
        The info dict corresponding to the composition of the transforms from
        info1 and info2, in that order. Fields not pertaining to the
        transformation are included as-is in the returned dict, with
        `info2`taking precedence in case of equal keys.

    """
    rotation_angle1 = info1.get('rotation_angle', 0.0)
    rotation_angle2 = info2.get('rotation_angle', 0.0)
    rotation_anchor1 = info1.get('rotation_anchor', (0.0, 0.0))
    rotation_anchor2 = info2.get('rotation_anchor', (0.0, 0.0))
    translation_vector1 = info1.get('translation_vector', (0.0, 0.0))
    translation_vector2 = info2.get('translation_vector', (0.0, 0.0))
    scaling_factor1 = info1.get('scaling_factor', 1.0)
    scaling_factor2 = info2.get('scaling_factor', 1.0)
    scaling_anchor1 = info1.get('scaling_anchor', (0.0, 0.0))
    scaling_anchor2 = info2.get('scaling_anchor', (0.0, 0.0))

    rot_ra1 = _rotate(*rotation_anchor1, angle=rotation_angle2)
    rot_ra2 = _rotate(*rotation_anchor2, angle=rotation_angle2)
    rot_tv1 = _rotate(*translation_vector1, angle=rotation_angle2)
    rot_sa1 = _rotate(*scaling_anchor1, angle=rotation_angle2)

    rotation_angle = rotation_angle1 + rotation_angle2
    rotation_anchor = rotation_anchor1
    translation_vector = tuple(
        rtv1 + rra1 - ra1 + ((tv2 + ra2 - rra2 +
                              (1.0 - scaling_factor1) * (rsa1 - sa2)) /
                             scaling_factor1)
        for (ra1, rra1, ra2, rra2, rtv1, tv2, rsa1, sa2) in
        zip(rotation_anchor1, rot_ra1, rotation_anchor2, rot_ra2,
            rot_tv1, translation_vector2, rot_sa1, scaling_anchor2))
    scaling_factor = scaling_factor1 * scaling_factor2
    scaling_anchor = scaling_anchor2

    info = dict(info1)
    info.update(info2)
    info.update(
        rotation_angle=rotation_angle,
        rotation_anchor=rotation_anchor,
        translation_vector=translation_vector,
        scaling_factor=scaling_factor,
        scaling_anchor=scaling_anchor,
    )
    return info
