#!/usr/bin/env python

"""File: formats.py
Module to read .mat-files of various compositions containing simultaneous
tetrode and spatial position recordings, and return the position and spike data
in a standardized format.

"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from os import path
import re
import numpy
from scipy import io, optimize


def pos_and_tetrode(posfile, cellfiles, led=0, range_=None, translate=False,
                    rotate=False):
    """
    Read a collection of .mat-files comprising one file with positional data and
    one file per identified cell cluster with spike times.

    Example: BEN_*.mat

    :posfile: file path or object with position data
    :cellfiles: list of file paths or objects to files with spike times
    :led: which led to use position data from. Possible values:
        0: Default -- the average position is used
        1: The position of LED 1 is used
        2: The position of LED 2 is used
    :range_: range specification giving the x and y values of the edges of the
             environment. The format is a tuple '((xmin, xmax), (ymin, ymax))'.
             The position data will be rescaled to fit within this range, such
             that the most extreme positions touch the range edges while the
             center position stays put. If None, no scaling is performed.
    :translate: if True, the position coordinates are translated to the center
                of the environment defined by range_ before scaling, such that
                the amount of scaling needed to fit the data inside the range is
                minimized. If range_ is None, this has no effect.
    :rotate: if True, the position coordinates are rotated such that the
             apparent edges of the environment line up with the axes of the
             coordinate system. This is done before any scaling and translation.
    :returns: dict with the following fields:
        't': array of position sample times
        'x': array of position sample x-values
        'y': array of position sample y-values
        'spike_ts': dict containing a field
            label: spike time array
                   for each recorded cell. The label key is extracted from the
                   cell file name.
        'info': dict containing information from the computation, in particular:
            'tilt': the angle the positions were rotated (if rotate == True)
            'disp': tuple with the x and y translation distance (if translate ==
                    True). This value is computed after scaling.
            'scale_factor': scale_factor

    """
    pos = io.loadmat(posfile, squeeze_me=True,
                     variable_names=['post', 'posx', 'posx2', 'posy', 'posy2'])
    t = pos['post']
    l = t.size

    # Scaling, translation and rotation is computed based on both LED positions
    xall = numpy.hstack((pos['posx'], pos['posx2']))
    yall = numpy.hstack((pos['posy'], pos['posy2']))
    xall, yall, info = _transform(xall, yall, range_=range_,
                                  translate=translate, rotate=rotate)

    x1, y1 = xall[:l], yall[:l]
    x2, y2 = xall[l:], yall[l:]

    if led == 0:
        x, y = .5 * (x1 + x2), .5 * (y1 + y2)
    elif led == 1:
        x, y = x1, y1
    elif led == 2:
        x, y = x2, y2
    elif not led == 0:
        raise ValueError("'led' must be 0, 1 or 2")

    spike_ts = {_cell_label(path.splitext(_filename(cf))[0]):
                io.loadmat(cf, squeeze_me=True,
                           variable_names=['cellTS'])['cellTS']
                for cf in cellfiles}

    return {'t': t, 'x': x, 'y': y, 'spike_ts': spike_ts, 'info': info}


def single_session(recfile, range_=None, translate=False, rotate=False):
    """
    Read a single .mat-file containing position and spike time data from
    a single recording session

    Example: Cells Square day 1.mat

    :recfile: file path or object with recorded data
    :range_: range specification giving the x and y values of the edges of the
             environment. The format is a tuple '((xmin, xmax), (ymin, ymax))'.
             The position data will be rescaled to fit within this range, such
             that the most extreme positions touch the range edges while the
             center position stays put. If None, no scaling is performed.
    :translate: if True, the position coordinates are translated to the center
                of the environment defined by range_ before scaling, such that
                the amount of scaling needed to fit the data inside the range is
                minimized. If range_ is None, this has no effect.
    :rotate: if True, the position coordinates are rotated such that the
             apparent edges of the environment line up with the axes of the
             coordinate system. This is done before any scaling and translation.
    :returns: dict with the following fields:
        't': array of position sample times
        'x': array of position sample x-values
        'y': array of position sample y-values
        'spike_ts': dict containing a field
            label: spike time array
                   for each recorded cell. The label key is extracted from the
                   cell file name.
        'info': dict containing information from the computation, in particular:
            'tilt': the angle the positions were rotated (if rotate == True)
            'disp': tuple with the x and y translation distance (if translate ==
                    True). This value is computed after scaling.
            'scale_factor': scale_factor

    """
    data = io.loadmat(recfile, squeeze_me=True)

    t = data.pop('post')
    x = data.pop('posx')
    y = data.pop('posy')

    x, y, info = _transform(x, y, range_=range_, translate=translate,
                            rotate=rotate)

    spike_ts = {}
    for key, value in data.items():
        try:
            spike_ts[_cell_label(key)] = value
        except ValueError:
            # Not all keys in data represent are associated with spike times. In
            # this case, _cell_label(key) will raise a ValueError, which is
            # silently ignored since these entries are of no interest anyway
            pass

    return {'t': t, 'x': x, 'y': y, 'spike_ts': spike_ts, 'info': info}


def multiple_sessions(recfile, range_=None, translate=False, rotate=False):
    """
    Read a single .mat-file containing position and spike time data from
    multiple recording sessions

    Example: Ivan_grids.mat

    :rec_file: file path or object with recorded data
    :range_: range specification giving the x and y values of the edges of the
             environment. The format is a tuple '((xmin, xmax), (ymin, ymax))'.
             The position data will be rescaled to fit within this range, such
             that the most extreme positions touch the range edges while the
             center position stays put. If None, no scaling is performed.
    :translate: if True, the position coordinates are translated to the center
                of the environment defined by range_ before scaling, such that
                the amount of scaling needed to fit the data inside the range is
                minimized. If range_ is None, this has no effect.
    :rotate: if True, the position coordinates are rotated such that the
             apparent edges of the environment line up with the axes of the
             coordinate system. This is done before any scaling and translation.
    :returns: list containing a dict per recording session, with the following
              fields:
        't': array of position sample times
        'x': array of position sample x-values
        'y': array of position sample y-values
        'spike_ts': dict containing a field
            label: spike time array
                   for each recorded cell. The index of the cell in the .mat
                   file arrays is used as the label key.
        'info': dict containing information from the computation, in particular:
            'tilt': the angle the positions were rotated (if rotate == True)
            'disp': tuple with the x and y translation distance (if translate ==
                    True). This value is computed after scaling.
            'scale_factor': scale_factor
    """
    data = io.loadmat(recfile, squeeze_me=True,
                      variable_names=['pos', 'stimes', 'allpos', 'allst'])

    if 'pos' in data.keys():
        poskey = 'pos'
    elif 'allpos' in data.keys():
        poskey = 'allpos'
    else:
        raise ValueError("{} contains no known variable names for position"
                         .format(recfile))

    if 'stimes' in data.keys():
        stkey = 'stimes'
    elif 'allst' in data.keys():
        stkey = 'allst'
    else:
        raise ValueError("{} contains no known variable names for spike times"
                         .format(recfile))

    stimes = data[stkey]
    positions = [numpy.transpose(p) for p in data[poskey]]
    return find_sessions(positions, stimes, range_=range_, translate=translate,
                         rotate=rotate)


def processed(recfile, range_=None, translate=False, rotate=False):
    """
    Read a single .mat-file containing position and spike time data from
    multiple recording sessions, along with a full set of processed data

    Example: omegaM2Fract.mat

    :recfile: file path or object with recorded data
    :range_: range specification giving the x and y values of the edges of the
             environment. The format is a tuple '((xmin, xmax), (ymin, ymax))'.
             The position data will be rescaled to fit within this range, such
             that the most extreme positions touch the range edges while the
             center position stays put. If None, no scaling is performed.
    :translate: if True, the position coordinates are translated to the center
                of the environment defined by range_ before scaling, such that
                the amount of scaling needed to fit the data inside the range is
                minimized. If range_ is None, this has no effect.
    :rotate: if True, the position coordinates are rotated such that the
             apparent edges of the environment line up with the axes of the
             coordinate system. This is done before any scaling and translation.
    :returns: list containing a dict per recording session, with the following
              fields:
        't': array of position sample times
        'x': array of position sample x-values
        'y': array of position sample y-values
        'spike_ts': dict containing a field
            label: spike time array
                   for each recorded cell. The 'CellID' parameter from the .mat
                   file is used as the label key.
        'info': dict containing information from the computation, in particular:
            'tilt': the angle the positions were rotated (if rotate == True)
            'disp': tuple with the x and y translation distance (if translate ==
                    True). This value is computed after scaling.
            'scale_factor': scale_factor

    """
    data = io.loadmat(recfile, variable_names=['rOmega'])['rOmega']

    stimes = data['cellTS'][0][0][0]
    positions = [numpy.hstack((t, x, y)).transpose()
                 for (t, x, y) in zip(data['PosT'][0][0][0],
                                      data['PosX'][0][0][0],
                                      data['PosY'][0][0][0])]
    labels = [_cell_label(path.splitext(sname[0])[0])
              for sname in data['SessionName'][0][0][0]]

    # There is a possibility to extract dates for session labels from
    # data['SessionName']. Will implement something if/when necessary.

    return find_sessions(positions, stimes, labels=labels, range_=range_,
                         translate=translate, rotate=rotate)


def find_sessions(positions, stimes, labels=None, range_=None, translate=False,
                  rotate=False):
    """
    Identify separate sessions in multiple position and spike time recordings

    :positions: list of position recordings for each cell
    :stimes: list of spike time recordings for each cell
    :labels: list of labels corresponding to the spike time recordings. These
             are used as keys identifying each cell in the spike_ts dict for
             each session. If None, the corresponding index in the 'stimes'
             array is used.
    :range_: range specification giving the x and y values of the edges of the
             environment. The format is a tuple '((xmin, xmax), (ymin, ymax))'.
             The position data will be rescaled to fit within this range, such
             that the most extreme positions touch the range edges while the
             center position stays put. If None, no scaling is performed.
    :translate: if True, the position coordinates are translated to the center
                of the environment defined by range_ before scaling, such that
                the amount of scaling needed to fit the data inside the range is
                minimized. If range_ is None, this has no effect.
    :rotate: if True, the position coordinates are rotated such that the
             apparent edges of the environment line up with the axes of the
             coordinate system. This is done before any scaling and translation.
    :returns: list containing a dict per recording session, with the following
              fields:
        't': array of position sample times
        'x': array of position sample x-values
        'y': array of position sample y-values
        'spike_ts': dict containing a field
            label: spike time array
                   for each recorded cell. The element in 'labels' corresponding
                   to the cell used as the label key.
        'info': dict containing information from the computation, in particular:
            'tilt': the angle the positions were rotated (if rotate == True)
            'disp': tuple with the x and y translation distance (if translate ==
                    True). This value is computed after scaling.
            'scale_factor': scale_factor

    """
    session_indices = _find_equal_indices(positions)

    if labels is None:
        labels = range(len(stimes))

    sessions = []
    for ind in session_indices:
        t = positions[ind[0]][0]
        x, y = positions[ind[0]][1:]
        x, y, info = _transform(x, y, range_=range_, translate=translate,
                                rotate=rotate)
        spike_ts = {labels[i]: stimes[i] for i in ind}
        sessions.append({'t': t, 'x': x, 'y': y, 'spike_ts': spike_ts,
                         'info': info})

    return sessions


def global_transform(dataset, range_=None, translate=False, rotate=False):
    """
    Apply scaling, rotation and translation to the aggregate of the positions
    from all sessions.

    This is an alternative to applying the transformations separately for each
    session, as is done by the format-specific functions if transformations are
    requested.

    :dataset: list of containing a dict per recording session, akin to the
              output of find_sessions().
    :range_: range specification giving the x and y values of the edges of the
             environment. The format is a tuple '((xmin, xmax), (ymin, ymax))'.
             The position data will be rescaled to fit within this range, such
             that the most extreme positions touch the range edges while the
             center position stays put. If None, no scaling is performed.
    :translate: if True, the position coordinates are translated to the center
                of the environment defined by range_ before scaling, such that
                the amount of scaling needed to fit the data inside the range is
                minimized. If range_ is None, this has no effect.
    :rotate: if True, the position coordinates are rotated such that the
             apparent edges of the environment line up with the axes of the
             coordinate system. This is done before any scaling and translation.
    :returns: transformed dataset

    """
    xarrs, yarrs, lengths = [], [], []
    for session in dataset:
        xarrs.append(session['x'])
        yarrs.append(session['y'])
        lengths.append(session['t'].size)
    xall, yall = numpy.hstack(xarrs), numpy.hstack(yarrs)

    xall, yall, info = _transform(xall, yall, range_=range_,
                                  translate=translate, rotate=rotate)

    new_dataset = []
    for (l, session) in zip(lengths, dataset):
        # Want side-effect free function, so the dict is copied
        new_session = dict(session)
        x, y = xall[:l], yall[:l]
        xall, yall = xall[l:], yall[l:]
        new_session['x'], new_session['y'] = x, y
        new_session['globalinfo'] = info
        new_dataset.append(new_session)

    return new_dataset


def _find_equal_indices(li):
    """
    Find the indices of equal elements in a list

    :li: list
    :returns: a list where each element is a list of indices that refer to equal
              elements in 'li'

    """
    # Make a disposable local copy of the list
    local_li = list(li)

    equal_indices = []
    while local_li:
        e1 = local_li[0]
        same = []
        # To find the correct indices, we must loop over the untouched list
        for (i, e2) in enumerate(li):
            # Use numpy.all() in case the list elements are arrays
            if numpy.all(e1 == e2):
                same.append(i)
                # Since numpy arrays don't return a bool on comparison,
                # local_pos.remove(p2) won't work if elements are numpy arrays.
                # This does the trick:
                for j in range(len(local_li)):
                    if local_li[j] is e2:
                        del local_li[j]
                        break
        equal_indices.append(same)

    return equal_indices


def _transform(x, y, range_=None, translate=False, rotate=False):
    """
    Scale, rotate and translate a set of position recordings

    :x: array of x coordinates
    :y: array of y coordinates
    :range_: range specification giving the x and y values of the edges of the
             environment. The format is a tuple '((xmin, xmax), (ymin, ymax))'.
             The position data will be rescaled to fit within this range, such
             that the most extreme positions touch the range edges while the
             center position stays put. If None, no scaling is performed.
    :translate: if True, the position coordinates are translated to the center
                of the environment defined by range_ before scaling, such that
                the amount of scaling needed to fit the data inside the range is
                minimized. If range_ is None, this has no effect.
    :rotate: if True, the position coordinates are rotated such that the
             apparent edges of the environment line up with the axes of the
             coordinate system. This is done before any scaling and translation.
    :returns: transformed x and y arrays, and a dict info containing information
              from the computation, in particular:
        'tilt': the angle the positions were rotated (if rotate == True)
        'disp': tuple with the x and y translation distance (if range_ is not
                None). This value is computed after scaling, such that it refers
                to the same units as range_.
        'scale_factor': scaling factor (if range_ is not None)

    """
    info = {'tilt': 0.0, 'disp': 0.0, 'scale_factor': 1.0}
    if rotate:
        tilt = _square_tilt(x, y)
        x, y = _rotate_pos(x, y, tilt)
        info['tilt'] = tilt

    if range_ is not None:
        if translate:
            rcx = .5 * (range_[0][0] + range_[0][1])
            rcy = .5 * (range_[1][0] + range_[1][1])
            xdisp, ydisp = rcx - _center(x), rcy - _center(y)
            x += xdisp
            y += ydisp
            info['disp'] = (xdisp, ydisp)

        xc, yc = _center(x), _center(y)
        xcentered, ycentered = x - xc, y - yc
        scale_factor = min((range_[0][0] - xc) / numpy.nanmin(xcentered),
                           (range_[1][0] - yc) / numpy.nanmin(ycentered),
                           (range_[0][1] - xc) / numpy.nanmax(xcentered),
                           (range_[1][1] - yc) / numpy.nanmax(ycentered))
        if scale_factor <= 0.0:
            raise ValueError("cannot scale position data into {} with "
                             "translate == {} and rotate == {}."
                             .format(range_, translate, rotate))

        xcentered *= scale_factor
        ycentered *= scale_factor
        x, y = xcentered + xc, ycentered + yc
        info['scale_factor'] = scale_factor

    return x, y, info


def _square_tilt(x, y):
    """
    Determine the most likely tilt of the square environment with respect to the
    position recording axes

    The tilt is the rotation angle that minimizes the area of the bounding box
    around the data.

    :x: numpy-like array of x positions
    :y: numpy-like array of y positions
    :returns: tilt angle

    """
    def bbox_area(tilt):
        new_x, new_y = _rotate_pos(x, y, tilt)
        dx, dy = (numpy.ptp(new_x[~numpy.isnan(new_x)]),
                  numpy.ptp(new_y[~numpy.isnan(new_y)]))
        return dx * dy

    pi_4 = .25 * numpy.pi
    tilt = optimize.minimize_scalar(bbox_area, bounds=(-pi_4, pi_4),
                                    method='bounded').x

    return tilt


def _rotate_pos(x, y, tilt):
    """
    Rotate the axes of position recordings with a specified tilt

    :x: numpy-like array of x positions
    :y: numpy-like array of y positions
    :tilt: angle to rotate
    :returns: numpy arrays of x and y positions along the rotated axes

    """
    new_x = x * numpy.cos(tilt) + y * numpy.sin(tilt)
    new_y = y * numpy.cos(tilt) - x * numpy.sin(tilt)
    return new_x, new_y


def _center(x):
    """
    Determine the value of the midpoint between the minimum and maximum value in
    an array

    :x: array-like
    :returns: average of minimum and maximum value of x

    """
    return 0.5 * (numpy.nanmin(x) + numpy.nanmax(x))


def _filename(file_):
    """
    Robustly extract the filename from a file path or object

    :file_: file path or object
    :returns: the name of the file

    """
    try:
        return path.basename(file_.name)
    except AttributeError:
        return path.basename(file_)


def _cell_label(string):
    """
    Extract numbers from a string identifying a tetrode and cluster, and compose
    a standardized label from these numbers.

    The format of the label is e.g. T12C3, where 12 is the tetrode number and
    3 is the cluster number. The numbers are extracted from 'string', assuming
    the last integer denotes the cluster and the next-to-last integer denotes
    the tetrode.

    :string: string identifying the cluster
    :returns: standardized label

    """
    numbers = re.findall(r'\d+', string)
    try:
        return "T" + numbers[-2] + "C" + numbers[-1]
    except IndexError:
        raise ValueError("'string' does not identify a tetrode and cluster")
