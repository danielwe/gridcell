#!/usr/bin/env python

"""File: kde.py
Module defining kernel density estimators

"""
# Copyright 2016 Daniel Wennberg
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

import numpy
from scipy.special import i0, gamma, hyp1f2
from scipy.integrate import quad
from scipy.spatial.distance import cdist
from .memoize.memoize import memoize_function


_PI = numpy.pi
_2PI = 2.0 * _PI
_PI_2 = 0.5 * _PI
_PI_SQ = _PI * _PI
_SQRT2PI = numpy.sqrt(_2PI)


def _vn(dim):
    dim_2 = 0.5 * dim
    return (_PI ** (dim_2)) / gamma(dim_2 + 1)


def _sn(dim):
    return _2PI * _vn(dim - 1)


def _tophat_form(u):
    uabs = numpy.abs(u)
    valid = uabs < 1.0
    out = numpy.zeros_like(uabs, dtype=numpy.float_)
    out[valid] = 1.0
    return out


def _tophat_volume(dim):
    return _vn(dim)


def _linear_form(u):
    uabs = numpy.abs(u)
    valid = uabs < 1.0
    out = numpy.zeros_like(u, dtype=numpy.float_)
    out[valid] = 1.0 - uabs[valid]
    return out


def _linear_volume(dim):
    return _vn(dim) / (dim + 1)


def _uniweight_form(u):
    uabs = numpy.abs(u)
    valid = uabs < 1.0
    out = numpy.zeros_like(uabs, dtype=numpy.float_)
    uabsvalid = uabs[valid]
    out[valid] = 1.0 - uabsvalid * uabsvalid
    return out


def _uniweight_volume(dim):
    return _vn(dim) * 2.0 / (dim + 2)


def _biweight_form(u):
    uweight = _uniweight_form(u)
    return uweight * uweight


def _biweight_volume(dim):
    return 8 * _vn(dim) / ((dim + 2) * (dim + 4))


def _triweight_form(u):
    uweight = _uniweight_form(u)
    return uweight * uweight * uweight


def _triweight_volume(dim):
    return 48 * _vn(dim) / ((dim + 2) * (dim + 4) * (dim + 6))


def _quadweight_form(u):
    uweight = _uniweight_form(u)
    return uweight * uweight * uweight * uweight


def _quadweight_volume(dim):
    return 384 * _vn(dim) / ((dim + 2) * (dim + 4) * (dim + 6) * (dim + 8))


def _bump_form(u):
    uweight = _uniweight_form(u)
    return numpy.exp(-(1.0 / uweight))


@memoize_function
def _bump_volume(dim):
    def integrand(u):
        return (u ** (dim - 1)) * _bump_form(u)
    return _sn(dim - 1) * quad(integrand, 0, 1)[0]


def _cosine_form(u):
    uabs = numpy.abs(u)
    valid = uabs < _PI_2
    out = numpy.zeros_like(uabs, dtype=numpy.float_)
    out[valid] = numpy.cos(uabs[valid])
    return out


def _cosine_volume(dim):
    hyp, _ = hyp1f2(0.5 * dim, 0.5, 0.5 * dim + 1, -_PI_SQ / 16.0)
    return _vn(dim) * (_PI_2 ** dim) * hyp


def _exponential_form(u):
    return numpy.exp(-numpy.abs(u))


def _exponential_volume(dim):
    return _sn(dim - 1) * gamma(dim)


def _gaussian_form(u):
    uabs = numpy.abs(u)
    return numpy.exp(-0.5 * (uabs * uabs))


def _gaussian_volume(dim):
    return _2PI ** (0.5 * dim)

kernel_pieces = {
    'tophat': (_tophat_form, _tophat_volume),
    'linear': (_linear_form, _linear_volume),
    'uniweight': (_uniweight_form, _uniweight_volume),
    'biweight': (_biweight_form, _biweight_volume),
    'triweight': (_triweight_form, _triweight_volume),
    'quadweight': (_quadweight_form, _quadweight_volume),
    'bump': (_bump_form, _bump_volume),
    'cosine': (_cosine_form, _cosine_volume),
    'exponential': (_exponential_form, _exponential_volume),
    'gaussian': (_gaussian_form, _gaussian_volume),
}


def _normalized_kernel(form, volume):
    sigma = numpy.sqrt(volume(3) / (_2PI * volume(1)))

    def normalized_kernel(u, bandwidth, dim=1):
        k = sigma / bandwidth
        return (1.0 / volume(dim)) * (k ** dim) * form(k * u)
    return normalized_kernel

kernels = {
    name: _normalized_kernel(*pieces)
    for name, pieces in kernel_pieces.items()
}


def _vonmises_kernel(u, bandwidth, dim=1):
    uabs = numpy.abs(u)
    kappa = 1.0 / (bandwidth * bandwidth)
    return ((1.0 / (_2PI * i0(kappa))) *
            numpy.exp(kappa * numpy.cos(uabs)))

kernels['vonmises'] = _vonmises_kernel


def _kde(kernel):

    def _fix_arr(arr):
        arr = numpy.asarray(arr)
        if len(arr.shape) == 1:
            arr = numpy.atleast_2d(arr).transpose()
        return arr

    def kde(data, sample_points, bandwidth, weights=None):
        data = _fix_arr(data)
        sample_points = _fix_arr(sample_points)

        # Loop over sample points to save memory
        nsp, dim = sample_points.shape
        nd, ddim = data.shape
        if not dim == ddim:
            raise ValueError("'data' and 'sample_points' have different "
                             "dimensions")

        density = numpy.empty((nsp, ))
        if weights is None:
            weights = numpy.empty((nd, ))
            weights.fill(1.0 / nd)
        for i in range(nsp):
            distances = cdist(sample_points[i:i + 1], data)
            density[i] = numpy.sum(
                weights * kernel(distances, bandwidth, dim),
                axis=-1,
            )
        return density
    return kde

kernel_density_estimators = {
    name: _kde(kernel) for name, kernel in kernels.items()
}
