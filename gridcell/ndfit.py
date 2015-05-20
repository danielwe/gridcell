#!/usr/bin/env python

"""File: imaps.py
Module defining routines for fitting multivariate functions to measurements

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
from scipy import linalg, optimize

from .utils import gaussian


def fit_ndgaussian(xdata, fdata):
    """
    Fit an n-dimensional Gaussian to a set of values.

    This is not your regular fit-gaussian-to-a-cloud-of-samples function, but
    rather a function fitting a Gaussian to a set of empirical function values
    at selected points (to the histogram of the cloud of samples, if you will).

    :xdata: an array of shape (m, n), where n is the dimension of the domain of
            the gaussian to fit, and m is the number of data points
    :fdata: an array of shape (m,), with empirical function values corresponding
            to each x value
    :returns: fitted parameters scale (scalar), mean (numpy array of shape (n,))
              and cov (numpy array of shape (n, n)) such that scale
              * gaussian(x, mean=mean, cov=cov) returns the values of the fitted
              Gaussian at the positions in x

    """
    m, n = xdata.shape
    n2 = 2 * n
    fsuminv = 1 / numpy.sum(fdata)

    # Estimate initial parameters
    mean = fsuminv * numpy.sum(fdata * xdata.transpose(), axis=1)
    dx = (xdata - mean).transpose()
    cov = fsuminv * (fdata * dx).dot(dx.transpose())

    evals, evecs = linalg.eigh(cov)
    covdet = numpy.prod(evals)

    scale = fdata.max() * numpy.sqrt(covdet * (2 * numpy.pi) ** n)

    # Make sure the matrix of eigenvectors is orthogonal and proper (det +1)
    if linalg.det(evecs) < 0:
        evecs[:, 0] = -evecs[:, 0]

    ## Use the Cayley transform to extract n(n - 1) / 2 independent parameters
    ## from the orthogonal eigenvector matrix
    #eye = numpy.eye(n)
    #evecs_c = (eye - evecs).dot(linalg.inv(eye + evecs))
    #upper = numpy.triu_indices(n, k=1)

    # Use the parametrization in orthogonal_matrix()
    angles = angles_from_orthogonal_matrix(evecs)

    # Make a list with the minimal number of parameters to specify a Gaussian
    #params = numpy.hstack((scale, mean, numpy.sqrt(evals), evecs_c[upper]))
    params = numpy.hstack((scale, mean, numpy.sqrt(evals), angles))
    #params = numpy.hstack((numpy.sqrt(scale), mean, numpy.sqrt(evals), angles))
    #params = numpy.hstack((scale, mean, evals, angles))

    def params_to_scale_mean_cov(params_):
        """
        Extract the scale, mean and covariance matrix from the minimal parameter
        array

        """
        # Extract scale and mean
        #scale_sqrt_ = params_[0]
        #scale_ = scale_sqrt_ * scale_sqrt_
        scale_ = params_[0]

        mean_ = params_[1:n + 1]

        # Get eigenvalues
        evals_sqrt_ = numpy.asarray(params_[n + 1:n2 + 1])
        evals_ = evals_sqrt_ * evals_sqrt_
        #evals_ = numpy.asarray(params_[n + 1:n2 + 1])

        ## Reconstruct the transformed eigenvector matrix
        #cov_c_ = numpy.zeros((n, n))
        #cov_c_[upper] = params_[n2 + 1:]
        #cov_c_.transpose()[upper] = -cov_c_[upper]
        #
        ## Use an inverse Cayley transform to get the true eigenvector matrix
        #evecs_ = (eye - cov_c_).dot(linalg.inv(eye + cov_c_))

        # Get eigenvector matrix from orthogonal_matrix()
        evecs_ = orthogonal_matrix_from_angles(n, params_[n2 + 1:])

        # Get the covariance matrix from the eigenvectors and eigenvalues
        cov_ = evecs_.dot(numpy.diag(evals_).dot(evecs_.transpose()))

        return scale_, mean_, cov_

    def param_gauss(xdata_, *params_):
        """
        Define a Gaussian function specified by a minimal number of parameters

        """
        scale_, mean_, cov_ = params_to_scale_mean_cov(params_)
        return scale_ * gaussian(xdata_, mean=mean_, cov=cov_)

    def error(params_):
        eps = fdata - param_gauss(xdata, *params_)
        return numpy.sum(eps * eps)

    # Find the parameter array that solves the least-squares curve fit problem
    #params, __ = optimize.curve_fit(param_gauss, xdata, fdata, p0=params)
    l = n * (n - 1) // 2
    bounds = ([(0.0, None)] +             # Scale must be positive
              [(None, None)] * n +        # Means for each axis -- any value
              [(None, None)] * n +        # Square roots of evals -- any value
              [(0.0, 2 * numpy.pi)] * l)  # Angles constrained to one cycle
    params = optimize.minimize(error, params, bounds=bounds).x

    scale, mean, cov = params_to_scale_mean_cov(params)

    return scale, mean, cov


def orthogonal_matrix_from_angles(m, angles):
    """
    Compute an orthogonal (m, m) matrix from m (m - 1) / 2 angles

    Reference: Raffenetti, R. C. and Ruedenberg, K. (1969), Parametrization of
    an orthogonal matrix in terms of generalized eulerian angles. Int. J.
    Quantum Chem., 4: 625--634. doi: 10.1002/qua.560040725

    :m: size of matrix to compute
    :angles: sequence of m (m - 1) / 2 angles, given in radians
    :returns: numpy array of shape (m, m) representing an orthogonal matrix

    """
    expected_l = m * (m - 1) // 2
    angles = list(angles)
    l = len(angles)
    if not l == expected_l:
        raise ValueError("need {0} angles to compute orthogonal "
                         "({1}, {1})-matrix".format(expected_l, m))

    matrix = 1.0
    for n in range(1, m):
        small_t = numpy.zeros((n + 1, n + 1))
        small_s = numpy.zeros((n + 1, n + 1))

        small_t[:n, :n] = matrix
        small_t[n, n] = 1.0

        matrix = numpy.zeros((n + 1, n + 1))
        small_s[0, n] = -1.0
        for k in range(n):
            gamma = angles.pop()
            cg, sg = numpy.cos(gamma), numpy.sin(gamma)
            matrix[k] = small_t[k] * cg - small_s[k] * sg
            small_s[k + 1] = small_t[k] * sg + small_s[k] * cg

        matrix[n] = -small_s[n]

    return matrix


def angles_from_orthogonal_matrix(matrix):
    """
    Compute the angles parametrizing an orthogonal matrix

    :matrix: orthogonal matrix
    :returns: list of angles such that orthogonal_matrix(matrix.shape[0],
              angles) returns the given matrix

    """
    n, m = matrix.shape
    if not ((n == m) and numpy.allclose(linalg.det(matrix), 1.0)):
        raise ValueError("'matrix' not orthogonal")
    l = n * (n - 1) // 2
    angles = numpy.zeros(l)

    def error(angles):
        eps = matrix - orthogonal_matrix_from_angles(n, angles)
        err_sq = numpy.sum(eps * eps)
        return err_sq

    # Find the angles that minimize the squared error
    bounds = [(0.0, 2 * numpy.pi)] * l
    angles = optimize.minimize(error, angles, bounds=bounds).x

    return angles
