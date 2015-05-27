#!/usr/bin/env python

"""File: shapes.py
Module defining objects for geometric shapes, with attributes to retrieve
parameters characterizing the shapes, and plotting methods to include the
shapes matplotlib figure.

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
from scipy import linalg
from matplotlib import pyplot, patches


EPS = numpy.finfo(numpy.float_).eps


class Ellipse(object):
    """
    Represent an ellipse and obtain all parameters related to it

    """

    def __init__(self, agparams=None, canonparams=None, fitpoints=None,
                 f0=1.0):
        """
        Initialize the Ellipse instance

        The instance can be initialized using either a set of canonical
        parameters (center (xc, yc), semi-major axis a, semi-minor axis b,
        angle tilt), a set of analytic geometry parameters (A, B, C, D, E, F),
        or a set of points to fit the ellipse to.

        :agparams: sequence of analytic geometry parameters:
            A, B, C, D, E, F such that the ellipse is described by
             A * x ** 2 + 2 * B * x * y + C * y ** 2 + 2 * f0 * (D * x + E * y)
             + f0 ** 2 * F = 0
        :canonparams: sequence of canonical ellipse parameters:
            (xc, yc), a, b, tilt
        :fitpoints: sequence of coordinates of points to fit the ellipse to:
            (x0, y0), (x1, y1), ...
        :f0: order-of-magnitude estimate of the characteristic length scale of
             the ellipse. Used to normalize the analytic geometry parameters
             for increased numerical stability.

        """
        self.f0 = f0

        if agparams is not None:
            # Save analytic geometry attributes
            self.A, self.B, self.C, self.D, self.E, self.F = agparams
            # Compute canonical attributes
            self.center, self.a, self.b, self.tilt = self.ag_to_canon(agparams,
                                                                      f0)
        elif canonparams is not None:
            # Save canonical attributes
            self.center, self.a, self.b, tilt = canonparams
            # Wrap tilt into [-pi, pi]
            pi_2 = 0.5 * numpy.pi
            self.tilt = numpy.mod(tilt + pi_2, numpy.pi) - pi_2
            # Compute analytic geometry attributes
            self.A, self.B, self.C, self.D, self.E, self.F = self.canon_to_ag(
                canonparams, f0)
        elif fitpoints is not None:
            # Fit and save analytic geometry attributes
            agparams_ = self.fit(fitpoints, f0)
            self.A, self.B, self.C, self.D, self.E, self.F = agparams_

            # Compute canonical attributes
            self.center, self.a, self.b, self.tilt = (
                self.ag_to_canon(agparams_, f0))
        else:
            raise ValueError("no data provided to initialize Ellipse")

        # Compute eccentricity
        self.ecc = numpy.sqrt(1.0 - (self.b * self.b) / (self.a * self.a))

    @staticmethod
    def ag_to_canon(agparams, f0):
        """
        Compute canonical parameters from analytic geometry parameters

        :agparams: sequence of analytic geometry parameters:
            [A, B, C, D, E, F] such that the ellipse is described by
             A * x ** 2 + 2 * B * x * y + C * y ** 2 + 2 * f0 * (D * x + E * y)
             + f0 ** 2 * F = 0
        :f0: order-of-magnitude estimate of the characteristic length scale of
             the ellipse. Used to normalize the analytic geometry parameters
             for increased numerical stability.
        :returns: canonical ellipse parameters:
            (xc, yc), a, b, tilt

        """
        A, B, C, D, E, F = agparams

        B2 = B * B
        AmC = A - C

        disc = B2 - A * C
        idisc = 1.0 / disc

        xc = idisc * f0 * (C * D - B * E)
        yc = idisc * f0 * (A * E - B * D)
        center = (xc, yc)

        lambda_ = idisc * f0 * (f0 * F + D * xc + E * yc)

        gamma = A + C
        kappa = numpy.sqrt(4 * B2 + AmC * AmC)

        sgn = numpy.abs(lambda_) / lambda_

        a = numpy.sqrt(0.5 * lambda_ * (gamma + sgn * kappa))
        b = numpy.sqrt(0.5 * lambda_ * (gamma - sgn * kappa))

        tilt = 0.5 * numpy.arctan2(-sgn * 2.0 * B / kappa, -sgn * AmC / kappa)

        return center, a, b, tilt

    @staticmethod
    def canon_to_ag(canonparams, f0):
        """
        Compute analytic geometry parameters from canonical parameters

        :canonparams: sequence of canonical ellipse parameters:
            (xc, yc), a, b, tilt
        :f0: order-of-magnitude estimate of the characteristic length scale of
             the ellipse. Used to normalize the analytic geometry parameters
             for increased numerical stability.
        :returns: analytic geometry parameters:
            [A, B, C, D, E, F] such that the ellipse is described by
             A * x ** 2 + 2 * B * x * y + C * y ** 2 + 2 * f0 * (D * x + E * y)
             + f0 ** 2 * F = 0
        """
        (xc, yc), a, b, tilt = canonparams

        a2, b2 = a * a, b * b
        st, ct = numpy.sin(tilt), numpy.cos(tilt)
        st2, ct2 = st * st, ct * ct

        if0 = 1.0 / f0

        A = b2 * ct2 + a2 * st2
        B = (b2 - a2) * st * ct
        C = b2 * st2 + a2 * ct2
        D = -if0 * (A * xc + B * yc)
        E = -if0 * (B * xc + C * yc)
        F = -if0 * (D * xc + E * yc + if0 * a2 * b2)

        return A, B, C, D, E, F

    @staticmethod
    def fit(fitpoints, f0):
        """
        Find the ellipse most closely fitting a set of points in the plane

        References:

        Kanatani, K., Al-Sharadqah, A., Chernov, N., & Sugaya, Y. (2012).
        Computer Vision -- ECCV 2012. (A. Fitzgibbon, S. Lazebnik, P. Perona,
        Y.  Sato, & C.  Schmid, Eds.)Lecture Notes in Computer Science
        (including subseries Lecture Notes in Artificial Intelligence and
        Lecture Notes in Bioinformatics) (Vol.  7574, pp. 384--397). Berlin,
        Heidelberg: Springer Berlin Heidelberg.  doi:10.1007/978-3-642-33712-3

        Masuzaki, T., Sugaya, Y., & Kanatani, K. (2014). High accuracy
        ellipse-specific fitting. In Lecture Notes in Computer Science
        (including subseries Lecture Notes in Artificial Intelligence and
        Lecture Notes in Bioinformatics) (Vol. 8333 LNCS, pp. 314--324).
        doi:10.1007/978-3-642-53842-1-27

        :fitpoints: list of coordinates to the points to fit the ellipse to.
        :f0: order-of-magnitude estimate of the characteristic length scale of
             the ellipse. Used to normalize the analytic geometry parameters
             for increased numerical stability.
        :returns: analytic geometry parameters:
            array([A, B, C, D, E, F]) such that the ellipse is described by
            A * x ** 2 + 2 * B * x * y + C * y ** 2 + 2 * f0 * (D * x + E * y)
            + f0 ** 2 * F = 0

        """
        px, py = numpy.asarray(zip(*fitpoints))
        px2, py2 = px * px, py * py
        pxy = px * py

        zer = numpy.zeros_like(px)

        f02 = numpy.empty_like(px)
        f02.fill(f0 * f0)
        fpx, fpy = f0 * px, f0 * py

        n = px.size

        xi = numpy.vstack((px2,
                           2.0 * pxy,
                           py2,
                           2.0 * fpx,
                           2.0 * fpy,
                           f02)).transpose()

        # Fit conic exactly if we can
        svdlvecs, svdvals, svdrvecs = linalg.svd(xi)
        svdtol = (svdvals.max() * max(xi.shape) * EPS)
        rank = numpy.sum(svdvals > svdtol)
        if rank == 5:
            #print("rank == 5 True")
            agparams = svdrvecs[svdvals.argmin()]

            # Check if the fitted conic is an ellipse
            disc = agparams[1] * agparams[1] - agparams[0] * agparams[2]
            if disc < 0.0:
                #print("ellipse True")
                return tuple(agparams)

        # Otherwise, proceed with fitting
        covmats = 4.0 * numpy.array(
            ((px2, pxy, zer, fpx, zer, zer),
             (pxy, px2 + py2, pxy, fpy, fpx, zer),
             (zer, pxy, py2, zer, fpy, zer),
             (fpx, fpy, zer, f02, zer, zer),
             (zer, fpx, fpy, zer, f02, zer),
             (zer, zer, zer, zer, zer, zer))).transpose()
        xixi = numpy.array([numpy.outer(x, x) for x in xi])
        e = numpy.zeros((6,))
        e[(0, 2), ] = 1.0
        sxie = numpy.array([numpy.outer(x, e) for x in xi])
        sxie = sxie + sxie.swapaxes(1, 2)

        agparams_old = numpy.zeros((6,))
        w = numpy.ones_like(px)
        for _ in xrange(100):
            mmat = numpy.average(w * xixi.transpose(), axis=-1)
            #print(mmat == mmat.transpose())

            # Find pseudoinverse of mmat with truncated rank
            svdlvecs, svdvals, svdrvecs = linalg.svd(mmat)
            isvdvals = 1.0 / svdvals
            isvdvals[svdvals.argmin()] = 0.0
            isvdvalm = linalg.diagsvd(isvdvals, *mmat.shape[::-1])
            mmat_pinv = svdrvecs.conj().transpose().dot(
                isvdvalm.dot(svdlvecs.conj().transpose()))

            t1, st2 = [], []
            for (x, xx, cov) in zip(xi, xixi, covmats):
                t1.append(x.dot(mmat_pinv.dot(x)) * cov)
                st2.append(cov.dot(mmat_pinv.dot(xx)))
            term1 = numpy.asarray(t1)
            sterm2 = numpy.asarray(st2)
            sterm2 = sterm2 + sterm2.swapaxes(1, 2)

            nmat = (numpy.average(w * (covmats + sxie), axis=0) -
                    numpy.average(w * w * (term1 + sterm2), axis=0) / n)
            #print(nmat == nmat.transpose())

            agvals, agvecs = linalg.eig(mmat, nmat)
            min_ = numpy.abs(agvals).argmin()
            agparams = numpy.real_if_close(agvecs[:, min_].squeeze())
            #print(agparams)

            if (numpy.allclose(agparams, agparams_old) or
                    numpy.allclose(agparams, -agparams_old)):
                #print(count)
                break
            else:
                w = numpy.array([1.0 / agparams.dot(cov.dot(agparams))
                                 for cov in covmats])
                agparams_old = agparams

        disc = agparams[1] * agparams[1] - agparams[0] * agparams[2]
        if disc > 0.0:
            raise ValueError("ellipse fit failed")

        return tuple(agparams)

    def patch(self, **kwargs):
        """
        Return an Ellipse patch instance from matplotlib.patches for this
        ellipse

        :kwargs: passed through to the matplotlib.patches.Ellipse constructor
        :returns: matplotlib.patches.Ellipse instance

        """
        return patches.Ellipse(self.center, 2.0 * self.a, 2.0 * self.b,
                               angle=numpy.rad2deg(self.tilt), **kwargs)

    def plot(self, axes=None, majaxis=False, minaxis=False, smajaxis=False,
             sminaxis=False, linestyle=None, linewidth=None, color=None,
             fill=False, axis_kw=None, **kwargs):
        """
        Plot the ellipse

        The ellipse can be added to an existing plot via the optional 'axes'
        argument.

        :axes: Axes instance to add the ellipse to. If None (default), the
               current Axes instance with equal aspect ratio is used if any, or
               a new one created.
        :majaxis,minaxis: if True, the [major,minor] axis is also added to the
                          plot. Default is False.
        :smajaxis,sminaxis: if True, the semi-[major,minor] axis is also added
                            to the plot. Default is False.
        :linestyle: a valid matplotlib linestyle specification to use for the
                    ellipse edge and [major,minor] axis. Defaults to None, such
                    that the rcParams decide. This keyword is provided to
                    easily set the edge and [major,minor] line styles equal,
                    but can be overridden for each of the cases individually
                    using 'axis_kw' and 'kwargs'.
        :linewidth: the linewidth to use for the ellipse edge and [major,minor]
                    axis. Defaults to None, such that the rcParams decide. This
                    keyword is provided to easily set the edge and
                    [major,minor] line styles equal, but can be overridden for
                    each of the cases individually using 'axis_kw' and
                    'kwargs'.
        :color: a valid matplotlib color specification to use for the ellipse
                and [major,minor] axis. Defaults to None, such that the
                rcParams decide. This keyword is provided to easily set the
                edge and [major,minor] line styles equal, but can be overridden
                for each of the cases individually using 'axis_kw' and
                'kwargs'.
        :fill: if True, plot a filled ellipse. If False (default), only plot
               the ellipse edge.
        :axis_kw: dict of keyword arguments to pass to the axes.plot() method
                  used to plot the (semi-)[major,minor] axis. Default: None
                  (empty dict)
        :kwargs: additional keyword arguments passed on to the
                 patches.Ellipse() constructor. Note in particular the keywords
                 'edgecolor', 'facecolor' and 'label'.
        :returns: list containing the plotted objects: one
                  matplotlib.pathces.Ellipse instance, and a Line2D instance
                  per plotted axis.

        """
        if axes is None:
            axes = pyplot.gca(aspect='equal')

        if axis_kw is None:
            axis_kw = {}

        #t = numpy.linspace(0, 2 * numpy.pi, 200)
        #u = self.a * numpy.cos(t)
        #v = self.b * numpy.sin(t)
        #
        #ct, st = numpy.cos(self.tilt), numpy.sin(self.tilt)
        #x = self.center[0] + u * ct - v * st
        #y = self.center[1] + u * st + v * ct
        #ell = axes.plot(x, y, linestyle=linestyle, linewidth=linewidth,
        #                color=color, **kwargs)

        ell = self.patch(linestyle=linestyle, linewidth=linewidth, color=color,
                         fill=fill, **kwargs)
        ell = axes.add_patch(ell)

        h = [ell]

        if majaxis or smajaxis:
            aphelion = (self.a * numpy.cos(self.tilt),
                        self.a * numpy.sin(self.tilt))
            if majaxis:
                xvals = (-aphelion[0], aphelion[0])
                yvals = (-aphelion[1], aphelion[1])
            else:
                xvals = (0.0, aphelion[0])
                yvals = (0.0, aphelion[1])
            h += axes.plot(xvals, yvals, linestyle=linestyle,
                           linewidth=linewidth, color=color, **axis_kw)

        if minaxis or sminaxis:
            perihelion = (-self.b * numpy.sin(self.tilt),
                          self.b * numpy.cos(self.tilt))
            if minaxis:
                xvals = (-perihelion[0], perihelion[0])
                yvals = (-perihelion[1], perihelion[1])
            else:
                xvals = (0.0, perihelion[0])
                yvals = (0.0, perihelion[1])
            h += axes.plot(xvals, yvals, linestyle=linestyle,
                           linewidth=linewidth, color=color, **axis_kw)

        return h
