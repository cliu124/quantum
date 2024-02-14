"""Polynomial-based differentation matrices.

The m-th derivative of the grid function f is obtained by the matrix-
vector multiplication

.. math::

f^{(m)}_i = D^{(m)}_{ij}f_j

References
----------
..[1] B. Fornberg, Generation of Finite Difference Formulas on Arbitrarily
Spaced Grids, Mathematics of Computation 51, no. 184 (1988): 699-706.

..[2] J. A. C. Weidemann and S. C. Reddy, A MATLAB Differentiation Matrix
Suite, ACM Transactions on Mathematical Software, 26, (2000) : 465-519

..[3] R. Baltensperger and M. R. Trummer, Spectral Differencing With A
Twist, SIAM Journal on Scientific Computing 24, (2002) : 1465-1487
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import cached_property, lru_cache

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import toeplitz
from scipy.special import roots_laguerre


class DiffMatrices(ABC):
    """Differentiation matrices."""

    @property
    @abstractmethod
    def nodes(self) -> NDArray:
        """Position of nodes."""

    @abstractmethod
    def at_order(self, order: int) -> NDArray:
        """Differentiation matrix for the order-th derivative."""


@dataclass(frozen=True)
class Lagrange(DiffMatrices):
    """General differentiation matrices.

    The matrix is constructed by differentiating N-th order Lagrange
    interpolating polynomial that passes through the speficied points.

    This function is based on code by Rex Fuzzle
    https://github.com/RexFuzzle/Python-Library

    Warning: these differentiation matrices are not stable against a strong
    stretching with DiffMatOnDomain.

    Attributes:
        nodes: position of N distinct arbitrary nodes.
        weights: vector of weight values, evaluated at nodes.
        weight_derivs: matrix of size M x N, element (i, j) is the i-th
            derivative of log(weights(x)) at the j-th node.
    """

    at_nodes: NDArray
    weights: NDArray
    weight_derivs: NDArray

    def __post_init__(self) -> None:
        assert self.nodes.ndim == 1
        assert self.nodes.shape == self.weights.shape
        assert self.weight_derivs.ndim == 2
        assert self.weight_derivs.shape[0] < self.nodes.size
        assert self.weight_derivs.shape[1] == self.nodes.size

    @staticmethod
    def with_unit_weights(nodes: NDArray) -> Lagrange:
        return Lagrange(
            nodes,
            weights=np.ones_like(nodes),
            weight_derivs=np.zeros((nodes.size - 1, nodes.size)),
        )

    @property
    def nodes(self) -> NDArray:
        return self.at_nodes

    @cached_property
    def _helper_matrices(self) -> tuple[NDArray, NDArray, NDArray]:
        N = self.nodes.size
        XX = np.tile(self.nodes, (N, 1))
        DX = np.transpose(XX) - XX  # DX contains entries x(k)-x(j)
        np.fill_diagonal(DX, 1.0)
        c = self.weights * np.prod(DX, 1)  # quantities c(j)
        C = np.tile(c, (N, 1))
        C = np.transpose(C) / C  # matrix with entries c(k)/c(j).
        Z = 1 / DX  # Z contains entries 1/(x(k)-x(j)
        np.fill_diagonal(Z, 0.0)
        # X is Z.T with diagonal removed
        X = Z[~np.eye(N, dtype=bool)].reshape(N, -1).T
        return C, Z, X

    @cached_property
    def _all_dmats(self) -> dict[int, tuple[NDArray, NDArray]]:
        # mapping from order to (Y, D) tuple
        N = self.nodes.size

        D = np.eye(N)  # differentation matrix
        Y = np.ones_like(D)  # matrix of cumulative sums
        return {0: (Y, D)}

        max_order = self.weight_derivs.shape[0]
        DM = np.empty((max_order, N, N))

        for ell in range(1, max_order + 1):
            DM[ell - 1, :, :] = D
        return DM

    def _dmats(self, order: int) -> tuple[NDArray, NDArray]:
        if order in self._all_dmats:
            return self._all_dmats[order]
        yprev, dprev = self._dmats(order - 1)
        C, Z, X = self._helper_matrices
        ynew = np.cumsum(
            np.vstack((self.weight_derivs[order - 1, :], order * yprev[:-1, :] * X)), 0
        )
        dnew = (
            order
            * Z
            * (C * np.transpose(np.tile(np.diag(dprev), (self.nodes.size, 1))) - dprev)
        )  # off-diag
        np.fill_diagonal(dnew, ynew[-1, :])  # diag
        self._all_dmats[order] = ynew, dnew
        return ynew, dnew

    def at_order(self, order: int) -> NDArray:
        assert 1 <= order <= self.weight_derivs.shape[0]
        return self._dmats(order)[1]


@dataclass(frozen=True)
class Chebyshev(DiffMatrices):
    """Chebyshev collocation differentation matrices on [-1, 1].

    The matrices are constructed by differentiating ncheb-th order Chebyshev
    interpolants.

    The code implements two strategies for enhanced accuracy suggested by
    W. Don and S. Solomonoff :

    (a) the use of trigonometric  identities to avoid the computation of
    differences x(k)-x(j)

    (b) the use of the "flipping trick"  which is necessary since sin t can
    be computed to high relative precision when t is small whereas sin (pi-t)
    cannot.

    It may, in fact, be slightly better not to implement the strategies
    (a) and (b). Please consult [3] for details.

    This function is based on code by Nikola Mirkov
    http://code.google.com/p/another-chebpy

    Attributes:
        degree: polynomial degree.
    """

    degree: int

    def __post_init__(self) -> None:
        assert self.degree > 0

    @property
    def max_order(self) -> int:
        """Maximum derivative order."""
        return self.degree

    @cached_property
    def nodes(self) -> NDArray:
        ncheb = self.degree
        # obvious way
        # np.cos(np.pi * np.arange(ncheb+1) / ncheb)
        # W&R way
        return np.sin(np.pi * (ncheb - 2 * np.arange(ncheb + 1)) / (2 * ncheb))

    @cached_property
    def _helper_matrices(self) -> tuple[NDArray, NDArray]:
        nnodes = self.nodes.size
        # indices used for flipping trick
        nn1 = int(np.floor(nnodes / 2))
        nn2 = int(np.ceil(nnodes / 2))
        k = np.arange(nnodes)
        # compute theta vector
        th = k * np.pi / self.degree

        T = np.tile(th / 2, (nnodes, 1))
        # trigonometric identity
        DX = 2 * np.sin(T.T + T) * np.sin(T.T - T)
        # flipping trick
        DX[nn1:, :] = -np.flipud(np.fliplr(DX[0:nn2, :]))
        np.fill_diagonal(DX, 1.0)
        DX = DX.T

        # matrix with entries c(k)/c(j)
        C = toeplitz((-1.0) ** k)
        C[0, :] *= 2
        C[-1, :] *= 2
        C[:, 0] *= 0.5
        C[:, -1] *= 0.5

        # Z contains entries 1/(x(k)-x(j))
        Z = 1 / DX
        # with zeros on the diagonal.
        np.fill_diagonal(Z, 0.0)
        return C, Z

    @cached_property
    def _all_dmats(self) -> dict[int, NDArray]:
        # mapping from order to differentation matrix
        return {0: np.eye(self.nodes.size)}

    def _dmat(self, order: int) -> NDArray:
        if order in self._all_dmats:
            return self._all_dmats[order]
        dprev = self._dmat(order - 1)
        C, Z = self._helper_matrices
        # off-diagonals
        dnew = order * Z * (C * np.tile(np.diag(dprev), (self.nodes.size, 1)).T - dprev)
        # negative sum trick
        np.fill_diagonal(dnew, -np.sum(dnew, axis=1))
        self._all_dmats[order] = dnew
        return dnew

    def at_order(self, order: int) -> NDArray:
        assert 0 < order <= self.max_order
        return self._dmat(order)


@dataclass(frozen=True)
class Laguerre(DiffMatrices):
    """Laguerre collocation differentiation matrices.

    The matrix is constructed by differentiating Laguerre interpolants.

    Warning: these differentiation matrices are backed by :class:`Lagrange` and
    are not stable against a strong stretching with DiffMatOnDomain.

    Attributes:
        degree: Laguerre polynomial degree. There are degree+1 nodes.
    """

    degree: int

    @property
    def max_order(self) -> int:
        """Maximum order of derivative."""
        return self.degree

    @cached_property
    def nodes(self) -> NDArray:
        nodes = np.zeros(self.degree + 1)
        nodes[1:] = roots_laguerre(self.degree)[0]
        return nodes

    @cached_property
    def _dmat(self) -> Lagrange:
        x = self.nodes
        alpha = np.exp(-x / 2)  # Laguerre weights

        # construct beta(l,j) = d^l/dx^l (alpha(x)/alpha'(x))|x=x_j recursively
        beta = np.zeros([self.max_order, x.size])
        d = np.ones(x.size)
        for ell in range(0, self.max_order):
            beta[ell, :] = pow(-0.5, ell + 1) * d

        return Lagrange(at_nodes=x, weights=alpha, weight_derivs=beta)

    def at_order(self, order: int) -> NDArray:
        return self._dmat.at_order(order)


@dataclass(frozen=True)
class DiffMatOnDomain(DiffMatrices):
    """Differentiation matrices stretched and shifted to a different domain.

    The stretching and shifting is done linearly between xmin and xmax.
    """

    xmin: float
    xmax: float
    dmat: DiffMatrices

    @cached_property
    def stretching(self) -> NDArray:
        return (self.dmat.nodes[-1] - self.dmat.nodes[0]) / (self.xmax - self.xmin)

    @cached_property
    def nodes(self) -> NDArray:
        return (self.dmat.nodes - self.dmat.nodes[0]) / self.stretching + self.xmin

    @lru_cache
    def at_order(self, order: int) -> NDArray:
        return self.stretching**order * self.dmat.at_order(order)


def cheb4c(ncheb: int) -> tuple[NDArray, NDArray]:
    """Chebyshev 4th derivative matrix incorporating clamped conditions.

    The function x, D4 =  cheb4c(N) computes the fourth
    derivative matrix on Chebyshev interior points, incorporating
    the clamped boundary conditions u(1)=u'(1)=u(-1)=u'(-1)=0.

    Input:
    ncheb: order of Chebyshev polynomials

    Output:
    x:      Interior Chebyshev points (vector of length N - 1)
    D4:     Fourth derivative matrix  (size (N - 1) x (N - 1))

    The code implements two strategies for enhanced
    accuracy suggested by W. Don and S. Solomonoff in
    SIAM J. Sci. Comp. Vol. 6, pp. 1253--1268 (1994).
    The two strategies are (a) the use of trigonometric
    identities to avoid the computation of differences
    x(k)-x(j) and (b) the use of the "flipping trick"
    which is necessary since sin t can be computed to high
    relative precision when t is small whereas sin (pi-t) cannot.

    J.A.C. Weideman, S.C. Reddy 1998.
    """
    if ncheb <= 1:
        raise Exception("ncheb in cheb4c must be strictly greater than 1")

    # initialize dd4
    dm4 = np.zeros((4, ncheb - 1, ncheb - 1))

    # nn1, nn2 used for the flipping trick.
    nn1 = int(np.floor((ncheb + 1) / 2 - 1))
    nn2 = int(np.ceil((ncheb + 1) / 2 - 1))
    # compute theta vector.
    kkk = np.arange(1, ncheb)
    theta = kkk * np.pi / ncheb
    # Compute interior Chebyshev points.
    xch = np.sin(np.pi * (np.linspace(ncheb - 2, 2 - ncheb, ncheb - 1) / (2 * ncheb)))
    # sin theta
    sth1 = np.sin(theta[0:nn1])
    sth2 = np.flipud(np.sin(theta[0:nn2]))
    sth = np.concatenate((sth1, sth2))
    # compute weight function and its derivative
    alpha = sth**4
    beta1 = -4 * sth**2 * xch / alpha
    beta2 = 4 * (3 * xch**2 - 1) / alpha
    beta3 = 24 * xch / alpha
    beta4 = 24 / alpha

    beta = np.vstack((beta1, beta2, beta3, beta4))
    thti = np.tile(theta / 2, (ncheb - 1, 1)).T
    # trigonometric identity
    ddx = 2 * np.sin(thti.T + thti) * np.sin(thti.T - thti)
    # flipping trick
    ddx[nn1:, :] = -np.flipud(np.fliplr(ddx[0:nn2, :]))
    # diagonals of D = 1
    ddx[range(ncheb - 1), range(ncheb - 1)] = 1

    # compute the matrix with entries c(k)/c(j)
    sss = sth**2 * (-1) ** kkk
    sti = np.tile(sss, (ncheb - 1, 1)).T
    cmat = sti / sti.T

    # Z contains entries 1/(x(k)-x(j)).
    # with zeros on the diagonal.
    zmat = np.array(1 / ddx, float)
    zmat[range(ncheb - 1), range(ncheb - 1)] = 0

    # X is same as Z', but with
    # diagonal entries removed.
    xmat = np.copy(zmat).T
    xmat2 = xmat
    for i in range(0, ncheb - 1):
        xmat2[i : ncheb - 2, i] = xmat[i + 1 : ncheb - 1, i]
    xmat = xmat2[0 : ncheb - 2, :]

    # initialize Y and D matrices.
    # Y contains matrix of cumulative sums
    # D scaled differentiation matrices.
    ymat = np.ones((ncheb - 2, ncheb - 1))
    dmat = np.eye(ncheb - 1)
    for ell in range(4):
        # diags
        ymat = np.cumsum(
            np.vstack((beta[ell, :], (ell + 1) * (ymat[0 : ncheb - 2, :]) * xmat)), 0
        )
        # off-diags
        dmat = (
            (ell + 1)
            * zmat
            * (cmat * np.transpose(np.tile(np.diag(dmat), (ncheb - 1, 1))) - dmat)
        )
        # correct the diagonal
        dmat[range(ncheb - 1), range(ncheb - 1)] = ymat[ncheb - 2, :]
        # store in dm4
        dm4[ell, :, :] = dmat
    dd4 = dm4[3, :, :]
    return xch, dd4
