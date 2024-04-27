### poly_diff.py
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








############### END OF poly_diff.py













import time
import sys
sys.path.append(r'C:\Users\zaidi\Downloads\poly_diff.py')
import poly_diff
# APPROACH WITH GENERIC BACKEND+NOISE
from qiskit_algorithms import VQE, VQD, NumPyEigensolver, optimizers
from qiskit.quantum_info.operators import Operator
from qiskit.quantum_info import SparsePauliOp
n = 2 #number of qubits
N = 2**n #number of matrix size
print(n, " = number of qubits")
print(N, " = number of matrix size")
#------------------- generate matrix from input-output analysis of plane Couette flow
Re=358
kx=1
kz=1
omega=0
#Construct matrix from Rayleigh Benard convection
ncheb=int(N)
ddm = Chebyshev(degree=ncheb + 1).at_order(2)
   # Enforce Dirichlet BCs
D2_bc = ddm[1 : ncheb + 1, 1 : ncheb + 1]

ddm1=Chebyshev(degree=ncheb+1).at_order(1)
D1_bc=ddm1[1 : ncheb+1, 1 : ncheb+1]
xxt, D4_bc = cheb4c(ncheb + 1)
zi=1j

I_bc = np.eye(D4_bc.shape[0])
zero_bc=np.zeros((D4_bc.shape[0],D4_bc.shape[1]))
U_bar=np.diag(xxt)
d_U_bar=I_bc
dd_U_bar=zero_bc
K2=kx**2+kz**2
Laplacian=D2_bc-K2*I_bc
inv_Laplacian=np.linalg.inv(Laplacian)
Laplacian_square=D4_bc-2*(kx**2+kz**2)*D2_bc+(kx**2+kz**2)**2*I_bc
A11=np.matmul(inv_Laplacian,Laplacian_square)/Re+zi*kx*dd_U_bar+np.matmul(inv_Laplacian,np.matmul(U_bar,-zi*kx*Laplacian))
A12=zero_bc
A21=-zi*kx*d_U_bar
A22=-zi*kx*U_bar+Laplacian/Re
A11_12=np.concatenate((A11, A12) ,axis=1)
A21_22=np.concatenate((A21, A22),axis=1)
A=np.concatenate([A11_12,A21_22],axis=0)
B11=np.matmul(inv_Laplacian,-zi*kx*D1_bc)
B12=inv_Laplacian*(-K2)
B13=inv_Laplacian*(-zi*kz*D1_bc)
B21=zi*kz*I_bc
B22=zero_bc
B23=-zi*kx*I_bc
B1=np.concatenate((B11,B12,B13),axis=1)
B2=np.concatenate((B21,B22,B23),axis=1)
B=np.concatenate([B1,B2],axis=0)
Bx=np.concatenate([B11,B21],axis=0)
By=np.concatenate([B12,B22],axis=0)
Bz=np.concatenate([B13,B23],axis=0)
C11=zi*kx*D1_bc/K2
C12=-zi*kz*I_bc/K2
C21=I_bc
C22=zero_bc
C31=zi*kz*D1_bc/K2
C32=zi*kx*I_bc/K2
C1=np.concatenate((C11,C12),axis=1)
C2=np.concatenate((C21,C22),axis=1)
C3=np.concatenate((C31,C32),axis=1)
C=np.concatenate([C1,C2,C3],axis=0)

##----Python version of constructing the integration weight for Chebyshev
D0=np.zeros((N+2,N+2))
num=N+1
vec=np.linspace(0,N+1,N+2)
for j in vec:
    D0[:,int(j)]=np.cos(j*np.pi*vec/num)

inte=np.zeros((N+2,N+2))
one=np.ones((N+2))
for i in np.arange(1,N+2,2):
    inte[:,int(i)-1]=2/(1-(i-1)**2)*one
    
weight_full=np.matmul(inte,np.linalg.inv(D0))
weight_full=weight_full[0,:]
weight_bc=weight_full[1:-1]
Iw_root_bc=np.diag(np.sqrt(weight_bc))

#--------
#only consider the H_ux component
H_unweight_ux=np.matmul(np.matmul(C1,np.linalg.inv(1j*omega-A)),Bx)
#H_ux=np.matmul(np.matmul(Iw_root_bc,H_unweight_ux),np.linalg.inv(Iw_root_bc))
Gradient=np.concatenate([zi*kx*I_bc,D1_bc,zi*kz*I_bc],axis=0)
Iw_root_bc_blk=np.kron(np.eye(3,dtype=int),Iw_root_bc)
H_ux_grad=np.matmul(Iw_root_bc_blk,np.matmul(np.matmul(Gradient,H_unweight_ux),np.linalg.inv(Iw_root_bc)))
# Compute the singular value decomposition (SVD)
u, s, vh = np.linalg.svd(H_ux_grad)

# The largest singular value is the maximum value in the array s
largest_singular_value = np.max(s)

print("Largest Singular Value:", largest_singular_value)

print("This should be the same as VQE output with Hamil", -largest_singular_value**2)
Hamil=-np.matmul(H_ux_grad.conj().T,H_ux_grad)

#-------resolvent from fluid dynamics above

print("Hamiltonian")
print(Hamil)
solver = NumPyEigensolver(k=N)
Hamil_Mat = Operator(Hamil, N)
classical_results = solver.compute_eigenvalues(Hamil_Mat)
print("Qubit Op Eigenvalues: ")
print(classical_results.eigenvalues)
eigenvalues,eigenvectors=np.linalg.eig(Hamil)
print("Eigenvalues from numpy:")
print(eigenvalues)
print("Minimal Eigenvalue from numpy:")
print(np.min(eigenvalues))
# Create a unitary operator from the hamiltonian matrix
Hamil_Qop = SparsePauliOp.from_operator(Hamil_Mat)

from qiskit_algorithms import NumPyMinimumEigensolver

numpy_solver = NumPyMinimumEigensolver()
result = numpy_solver.compute_minimum_eigenvalue(operator=Hamil_Qop)
ref_value = result.eigenvalue.real
print(f"Reference value: {ref_value:.5f}")

# define ansatz and optimizer
from qiskit.circuit.library import TwoLocal
from qiskit_algorithms.optimizers import SLSQP

iterations = 125
ansatz=TwoLocal(rotation_blocks="ry", entanglement_blocks="cz")
# ansatz = EfficientSU2(Hamil_Qop.num_qubits)
optimizer = optimizers.SLSQP()

# define callback
# note: Re-run this cell to restart lists before training
counts = []
values = []


def store_intermediate_result(eval_count, parameters, mean, std):
    counts.append(eval_count)
    values.append(mean)

from qiskit_algorithms.utils import algorithm_globals
from qiskit_aer.primitives import Estimator as AerEstimator

seed = 170
algorithm_globals.random_seed = seed

noiseless_estimator = AerEstimator(
    run_options={"seed": seed, "shots": 1024},
    transpile_options={"seed_transpiler": seed},
)

from qiskit_algorithms import VQE
vqe = VQE(noiseless_estimator, ansatz, optimizer=optimizer, callback=store_intermediate_result)
result = vqe.compute_minimum_eigenvalue(operator=Hamil_Qop)
print(f"VQE on Aer qasm simulator (no noise): {result.eigenvalue.real:.5f}")
print(f"Delta from reference energy value is {(result.eigenvalue.real - ref_value):.5f}")

from qiskit_aer.noise import NoiseModel
from qiskit.providers.fake_provider import GenericBackendV2

coupling_map = [(0, 1), (1, 2), (2, 3), (3, 4)]
device = GenericBackendV2(num_qubits=5, coupling_map=coupling_map, seed=54)

noise_model = NoiseModel.from_backend(device)

print(noise_model)

noisy_estimator = AerEstimator(
    backend_options={
        "method": "density_matrix",
        "coupling_map": coupling_map,
        "noise_model": noise_model,
    },
    run_options={"seed": seed, "shots": 1024},
    transpile_options={"seed_transpiler": seed},
)

counts = []
values = []

vqe.estimator = noisy_estimator

result1 = vqe.compute_minimum_eigenvalue(operator=Hamil_Qop)

print(f"VQE on Aer qasm simulator (with noise): {result1.eigenvalue.real:.5f}")
print(f"Delta from reference energy value is {(result1.eigenvalue.real - ref_value):.5f}")



# ##############################################################################################################################################################
# import sys
# sys.path.append(r'C:\Users\zaidi\Downloads\poly_diff.py')
# import poly_diff
# #APPROACH WITH RUNTIME and fake backends, Hamiltonian same as the one used for the above approach
# import numpy as np
# import warnings

# warnings.filterwarnings("ignore")

# # Pre-defined ansatz circuit and operator class for Hamiltonian
# from qiskit.circuit.library import EfficientSU2
# from qiskit.quantum_info import SparsePauliOp

# # SciPy minimizer routine
# from scipy.optimize import minimize

# # Plotting functions
# import matplotlib.pyplot as plt

# from qiskit_ibm_runtime import QiskitRuntimeService, Session
# from qiskit_ibm_runtime import EstimatorV2 as Estimator
# from qiskit_ibm_runtime.fake_provider import FakeBrisbane
# from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

# #FOR FAKE BACKEND UNCOMMENT THE TWO LINES BELOW
# # fake_brisbane = FakeBrisbane()
# # pm = generate_preset_pass_manager(backend=fake_brisbane, optimization_level=1)

# #FOR NOISE MODEL FROM THE CURRENT NOISE OF THE ACTUAL HARDWARE, USE ALL THE LINES BELOW (220-225)
# from qiskit_aer import AerSimulator
# service = QiskitRuntimeService()
# # Specify a system to use for the noise model
# backend = service.backend("ibm_brisbane")
# fake_brisbane = AerSimulator.from_backend(backend)
# pm = generate_preset_pass_manager(backend=fake_brisbane, optimization_level=1)
# # isa_qc = pm.run(qc)
 
# # # You can use a fixed seed to get fixed results.
# # options = {"simulator": {"seed_simulator": 42}}
# # estimator = Sampler(backend=fake_brisbane, options=options)

# hamiltonian = Hamil_Qop

# ansatz = EfficientSU2(hamiltonian.num_qubits)
# ansatz.decompose().draw("mpl", style="iqp")

# ansatz_isa = pm.run(ansatz)

# hamiltonian_isa = hamiltonian.apply_layout(layout=ansatz_isa.layout)

# def cost_func(params, ansatz, hamiltonian, estimator):
#     """Return estimate of energy from estimator

#     Parameters:
#         params (ndarray): Array of ansatz parameters
#         ansatz (QuantumCircuit): Parameterized ansatz circuit
#         hamiltonian (SparsePauliOp): Operator representation of Hamiltonian
#         estimator (EstimatorV2): Estimator primitive instance

#     Returns:
#         float: Energy estimate
#     """
#     pub = (ansatz, [hamiltonian], [params])
#     result = estimator.run(pubs=[pub]).result()
#     energy = result[0].data.evs[0]

#     return energy

# def build_callback(ansatz, hamiltonian, estimator, callback_dict):
#     """Return callback function that uses Estimator instance,
#     and stores intermediate values into a dictionary.

#     Parameters:
#         ansatz (QuantumCircuit): Parameterized ansatz circuit
#         hamiltonian (SparsePauliOp): Operator representation of Hamiltonian
#         estimator (EstimatorV2): Estimator primitive instance
#         callback_dict (dict): Mutable dict for storing values

#     Returns:
#         Callable: Callback function object
#     """

#     def callback(current_vector):
#         """Callback function storing previous solution vector,
#         computing the intermediate cost value, and displaying number
#         of completed iterations and average time per iteration.

#         Values are stored in pre-defined 'callback_dict' dictionary.

#         Parameters:
#             current_vector (ndarray): Current vector of parameters
#                                       returned by optimizer
#         """
#         # Keep track of the number of iterations
#         callback_dict["iters"] += 1
#         # Set the prev_vector to the latest one
#         callback_dict["prev_vector"] = current_vector
#         # Compute the value of the cost function at the current vector
#         # This adds an additional function evaluation
#         pub = (ansatz, [hamiltonian], [current_vector])
#         result = estimator.run(pubs=[pub]).result()
#         current_cost = result[0].data.evs[0]
#         callback_dict["cost_history"].append(current_cost)
#         # Print to screen on single line
#         print(
#             "Iters. done: {} [Current cost: {}]".format(callback_dict["iters"], current_cost),
#             end="\r",
#             flush=True,
#         )
#         # print(callback)

#     return callback

# callback_dict = {
#     "prev_vector": None,
#     "iters": 0,
#     "cost_history": [],
# }

# num_params = ansatz.num_parameters
# num_params

# x0 = 2 * np.pi * np.random.random(num_params)

# hamiltonian_isa.coeffs = hamiltonian_isa.coeffs.real

# with Session(backend=fake_brisbane) as session:
#     estimator = Estimator(session=session)
#     estimator.options.default_shots = 10_000

#     callback = build_callback(ansatz_isa, hamiltonian_isa, estimator, callback_dict)

#     res = minimize(
#         cost_func,
#         x0,
#         args=(ansatz_isa, hamiltonian_isa, estimator),
#         method="SLSQP",
#         callback=callback,
#     )

# res
