import numpy as np
from qiskit_algorithms import VQE, VQD, NumPyEigensolver, optimizers
from qiskit.quantum_info.operators import Operator
from qiskit.quantum_info import SparsePauliOp
#from qiskit.circuit.library import TwoLocal
#from qiskit.circuit.library import NLocal
from qiskit.circuit.library import EfficientSU2
from qiskit.primitives import Sampler, Estimator
from qiskit_algorithms.state_fidelities import ComputeUncompute
from qiskit import Aer
from qiskit_ibm_runtime import QiskitRuntimeService

from poly_diff import cheb4c
from poly_diff import Chebyshev
###simulator
backend = Aer.get_backend("qasm_simulator")

#QiskitRuntimeService.save_account(channel="ibm_quantum", token="3e240c42418c07b80ef72d580d6074ef560da4546167a9d2ad9591c0f845526e71e03f2d0fa393a5302212d30b77e406e744cc55bd7e4e22670f095d1f8791d0",overwrite=True)
#service = QiskitRuntimeService(channel='ibm_quantum')
#backend = service.least_busy(operational=True, simulator=False)
print(backend.name)
#backend = service.get_backend('ibmq_qasm_simulator')

# c1 = -0.5
# c2 = 0.75
# c3 = 2

# n1 = 3


# Hamil = np.zeros((n, n))
# Hamil[1, 1] = c2
# Hamil[2, 2] = -c3 / 2 + c2
# Hamil[0, 2] = c1
# Hamil[1, 2] = c1
# Hamil[2, 0] = c1
# Hamil[2, 1] = c1

###Below is the random Hermitian matrix
n = 2 #number of qubits
N = 2**n #number of matrix size

#construct a Hermitian matrix
#Hamil_real = np.random.randn(N,N)
#Hamil_imag = np.random.randn(N,N)
#Hamil = Hamil_real + np.transpose(Hamil_real) + 1j*(Hamil_imag-np.transpose(Hamil_imag))

#------------------- generate matrix from Rayleigh Benard convection
Pr=1
Ra=1708
kx=2*np.pi/2.016
ky=0
#Construct matrix from Rayleigh Benard convection
ncheb=int(N/2)
ddm = Chebyshev(degree=ncheb + 1).at_order(2)
   # Enforce Dirichlet BCs
dd2 = ddm[1 : ncheb + 1, 1 : ncheb + 1]
xxt, dd4 = cheb4c(ncheb + 1)
D2=dd2*4
D4=dd4*16
I = np.eye(dd4.shape[0])
Laplacian=D2-(kx**2+ky**2)*I
inv_Laplacian=np.linalg.inv(Laplacian)
print(inv_Laplacian)
Laplacian_square=D4-2*(kx**2+ky**2)*D2+(kx**2+ky**2)**2*I
print('Laplacian_square')
print(Laplacian_square)
A11=Pr*inv_Laplacian*Laplacian_square

A12=-(kx**2+ky**2)*inv_Laplacian*Pr*Ra

A11_12=np.concatenate((A11, A12) ,axis=1)
A21_22=np.concatenate((I, Laplacian),axis=1)
Hamil=-np.concatenate([A11_12,A21_22],axis=0)

#--------------------------


###Below is the matrix from Rayleigh-Benard convection, it is normal matrix but not Hermitian


print("(4x4) Hamiltonian")
print(Hamil)
solver = NumPyEigensolver(k=N)
Hamil_Mat = Operator(Hamil, N)
classical_results = solver.compute_eigenvalues(Hamil_Mat)
print("Qubit Op Eigenvalues: ")
print(classical_results.eigenvalues)

eigenvalues,eigenvectors=np.linalg.eig(Hamil)
print("Eigenvalues from numpy:")
print(eigenvalues)

# Create a unitary operator from the hamiltonian matrix
Hamil_Qop = SparsePauliOp.from_operator(Hamil_Mat)
print(Hamil_Qop)

#Old version ansatz, from Kalin
#ansatz = TwoLocal(2, rotation_blocks=["ry", "rz"], entanglement_blocks="cz", reps=1)

#from https://learning.quantum.ibm.com/tutorial/variational-quantum-eigensolver
#work for arbitrary qubit numbers
ansatz = EfficientSU2(Hamil_Qop.num_qubits)

optimizer = optimizers.SLSQP()
ansatz.decompose().draw("mpl")

estimator = Estimator()
sampler = Sampler()
fidelity = ComputeUncompute(sampler)
counts = []
values = []
steps = []

def callback(eval_count, params, value, meta, step):
    counts.append(eval_count)
    values.append(value)
    steps.append(step)

# #Print these to test. 
# print('estimator')
# print(estimator)

# print('ansatz')
# print(ansatz)

# print('optimizer')
# print(optimizer)

# # Create a circuit that uses this operator as a unitary circuit element
# vqd = VQD(estimator, fidelity, ansatz, optimizer, k=1,callback=callback)
# vqd_result = vqd.compute_eigenvalues(operator = Hamil_Qop)
# vqd_values = vqd_result.eigenvalues
# print('VQD')
# print(vqd_values)
# print(vqd_result)

# Create a circuit for VQE computation
vqe = VQE(estimator, ansatz, optimizer)
vqe_result = vqe.compute_minimum_eigenvalue(operator = Hamil_Qop)
vqe_values = vqe_result.eigenvalue
print('VQE')
print(vqe_values)
print(vqe_result)
