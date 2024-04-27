import numpy as np
from qiskit_algorithms import VQE, VQD, NumPyEigensolver, optimizers
from qiskit.quantum_info.operators import Operator
from qiskit.quantum_info import SparsePauliOp
#from qiskit.circuit.library import TwoLocal
#from qiskit.circuit.library import NLocal
from qiskit.circuit.library import EfficientSU2
#from qiskit.primitives import Sampler, Estimator
from qiskit_algorithms.state_fidelities import ComputeUncompute
from qiskit_aer import Aer
from qiskit_ibm_runtime import QiskitRuntimeService, Sampler, Estimator
#MORE IMPORTS
from qiskit.primitives import BackendEstimator, BackendSampler
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

from poly_diff import cheb4c
from poly_diff import Chebyshev

###simulator
#backend = Aer.get_backend("qasm_simulator")
QiskitRuntimeService.save_account(channel="ibm_quantum", token="3e240c42418c07b80ef72d580d6074ef560da4546167a9d2ad9591c0f845526e71e03f2d0fa393a5302212d30b77e406e744cc55bd7e4e22670f095d1f8791d0",overwrite=True)
service = QiskitRuntimeService(channel='ibm_quantum')
#service = QiskitRuntimeService(name="UConn_quantum_credit",instance="UConn")
backend = service.least_busy(operational=True, simulator=False)
print(backend.name)

print(backend.target.keys())

backend = GenericBackendV2(num_qubits = backend.num_qubits, basis_gates = ['id', 'rz', 'sx', 'x', 'ecr'], coupling_map=backend.coupling_map, calibrate_instructions=backend.instruction_schedule_map, dtm=backend.dtm)


###Below is the random Hermitian matrix
n = 2 #number of qubits
N = 2**n #number of matrix size

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
print(Hamil_Qop)
#Old version ansatz, from Kalin
#ansatz = TwoLocal(2, rotation_blocks=["ry", "rz"], entanglement_blocks="cz", reps=1)
#from https://learning.quantum.ibm.com/tutorial/variational-quantum-eigensolver
#work for arbitrary qubit numbers
ansatz = EfficientSU2(Hamil_Qop.num_qubits)

# ADDED PART BELOW

target = backend.target
pm = generate_preset_pass_manager(target=target, optimization_level=3)

ansatz_isa = pm.run(ansatz)
hamiltonian_isa = Hamil_Qop.apply_layout(layout=ansatz_isa.layout)

# ADDED PART ABOVE

optimizer = optimizers.SLSQP()
ansatz.decompose().draw("mpl")
estimator = BackendEstimator(backend=backend)
sampler = BackendSampler(backend=backend)
fidelity = ComputeUncompute(sampler)
counts = []
values = []
steps = []
def callback(eval_count, params, value, meta, step):
    counts.append(eval_count)
    values.append(value)
    steps.append(step)
    print(f"count: {counts[-1]}, value: {values[-1]}, step: {steps[-1]}")
# #Print these to test.
# print('estimator')
# print(estimator)
# print('ansatz')
# print(ansatz)
# print('optimizer')
# print(optimizer)
# # Create a circuit that uses this operator as a unitary circuit element
#vqd = VQD(estimator, fidelity, ansatz_isa, optimizer, k=1,callback=callback)
#vqd_result = vqd.compute_eigenvalues(operator = hamiltonian_isa)
#vqd_values = vqd_result.eigenvalues
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