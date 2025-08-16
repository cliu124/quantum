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

###simulator
#backend = Aer.get_backend("qasm_simulator")
QiskitRuntimeService.save_account(channel="ibm_quantum", token="0b0acd48100d4ecf49085db3e01fb7b2cd1201f2ad1daa119c7a86f8ed39ec29c8e416fe1d0ffc40007ed373900cab4148c39c595775b9be4766afd5d71b5926",overwrite=True)
service = QiskitRuntimeService(channel='ibm_quantum')
backend = service.least_busy(operational=True, simulator=False)
print(backend.name)

print(backend.target.keys())

backend = GenericBackendV2(num_qubits = backend.num_qubits, basis_gates = ['id', 'rz', 'sx', 'x', 'ecr'], coupling_map=backend.coupling_map, calibrate_instructions=backend.instruction_schedule_map, dtm=backend.dtm)

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
n = 2
#construct a Hermitian matrix
Hamil_real = np.random.randn(n,n)
Hamil_imag = np.random.randn(n,n)
Hamil = Hamil_real + np.transpose(Hamil_real) + 1j*(Hamil_imag-np.transpose(Hamil_imag))
print("(4x4) Hamiltonian")
print(Hamil)
solver = NumPyEigensolver(k=n)
Hamil_Mat = Operator(Hamil, n)
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