import numpy as np
from qiskit_algorithms import VQD, NumPyEigensolver, optimizers
from qiskit.quantum_info.operators import Operator
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import TwoLocal
from qiskit.primitives import Sampler, Estimator
from qiskit.algorithms.state_fidelities import ComputeUncompute
from qiskit import Aer

backend = Aer.get_backend("qasm_simulator")

c1 = -0.5
c2 = 0.75
c3 = 2

n1 = 3


n = 4
Hamil = np.zeros((n, n))
Hamil[1, 1] = c2
Hamil[2, 2] = -c3 / 2 + c2
Hamil[0, 2] = c1
Hamil[1, 2] = c1
Hamil[2, 0] = c1
Hamil[2, 1] = c1


print("(4x4) Hamiltonian")
print(Hamil)
solver = NumPyEigensolver(k=4)
Hamil_Mat = Operator(Hamil, 4)
classical_results = solver.compute_eigenvalues(Hamil_Mat)
print("Qubit Op Eigenvalues: ")
print(classical_results.eigenvalues)

# Create a unitary operator from the hamiltonian matrix
Hamil_Qop = SparsePauliOp.from_operator(Hamil_Mat)
print(Hamil_Qop)
ansatz = TwoLocal(2, rotation_blocks=["ry", "rz"], entanglement_blocks="cz", reps=1)

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


# Create a circuit that uses this operator as a unitary circuit element
vqd = VQD(estimator, fidelity, ansatz, optimizer, k=4,callback=callback)
result = vqd.compute_eigenvalues(operator = Hamil_Qop)
vqd_values = result.eigenvalues
print(vqd_values)

