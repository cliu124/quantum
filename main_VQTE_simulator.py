import numpy as np
from qiskit_algorithms import VQE, VQD, NumPyEigensolver, optimizers, TimeEvolutionProblem, VarQITE, SciPyImaginaryEvolver, VarQRTE, SciPyRealEvolver
from qiskit_algorithms.time_evolvers.variational import ImaginaryMcLachlanPrinciple, RealMcLachlanPrinciple
from qiskit.quantum_info.operators import Operator
from qiskit.quantum_info import SparsePauliOp, Statevector
#from qiskit.circuit.library import TwoLocal
#from qiskit.circuit.library import NLocal
from qiskit.circuit.library import EfficientSU2
from qiskit.primitives import Sampler, Estimator
from qiskit_algorithms.state_fidelities import ComputeUncompute
from qiskit import Aer
from qiskit_ibm_runtime import QiskitRuntimeService
import pylab

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

n = 4

#construct a Hermitian matrix
Hamil_real = np.random.randn(n,n)
Hamil_imag = np.random.randn(n,n)
Hamil = Hamil_real + np.transpose(Hamil_real) + 1j*(Hamil_imag-np.transpose(Hamil_imag))
magnetization = SparsePauliOp([ 'IZ', 'ZI'], coeffs=[1, 1])

Hamil_Mat = Operator(Hamil, n)

# Create a unitary operator from the hamiltonian matrix
Hamil_Qop = SparsePauliOp.from_operator(Hamil_Mat)
print(Hamil_Qop)

#Old version ansatz, from Kalin
#ansatz = TwoLocal(2, rotation_blocks=["ry", "rz"], entanglement_blocks="cz", reps=1)

#from https://learning.quantum.ibm.com/tutorial/variational-quantum-eigensolver
#work for arbitrary qubit numbers
ansatz = EfficientSU2(Hamil_Qop.num_qubits,reps=1)

optimizer = optimizers.SLSQP()
ansatz.decompose().draw("mpl")

#Setup initial condition
init_param_values={}
for i in range(len(ansatz.parameters)):
    init_param_values[ansatz.parameters[i]]=np.pi/2

var_principle = ImaginaryMcLachlanPrinciple()

#setup the final time and the TimeEvolutionProblem
time = 5.0
aux_ops = [Hamil_Qop]
evolution_problem = TimeEvolutionProblem(Hamil_Qop, time, aux_operators=aux_ops)

#main code of VarQITE
print('start running VarQITE')
var_qite = VarQITE(ansatz, init_param_values, var_principle, Estimator())
# an Estimator instance is necessary, if we want to calculate the expectation value of auxiliary operators.
evolution_result = var_qite.evolve(evolution_problem)

#exact solution from scipy
init_state = Statevector(ansatz.assign_parameters(init_param_values))
evolution_problem = TimeEvolutionProblem(Hamil_Qop, time, initial_state=init_state, aux_operators=aux_ops)
exact_evol = SciPyImaginaryEvolver(num_timesteps=501)
sol = exact_evol.evolve(evolution_problem)

h_exp_val = np.array([ele[0][0] for ele in evolution_result.observables])
exact_h_exp_val = sol.observables[0][0].real

print('error between VarQITE and exact solutions:')
print(h_exp_val-exact_h_exp_val)


print('start running VarQRTE')
var_principle = RealMcLachlanPrinciple()

init_state = Statevector(ansatz.assign_parameters(init_param_values))
print(init_state)

aux_ops = [magnetization]
time = 10.0

#running VarQRTE
print('start VarQRTE')
evolution_problem = TimeEvolutionProblem(Hamil_Qop, time, aux_operators=aux_ops)
var_qrte = VarQRTE(ansatz, init_param_values,var_principle, Estimator())
evolution_result_re = var_qrte.evolve(evolution_problem)

#Solution on classical computer
init_circ = ansatz.assign_parameters(init_param_values)
evolution_problem = TimeEvolutionProblem(Hamil_Qop, time, initial_state = init_circ, aux_operators=aux_ops)
rtev = SciPyRealEvolver(1001)
sol = rtev.evolve(evolution_problem)

#get results and compare
mz_exp_val_re = np.array([ele[0][0] for ele in evolution_result_re.observables])
exact_mz_exp_val_re = sol.observables[0][0].real
print(mz_exp_val_re-exact_mz_exp_val_re)
