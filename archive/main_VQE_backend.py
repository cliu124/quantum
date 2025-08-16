import numpy as np
from scipy.optimize import minimize

from qiskit_algorithms import VQE, VQD, NumPyEigensolver, optimizers
from qiskit.quantum_info.operators import Operator
from qiskit.quantum_info import SparsePauliOp
#from qiskit.circuit.library import TwoLocal
#from qiskit.circuit.library import NLocal
from qiskit.circuit.library import EfficientSU2
#from qiskit.primitives import Sampler, Estimator
from qiskit_algorithms.state_fidelities import ComputeUncompute
from qiskit import Aer
from qiskit_ibm_runtime import QiskitRuntimeService, Estimator, Session, Options, Sampler
from qiskit.transpiler import PassManager
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime.transpiler.passes.scheduling import (
    ALAPScheduleAnalysis,
    PadDynamicalDecoupling,
)
from qiskit.circuit.library import XGate
import qiskit_ibm_runtime
import matplotlib.pyplot as plt


###simulator
#backend = Aer.get_backend("qasm_simulator")

QiskitRuntimeService.save_account(channel="ibm_quantum", token="3e240c42418c07b80ef72d580d6074ef560da4546167a9d2ad9591c0f845526e71e03f2d0fa393a5302212d30b77e406e744cc55bd7e4e22670f095d1f8791d0",overwrite=True)
service = QiskitRuntimeService(channel='ibm_quantum')
backend = service.least_busy(operational=True, simulator=False)
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

optimizer = optimizers.SLSQP()
ansatz.decompose().draw("mpl")

target = backend.target
pm = generate_preset_pass_manager(target=target, optimization_level=3)
pm.scheduling = PassManager(
    [
        ALAPScheduleAnalysis(durations=target.durations()),
        PadDynamicalDecoupling(
            durations=target.durations(),
            dd_sequences=[XGate(), XGate()],
            pulse_alignment=target.pulse_alignment,
        ),
    ]
)

ansatz_ibm = pm.run(ansatz)
num_params = ansatz.num_parameters
x0 = 2 * np.pi * np.random.random(num_params)


def cost_func(params, ansatz, hamiltonian, estimator):
    """Return estimate of energy from estimator

    Parameters:
        params (ndarray): Array of ansatz parameters
        ansatz (QuantumCircuit): Parameterized ansatz circuit
        hamiltonian (SparsePauliOp): Operator representation of Hamiltonian
        estimator (Estimator): Estimator primitive instance

    Returns:
        float: Energy estimate
    """
    energy = estimator.run(ansatz, hamiltonian, parameter_values=params).result().values[0]
    return energy


def build_callback(ansatz, hamiltonian, estimator, callback_dict):
    """Return callback function that uses Estimator instance,
    and stores intermediate values into a dictionary.

    Parameters:
        ansatz (QuantumCircuit): Parameterized ansatz circuit
        hamiltonian (SparsePauliOp): Operator representation of Hamiltonian
        estimator (Estimator): Estimator primitive instance
        callback_dict (dict): Mutable dict for storing values

    Returns:
        Callable: Callback function object
    """

    def callback(current_vector):
        """Callback function storing previous solution vector,
        computing the intermediate cost value, and displaying number
        of completed iterations and average time per iteration.

        Values are stored in pre-defined 'callback_dict' dictionary.

        Parameters:
            current_vector (ndarray): Current vector of parameters
                                      returned by optimizer
        """
        # Keep track of the number of iterations
        callback_dict["iters"] += 1
        # Set the prev_vector to the latest one
        callback_dict["prev_vector"] = current_vector
        # Compute the value of the cost function at the current vector
        # This adds an additional function evaluation
        current_cost = (
            estimator.run(ansatz, hamiltonian, parameter_values=current_vector).result().values[0]
        )
        callback_dict["cost_history"].append(current_cost)
        # Print to screen on single line
        print(
            "Iters. done: {} [Current cost: {}]".format(callback_dict["iters"], current_cost),
            end="\r",
            flush=True,
        )

    return callback


options = Options()
options.transpilation.skip_transpilation = True
options.execution.shots = 10000

hamiltonian_ibm = Hamil_Qop.apply_layout(ansatz_ibm.layout)

with Session(backend=backend):
    estimator = Estimator(options=options)
    callback = build_callback(ansatz_ibm, hamiltonian_ibm, estimator, callback_dict)
    res = minimize(
        cost_func,
        x0,
        args=(ansatz_ibm, hamiltonian_ibm, estimator),
        method="cobyla",
        callback=callback,
    )

res
all(callback_dict["prev_vector"] == res.x)
callback_dict["iters"] == res.nfev
fig, ax = plt.subplots()
ax.plot(range(callback_dict["iters"]), callback_dict["cost_history"])
ax.set_xlabel("Iterations")
ax.set_ylabel("Cost")
plt.draw()

qiskit_ibm_runtime.version.get_version_info()

# estimator = Estimator(backend=backend)
# sampler = Sampler(backend=backend)
# fidelity = ComputeUncompute(sampler)
# counts = []
# values = []
# steps = []

# def callback(eval_count, params, value, meta, step):
#     counts.append(eval_count)
#     values.append(value)
#     steps.append(step)

# # #Print these to test. 
# # print('estimator')
# # print(estimator)

# # print('ansatz')
# # print(ansatz)

# # print('optimizer')
# # print(optimizer)

# # # Create a circuit that uses this operator as a unitary circuit element
# # vqd = VQD(estimator, fidelity, ansatz, optimizer, k=1,callback=callback)
# # vqd_result = vqd.compute_eigenvalues(operator = Hamil_Qop)
# # vqd_values = vqd_result.eigenvalues
# # print('VQD')
# # print(vqd_values)
# # print(vqd_result)

# # Create a circuit for VQE computation
# vqe = VQE(estimator, ansatz, optimizer)
# vqe_result = vqe.compute_minimum_eigenvalue(operator = Hamil_Qop)
# vqe_values = vqe_result.eigenvalue
# print('VQE')
# print(vqe_values)
# print(vqe_result)
