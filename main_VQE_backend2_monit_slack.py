# basic imports
import numpy as np
from qiskit_algorithms import VQE, VQD, NumPyEigensolver, optimizers
from qiskit.quantum_info.operators import Operator
from qiskit.quantum_info import SparsePauliOp

from qiskit.circuit.library import EfficientSU2
from qiskit import transpile
from qiskit_algorithms.state_fidelities import ComputeUncompute
from qiskit_aer import Aer
from qiskit_ibm_runtime import QiskitRuntimeService, Sampler, Estimator, Options, Session

# loading the IBM Acoount with the Backend
QiskitRuntimeService.save_account(channel="ibm_quantum", token="0b0acd48100d4ecf49085db3e01fb7b2cd1201f2ad1daa119c7a86f8ed39ec29c8e416fe1d0ffc40007ed373900cab4148c39c595775b9be4766afd5d71b5926",overwrite=True)
service = QiskitRuntimeService(channel='ibm_quantum')
backend = service.least_busy(operational=True, simulator=False)


print("The Backend is: ",backend.name)

n = 2
#construct a Hermitian matrix
Hamil_real = np.random.randn(n,n)
Hamil_imag = np.random.randn(n,n)
Hamil = Hamil_real + np.transpose(Hamil_real) + 1j*(Hamil_imag-np.transpose(Hamil_imag))
print("(4x4) Hamiltonian")
print(Hamil)

# Compute the eigenvalues of the matrix using numpy
solver = NumPyEigensolver(k=n)
Hamil_Mat = Operator(Hamil, n)
classical_results = solver.compute_eigenvalues(Hamil_Mat)

# Print the results
print("Qubit Op Eigenvalues: ")
print(classical_results.eigenvalues)

# Compute the eigenvalues of the matrix using numpy
eigenvalues,eigenvectors=np.linalg.eig(Hamil)
print("Eigenvalues from numpy:")
print(eigenvalues)


# Create a unitary operator from the hamiltonian matrix
Hamil_Qop = SparsePauliOp.from_operator(Hamil_Mat)

x = Hamil_Qop.num_qubits
print("Number of Qubits: ",x)
print("Hamil_Qop",Hamil_Qop)

print("-----------------")


ansatz = EfficientSU2(Hamil_Qop.num_qubits)
ansatz.decompose().draw()


from qiskit.transpiler import PassManager
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit.transpiler.passes import (
    ALAPScheduleAnalysis,
    PadDynamicalDecoupling,
    ConstrainedReschedule,
)
from qiskit.circuit.library import XGate

target = backend.target
pm = generate_preset_pass_manager(target=target, optimization_level=2)
pm.scheduling = PassManager(
    [
        ALAPScheduleAnalysis(target=target),
        ConstrainedReschedule(target.acquire_alignment, target.pulse_alignment),
        PadDynamicalDecoupling(target=target, dd_sequence=[XGate(), XGate()], pulse_alignment=target.pulse_alignment),
    ]
)

ansatz_ibm = pm.run(ansatz)

hamiltonian_ibm = Hamil_Qop.apply_layout(ansatz_ibm.layout)
hamiltonian_ibm
num_params = ansatz_ibm.num_parameters
num_params



print(Hamil_Qop)




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


callback_dict = {
    "prev_vector": None,
    "iters": 0,
    "cost_history": [],
}




x0 = 2 * np.pi * np.random.random(num_params)







# To run on local simulator:
#   1. Use the Estimator from qiskit.primitives instead.
#   2. Remove the Session context manager below.
from scipy.optimize import minimize
options = Options()
options.transpilation.skip_transpilation = True
options.execution.shots = 100

session = Session(backend=backend,max_time= "0.5m")
estimator = Estimator(session=session,options=options)
callback = build_callback(ansatz_ibm, hamiltonian_ibm, estimator, callback_dict)
res = minimize( cost_func,
    x0,
    args=(ansatz_ibm, hamiltonian_ibm, estimator),
    method="cobyla",
    callback=callback)

session.close()
print(res)


all(callback_dict["prev_vector"] == res.x)

callback_dict["iters"] == res.nfev

import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.plot(range(callback_dict["iters"]), callback_dict["cost_history"])
ax.set_xlabel("Iterations")
ax.set_ylabel("Cost")
plt.draw()