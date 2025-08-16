from qiskit import Aer, QuantumCircuit, transpile, execute  
from qiskit.circuit.library import EfficientSU2  
from qiskit.quantum_info import Operator  
from qiskit.opflow import PauliSumOp  
import numpy as np  

# Define the Hamiltonian as a PauliSumOp (example: H = Z0 + 0.5*X1)  
from qiskit.opflow import I, X, Z  
H = Z ^ I + 0.5 * (I ^ X)  

# Parameters  
n_qubits = 2  
max_krylov_dim = 5  
backend = Aer.get_backend('aer_simulator_statevector')  

# Define a trial state (e.g., an initial guess for the eigenstate)  
def trial_state():  
    qc = QuantumCircuit(n_qubits)  
    qc.h(0)  
    qc.h(1)  
    return qc  

# Compute the action of H on a state (statevector-based simulation)  
def apply_hamiltonian(statevector, hamiltonian):  
    op_matrix = Operator(hamiltonian).data  
    return np.dot(op_matrix, statevector)  

# Quantum Krylov Subspace Method  
def quantum_krylov(hamiltonian, initial_circuit, max_dim):  
    # Create initial state  
    initial_circuit = transpile(initial_circuit, backend)  
    result = execute(initial_circuit, backend).result()  
    psi0 = result.get_statevector()  

    # Krylov subspace basis  
    krylov_basis = [psi0]  

    for k in range(1, max_dim):  
        new_vec = apply_hamiltonian(krylov_basis[-1], hamiltonian)  

        # Orthogonalize against previous basis vectors  
        for vec in krylov_basis:  
            overlap = np.dot(vec.conj(), new_vec)  
            new_vec -= overlap * vec  

        # Normalize the new vector  
        norm = np.linalg.norm(new_vec)  
        if norm < 1e-10:  
            break  
        new_vec /= norm  

        krylov_basis.append(new_vec)  

    # Build the Hamiltonian matrix in the Krylov basis  
    H_krylov = np.zeros((len(krylov_basis), len(krylov_basis)), dtype=complex)  
    for i, vec_i in enumerate(krylov_basis):  
        for j, vec_j in enumerate(krylov_basis):  
            H_krylov[i, j] = np.dot(vec_i.conj(), apply_hamiltonian(vec_j, hamiltonian))  

    # Diagonalize the Krylov Hamiltonian  
    eigenvalues, eigenvectors = np.linalg.eigh(H_krylov)  

    # Return the smallest eigenvalue (ground state energy)  
    return eigenvalues[0], eigenvectors[:, 0]  

# Run the algorithm  
trial_circuit = trial_state()  
ground_energy, ground_state = quantum_krylov(H, trial_circuit, max_krylov_dim)  

print("Ground State Energy:", ground_energy)
