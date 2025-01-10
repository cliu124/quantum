from qiskit import QuantumCircuit, Aer, execute
from qiskit.quantum_info import SparsePauliOp, Statevector
import numpy as np

def apply_hamiltonian(hamiltonian, statevector):
    """
    Apply the Hamiltonian to a statevector.

    Args:
        hamiltonian (SparsePauliOp): The Hamiltonian.
        statevector (Statevector): The statevector.

    Returns:
        Statevector: The resulting statevector after applying the Hamiltonian.
    """
    #return Statevector(hamiltonian.to_matrix() @ statevector.data)
    return statevector.evolve(hamiltonian.to_operator())

def quantum_krylov_subspace(hamiltonian, initial_state, num_krylov_vectors):
    """
    Implement a quantum Krylov subspace algorithm to approximate the ground state energy of a Hamiltonian.

    Args:
        hamiltonian (SparsePauliOp): The Hamiltonian of the system.
        initial_state (QuantumCircuit): The initial quantum state prepared as a circuit.
        num_krylov_vectors (int): Number of Krylov vectors to generate.

    Returns:
        energies (list): Approximate eigenvalues from the Krylov subspace.
    """
    # Set up the backend
    backend = Aer.get_backend('statevector_simulator')

    # Simulate the initial state
    job = execute(initial_state, backend)
    result = job.result()
    psi_0 = Statevector(result.get_statevector())

    # Initialize Krylov subspace vectors
    krylov_vectors = [psi_0]
    for _ in range(1, num_krylov_vectors):
        # Apply Hamiltonian to the previous vector
        h_psi = apply_hamiltonian(hamiltonian, krylov_vectors[-1])

        # Orthonormalize
        for vec in krylov_vectors:
            overlap = np.vdot(vec.data, h_psi.data)
            h_psi = Statevector(h_psi.data - overlap * vec.data)

        # Normalize the vector
        norm = np.linalg.norm(h_psi.data)
        h_psi = Statevector(h_psi.data / norm)

        krylov_vectors.append(h_psi)

    # Construct the Krylov Hamiltonian (H_k)
    H_k = np.zeros((num_krylov_vectors, num_krylov_vectors), dtype=complex)
    for i, vi in enumerate(krylov_vectors):
        for j, vj in enumerate(krylov_vectors):
            H_k[i, j] = np.vdot(vi.data, apply_hamiltonian(hamiltonian, vj).data)

    # Solve the eigenvalue problem for H_k
    eigvals, eigvecs = np.linalg.eigh(H_k)

    return eigvals.real

# Example usage:
if __name__ == "__main__":
    # Define a sparse Hamiltonian (example: H = X0Z1 + Z0X1)
    paulis = ["XZ", "ZX"]
    coefficients = [1.0, 1.0]
    hamiltonian = SparsePauliOp(paulis, coeffs=coefficients)

    # Prepare an initial state (e.g., |00>)
    initial_state = QuantumCircuit(2)

    # Run the Krylov subspace method
    num_krylov_vectors = 2
    energies = quantum_krylov_subspace(hamiltonian, initial_state, num_krylov_vectors)
    print("Approximate eigenvalues:", energies)

