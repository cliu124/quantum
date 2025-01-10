from qiskit import QuantumCircuit, Aer, execute
from qiskit.quantum_info import SparsePauliOp, Statevector
import numpy as np
import time
from itertools import permutations


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
    
    n=10 #number of qubit for one dimension.
    classical=1
    quantum='aer' #['aer','backend1','fackbackend']
    dimension =1 #1, 2, 3, The physical dimension of heat equation. The 
    #The total qubits for the computation is n* dimension, and Hamiltonian will be in size 2**(n*dimension) * 2**(n*dimension)
    
    N=2**n #matrix size
    h=1/(N+1) #step size
    
    start_time_pauli=time.time()
    
    A=['X']
    coeff=[1]
    #generate the label and coefficient of off-diagonal matrices following
    # https://quantumcomputing.stackexchange.com/questions/23584/what-is-the-best-way-to-write-a-tridiagonal-matrix-as-a-linear-combination-of-pa
    # and https://quantumcomputing.stackexchange.com/questions/23522/how-to-write-the-three-qubit-ghz-state-in-the-pauli-basis/23525#23525
    for n_local in range(2,n+1): #iteration to update A_{n+1} using A_n
        A=['I' + x for x in A]
        
        #Here, use i that is the local n as the summation limit. 
        for t in range(0,int(np.floor(n_local/2))+1):
            string='X'*(n_local-2*t)+'Y'*(2*t) #generate the string
            
            #generate perms_t that is the list of permutation of string without repeatness
            perms = [''.join(p) for p in permutations(string)] #generate the permutation
            perms_t=list(set(perms))
            
            #get the coefficient for corresponding coefficient
            #This will encode B_n in https://quantumcomputing.stackexchange.com/questions/23584/what-is-the-best-way-to-write-a-tridiagonal-matrix-as-a-linear-combination-of-pa
            coeff_t=[(-1)**t/2**(n_local-1)]*len(perms_t)
    
            #This will encode (X \kron I)B_n (X \kron I). Only if the first gate is Y gate, we need to add a minus sign for coefficient
            for label_ind in range(len(perms_t)):
                if perms_t[label_ind][0] == 'Y':
                    coeff_t[label_ind]=-coeff_t[label_ind]
            
            #add them to the label list and coefficient list. 
            A=A+perms_t
            coeff=coeff+coeff_t
         
    #Add the coefficient of the off-diagonal matrix        
    coeff=[ (1/h**2)*i for i in coeff]        
    
    #Add the diagonal component of the tridiagonal matrix after finite difference        
    A=A+['I'*n]
    coeff=coeff+ [(-2/h**2)]
    
    #if dimension ==1, do not need to do anything. 
    if dimension ==2:
        #Laplacian in 2D will be 
        # I \kron D2 + D2 \kron I
        
        Axx=['I'*n + x for x in A]
        Ayy=[x+'I'*n for x in A]
        A=Axx+Ayy
        coeff=coeff+coeff
    elif dimension ==3:
        #Laplacian in 3D will be 
        # I \kron I \kron D2+ I \kron D2 \kron I + D2 \kron I \kron I 
        Axx=['I'*(2*n) + x for x in A]
        Ayy=['I'*n +x+ 'I'*n for x in A]
        Azz=[x+ 'I'*(2*n) for x in A]
        A=Axx+Ayy+Azz
        coeff=coeff+coeff+coeff
        
        
    #reverse the sign for all coefficients as VQE only compute minimal eigenvalue, but we want to solve maximum eigenvalue
    coeff=[ -i for i in coeff]        
    
    end_time_pauli=time.time()
    print('Time for construct Pauli decomposition of tridiagonal matrices')
    print(end_time_pauli-start_time_pauli)
    #-------------------
    #construct the Hamiltonian operator using labels (A) and coeff
    hamiltonian = SparsePauliOp(A, coeff)
    
    
    #print(Hamil_Qop)
    
    #paulis = ["XZ", "ZX"]
    #coefficients = [1.0, 1.0]
    #hamiltonian = SparsePauliOp(paulis, coeffs=coefficients)

    # Prepare an initial state (e.g., |00>)
    initial_state = QuantumCircuit(n*dimension)

    # Run the Krylov subspace method
    num_krylov_vectors = 2
    energies = quantum_krylov_subspace(hamiltonian, initial_state, num_krylov_vectors)
    print("Approximate eigenvalues:", energies)


    
    if classical:#If 1, then convert back to classical Hamiltonian matrix and use numpy to compute eigenvalues
            
        #Four different Pauli basis
        I=np.array([[1,0],[0,1]])
        X=np.array([[0,1],[1,0]])
        Y=np.array([[0,-1j],[1j,0]])
        Z=np.array([[1,0],[0,-1]])
        
        #initialize the zero Hamiltonian matrix
        Ham_mat=np.zeros((2**(n*dimension),2**(n*dimension)))
        
        start_time_classical_pauli=time.time()
        for label_ind in range(len(A)):
            label = A[label_ind]
            
            #Take the Kronecker product based on the label to construct the basis 
            basis=1
            for char_ind in range(len(label)):
                if label[char_ind] =='I':
                    basis = np.kron(basis,I)
                elif label[char_ind] =='X':
                    basis = np.kron(basis,X)
                elif label[char_ind] =='Y':
                    basis = np.kron(basis,Y)
                elif label[char_ind] =='Z':
                    basis = np.kron(basis,Z)
                    
            #construct the Hamiltonian matrix based on the coefficients and the basis        
            Ham_mat = Ham_mat+ coeff[label_ind]*basis
            
        end_time_classical_pauli=time.time()
        print('Time for constructing Hamiltonian matrix from pauli decomposition')
        print(end_time_classical_pauli-start_time_classical_pauli)    
        
        start_time_numpy_eig=time.time()    
        eigenvalues,eigenvectors=np.linalg.eig(Ham_mat)
        end_time_numpy_eig=time.time()
        print("Eigenvalues from numpy.linalg.eig:")
        print(eigenvalues)
        print("Minimal Eigenvalue from numpy:")
        print(np.min(eigenvalues))  
        print("Time for classical eig solver in numpy:")
        print(end_time_numpy_eig-start_time_numpy_eig)
    
        # #This scipy even just compute one eigenvalue is not faster than numpy solver for large scale matrix.
        # start_time_scipy_eigsh=time.time()
        # eigenvalues_eigsh, eigenvectors_eigsh = eigsh(Ham_mat, k=1,which='SA')
        # end_time_scipy_eigsh=time.time()
        # print("Minimal eignevalue from scipy.linalg.eigsh:")
        # print(eigenvalues_eigsh)
        # print("Time for classical eigsh solver in scipy:")
        # print(end_time_scipy_eigsh-start_time_scipy_eigsh)
    
    print("Analytical solution for the heat equation (D=1):")
    print(np.pi**2*dimension)
