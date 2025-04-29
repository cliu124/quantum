# -*- coding: utf-8 -*-
"""
Created on Sat May 11 20:13:53 2024

@author: chang
"""
import numpy as np
from itertools import permutations
import functools as ft
from qiskit_algorithms import VQE, optimizers
from qiskit.quantum_info.operators import Operator
from qiskit.quantum_info import SparsePauliOp
#from qiskit.circuit.library import TwoLocal
#from qiskit.circuit.library import NLocal
from qiskit.circuit.library import EfficientSU2, TwoLocal
#from qiskit.primitives import Sampler, Estimator
from qiskit_algorithms.state_fidelities import ComputeUncompute
#from qiskit_aer import Aer
from qiskit_ibm_runtime import QiskitRuntimeService
import time
from scipy.sparse.linalg import eigsh


n=2 #number of qubit for one dimension.
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
Hamil_Qop = SparsePauliOp(A, coeff)
print(Hamil_Qop)

#setup classical optimizer
optimizer = optimizers.SLSQP()
#ansatz.decompose().draw("mpl")

counts = []
values = []
#steps = []


def store_intermediate_result(eval_count, parameters, mean, std):
    counts.append(eval_count)
    values.append(mean)
    print(f"count: {counts[-1]}, value: {values[-1]}")

# def callback(eval_count, params, value, meta, step):
#     counts.append(eval_count)
#     values.append(value)
#     steps.append(step)
#     print(f"count: {counts[-1]}, value: {values[-1]}, step: {steps[-1]}")

def cost_func(params, ansatz, Hamil_Qop, estimator):
    """Return estimate of energy from estimator

    Parameters:
        params (ndarray): Array of ansatz parameters
        ansatz (QuantumCircuit): Parameterized ansatz circuit
        Hamil_Qop (SparsePauliOp): Operator representation of Hamiltonian
        estimator (EstimatorV2): Estimator primitive instance
        cost_history_dict: Dictionary for storing intermediate results

    Returns:
        float: Energy estimate
    """
    pub = (ansatz, [Hamil_Qop], [params])
    result = estimator.run(pubs=[pub]).result()
    energy = result[0].data.evs[0]

    cost_history_dict["iters"] += 1
    cost_history_dict["prev_vector"] = params
    cost_history_dict["cost_history"].append(energy)
    print(f"Iters. done: {cost_history_dict['iters']} [Current cost: {energy}]")

    return energy


cost_history_dict = {
    "prev_vector": None,
    "iters": 0,
    "cost_history": [],
}


if quantum=='aer':
    from qiskit.primitives import Sampler, Estimator
    from qiskit_aer import Aer
    #get the quantum backend (hardware or simulator) and then compute
    backend = Aer.get_backend("qasm_simulator")
    
    #QiskitRuntimeService.save_account(channel="ibm_quantum", token="3e240c42418c07b80ef72d580d6074ef560da4546167a9d2ad9591c0f845526e71e03f2d0fa393a5302212d30b77e406e744cc55bd7e4e22670f095d1f8791d0",overwrite=True)
    #service = QiskitRuntimeService(channel='ibm_quantum')
    #backend = service.least_busy(operational=True, simulator=False)
    print(backend.name)
    
    estimator = Estimator()
    sampler = Sampler()
    fidelity = ComputeUncompute(sampler)
    
    #from https://learning.quantum.ibm.com/tutorial/variational-quantum-eigensolver
    #work for arbitrary qubit numbers
    #ansatz = EfficientSU2(Hamil_Qop.num_qubits)
    ansatz = TwoLocal(num_qubits=Hamil_Qop.num_qubits)
    start_time_VQE=time.time()
    vqe = VQE(estimator, ansatz, optimizer,callback=store_intermediate_result)
    #    vqe = VQE(estimator, ansatz, optimizer)

    vqe_result = vqe.compute_minimum_eigenvalue(operator = Hamil_Qop)
    vqe_values = vqe_result.eigenvalue
    
    end_time_VQE=time.time()
    print('VQE')
    print(vqe_result)
    
    print('Minimal eigenvalue from VQE is:')
    print(vqe_values)
    
    print('Computing Time of VQE:')
    print(end_time_VQE-start_time_VQE)
    
elif quantum =='fakebackend':
    
    # define ansatz and optimizer
    from qiskit.circuit.library import TwoLocal
    
    iterations = 125
    ansatz=TwoLocal(rotation_blocks="ry", entanglement_blocks="cz")
    # ansatz = EfficientSU2(Hamil_Qop.num_qubits)
    
    from qiskit_algorithms.utils import algorithm_globals
    from qiskit_aer.primitives import Estimator as AerEstimator
    
    seed = 170
    algorithm_globals.random_seed = seed
    
    noiseless_estimator = AerEstimator(
        run_options={"seed": seed, "shots": 1024},
        transpile_options={"seed_transpiler": seed},
    )
    
    start_time_VQE=time.time()
    vqe = VQE(noiseless_estimator, ansatz, optimizer=optimizer, callback=store_intermediate_result)
    end_time_VQE=time.time()
    #    vqe = VQE(noiseless_estimator, ansatz, optimizer=optimizer)

    result = vqe.compute_minimum_eigenvalue(operator=Hamil_Qop)
    print(f"VQE on Aer qasm simulator (no noise): {result.eigenvalue.real:.5f}")
    print('Computing Time of VQE (no noise):')
    print(end_time_VQE-start_time_VQE)
    
    from qiskit_aer.noise import NoiseModel
    from qiskit.providers.fake_provider import GenericBackendV2
    
    #coupling_map = [(0, 1), (1, 2), (2, 3), (3, 4)]
    device = GenericBackendV2(num_qubits=n, seed=54)
    
    noise_model = NoiseModel.from_backend(device)
    
    print(noise_model)
    
    noisy_estimator = AerEstimator(
        backend_options={
            "method": "density_matrix",#"coupling_map": coupling_map, delete this line for coupling map
            "noise_model": noise_model,
        },
        run_options={"seed": seed, "shots": 1024},
        transpile_options={"seed_transpiler": seed},
    )
    
    vqe.estimator = noisy_estimator
    
    start_time_VQE=time.time()
    result1 = vqe.compute_minimum_eigenvalue(operator=Hamil_Qop)
    end_time_VQE=time.time()
    
    print(f"VQE on Aer qasm simulator (with noise): {result1.eigenvalue.real:.5f}")
    print('Computing Time of VQE (with noise):')
    print(end_time_VQE-start_time_VQE)


elif quantum =='backend1':
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    from qiskit.primitives import BackendEstimator, BackendSampler 
    from qiskit_ibm_runtime import EstimatorV2, SamplerV2, Session
    from scipy.optimize import minimize

    #This is using hardware from IBM
    #instance of 10 mins free time per month
    service = QiskitRuntimeService(channel="ibm_cloud",token="",instance="crn:v1:bluemix:public:quantum-computing:us-east:a/f84b32721d2f4ddfa35b85de7b1230a5:8be1d03e-b2db-4077-8702-2d0958b27252::")

    #Quantum credit program
    service = QiskitRuntimeService(channel="ibm_cloud",token="",instance="crn:v1:bluemix:public:quantum-computing:us-east:a/f84b32721d2f4ddfa35b85de7b1230a5:9852b119-18cc-4f58-ac25-2b008a7aeb2f::")
    backend = service.least_busy(operational=True, simulator=False)

    ansatz = EfficientSU2(Hamil_Qop.num_qubits)
    #ansatz.decompose().draw("mpl", style="iqp")
    
    # ADDED PART BELOW
    target = backend.target
    pm = generate_preset_pass_manager(target=target, optimization_level=3)
    
    ansatz_isa = pm.run(ansatz)
    #ansatz_isa.draw(output="mpl", idle_wires=False, style="iqp")
    hamiltonian_isa = Hamil_Qop.apply_layout(layout=ansatz_isa.layout)
    estimator = EstimatorV2(backend)
    #sampler = SamplerV2(backend)
    #fidelity = ComputeUncompute(sampler)

    start_time_VQE=time.time()
    
    num_params = ansatz.num_parameters
    print('num_params=',num_params)
    x0 = 2 * np.pi * np.random.random(num_params)
    print('x0=',x0)
    
    with Session(backend=backend) as session:
        estimator = EstimatorV2(mode=session)
        estimator.options.default_shots = 10000
    
        res = minimize(
            cost_func,
            x0,
            args=(ansatz_isa, hamiltonian_isa, estimator),
            method="cobyla",
        )
    #This VQE is not compatible with the qiskit>=1.4.1, and qiskit-ibm-runtime>0.38
    #vqe = VQE(estimator, ansatz, optimizer,callback=store_intermediate_result)
    #vqe_result = vqe.compute_minimum_eigenvalue(operator = Hamil_Qop)
    #vqe_values = vqe_result.eigenvalue
    
    end_time_VQE=time.time()
    print('VQE')
    print(vqe_result)
    
    print('Minimal eigenvalue from VQE is:')
    print(vqe_values)
    
    print('Computing Time of VQE:')
    print(end_time_VQE-start_time_VQE)

#-----------------------------
#Below is convert to classical computation and check
#convert the pauli matrices encoding back to the classical expression of Hamiltonian

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