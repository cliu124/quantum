import numpy as np
import scipy
import h5py

from qiskit.aqua.algorithms import VQE, NumPyEigensolver
from qiskit.chemistry import FermionicOperator
from qiskit import Aer

backend = Aer.get_backend("qasm_simulator")

c1=1
c2=2
c3=3

ff=np.zeros((3,3))
ff[1,1]=2
ff[2,2]=1

n=3

Hamil=np.zeros((n,n))
Hamil[1,1]=c2
Hamil[2,2]=-c3/2+c2
Hamil[0,2]=np.sqrt(2)*c1
Hamil[1,2]=np.sqrt(2)*c1
Hamil[2,0]=np.sqrt(2)*c1
Hamil[2,1]=np.sqrt(2)*c1

vals,vecs=np.linalg.eig(Hamil)

print("Standard Eigenvalues: ")
print(vals)

Hamil_op = FermionicOperator(h1=Hamil)
Hamil_ops = Hamil_op.mapping(map_type='parity', threshold=1e-12)
result = NumPyEigensolver(Hamil_ops,k=int(2**n)).run()

print("Quibit Eigenvalues=")
print(result['eigenvalues'])

vqe = VQE(operator=Hamil_ops)
vqe_result = np.real(vqe.run(backend)['eigenvalue'])
print("VQE Eigenvalue")
print(vqe_result)