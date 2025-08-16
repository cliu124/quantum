# -*- coding: utf-8 -*-
"""
Created on Sat May 11 20:13:53 2024

@author: chang
"""
import numpy as np
from itertools import permutations
import functools as ft


A=['X']
coeff=[1]

n=4

#generate the label and coefficient of off-diagonal matrices following
# https://quantumcomputing.stackexchange.com/questions/23584/what-is-the-best-way-to-write-a-tridiagonal-matrix-as-a-linear-combination-of-pa
# and https://quantumcomputing.stackexchange.com/questions/23522/how-to-write-the-three-qubit-ghz-state-in-the-pauli-basis/23525#23525
for n_local in range(2,n+1): #iteration to update A_{n+1} using A_n
    A=['I' + x for x in A]
    
    #Here, use i that is the local n as the summation limit. 
    for t in range(0,int(np.floor(n_local/2))+1):
        string='X'*(n_local-2*t)+'Y'*(2*t) #generate the string
        perms = [''.join(p) for p in permutations(string)] #generate the permutation
        perms_t=list(set(perms))
        coeff_t=[(-1)**t/2**(n_local-1)]*len(perms_t)

        for label_ind in range(len(perms_t)):
            if perms_t[label_ind][0] == 'Y':
                coeff_t[label_ind]=-coeff_t[label_ind]
        
        A=A+perms_t
        coeff=coeff+coeff_t
        
I=np.array([[1,0],[0,1]])
X=np.array([[0,1],[1,0]])
Y=np.array([[0,-1j],[1j,0]])
Ham_mat=np.zeros((2**n,2**n))
for label_ind in range(len(A)):
    label = A[label_ind]
    
    #Take the Kronecker product based on the label to construct the basis 
    basis=1
    for char_ind in range(len(label)):
        if label[char_ind] =='X':
            basis = np.kron(basis,X)
        elif label[char_ind] =='Y':
            basis = np.kron(basis,Y)
        elif label[char_ind] =='I':
            basis = np.kron(basis,I)
            
    #construct the Hamiltonian matrix based on the coefficients and the basis        
    Ham_mat = Ham_mat+ coeff[label_ind]*basis
    
    