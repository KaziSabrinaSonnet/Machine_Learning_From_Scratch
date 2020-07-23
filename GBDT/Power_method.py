#%%
import numpy as np 

#%%
def f(C,K,k):
    b_k = np.random.rand(C.shape[0])
    v_j = []
    lamda_j = []
    for i in range(k):
        b_k = np.dot(C, b_k)
        b_k = b_k/np.linalg.norm(b_k)
        v_j.append(b_k)
        temp1 = np.dot(C, b_k)
        eig_val = temp1[0]/b_k[0]
        lamda_j.append(eig_val)
    return v_j[:K],lamda_j[:K]
vj, lamdaj = f(np.array([[0.5, 0.5], [0.2, 0.8]]), 10, 10)
#%%
#Testing
mat = np.array([[0.5, 0.5], [0.2, 0.8]])
[eigs, vecs] = np.linalg.eig(mat)
abs_eigs = list(abs(eigs))
max_abseigs = abs_eigs.index(max(abs_eigs))

# %%
