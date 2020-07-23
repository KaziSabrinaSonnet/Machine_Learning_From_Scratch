#%%
import math
import numpy as np
import matplotlib.pyplot as plt

#%%

def create_cov_matrix(n):
    A = np.random.rand(n, n)
    B= np.transpose(A)
    cov = np.dot(B, A)
    return cov
def create_mean_matrix(n):
    m = np.array(np.arange(1, n*3, 3))
    return m
def multivariate_norm(dimension):
    covariance = create_cov_matrix(dimension)
    mean = create_mean_matrix(dimension)
    sample = 100
    Z = np.random.multivariate_normal(mean, covariance, sample)
    return Z

m_dimensional_gauss = multivariate_norm(40)
mean = (np.mean(m_dimensional_gauss, axis= 0)).tolist()
standard_dev =(np.std(m_dimensional_gauss, axis= 0)).tolist()
list1 = []
for i in range(1, 41):
    list1.append(i)


#%%
plt.plot(list1, standard_dev)
plt.title("Dimension Vs Standard Deviation")
plt.xlabel("Dimension")
plt.ylabel("Standard Deviation")



# %%
plt.plot(list1, mean )
plt.title("Dimension Vs Mean")
plt.xlabel("Dimension")
plt.ylabel("Mean")
# %%
