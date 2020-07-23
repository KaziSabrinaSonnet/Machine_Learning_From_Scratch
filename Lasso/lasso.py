#%%
# all the packages you need
from __future__ import division
import sys
import numpy as np
import time
import scipy.io as io
import scipy.sparse as sparse
import matplotlib.pyplot as plt
import scipy.sparse.linalg
%matplotlib inline

# %%
# synthetic data generator
# n is number of samples, d is number of dimensions, k is number of nonzeros in w, sigma is std of noise, 
# X is a n x d data matrix, y=Xw+w_0+noise is a n-dimensional vector, w is the true weight vector, w0 is true intercept
def DataGenerator(n = 50, d = 75, k = 5, sigma = 1.0, w0 = 0.0, seed = 256):
    
    np.random.seed(seed)
    X = np.random.normal(0,1,(n,d))
    w = np.random.binomial(1,0.5,k)
    noise = np.random.normal(0,sigma,n)
    w[w == 1] = 10.0
    w[w == 0] = -10.0
    w = np.append(w, np.zeros(d - k))
    y = X.dot(w) + w0 + noise
    return (X, y, w, w0)
#%%
def Initialw(X, y):

    n, d = X.shape
    # increment X
    if sparse.issparse(X):
        XI = sparse.hstack((X, np.ones(n).reshape(n,1)))
    else:
        XI = np.hstack((X, np.ones(n).reshape(n,1)))

    if sparse.issparse(X):
        if n >= d:
            w = sparse.linalg.lsqr(XI, y)[0]
        else:
            w = sparse.linalg.inv(XI.T.dot(XI) + 1e-3 * sparse.eye(d+1)).dot(XI.T.dot(y))
            w = w.T
    else:
        if n >= d:
            w = np.linalg.lstsq(XI, y)[0]
        else:
            w = np.linalg.inv(XI.T.dot(XI) + 1e-3 * np.eye(d+1)).dot(XI.T.dot(y))
 
    return (w[:d], w[d])

# %%
# Helper and example function of sparse matrix operation for Problem 2.5
# W: a scipy.sparse.csc_matrix
# x: a vector with length equal to the number of columns of W
# In place change the data stored in W,
# so that every row of W gets element-wise multiplied by x
def cscMatInplaceEleMultEveryRow(W, x):
    indptr = W.indptr
    last_idx = indptr[0]
    for col_id, idx in enumerate(indptr[1:]):
        if idx == last_idx:
            continue
        else:
            W.data[last_idx:idx] *= x[col_id]
            last_idx = idx
#%%
# Problem 2.1
# TODO: coordinate descent of lasso, note lmda stands for lambda

def lasso(X, y, lmda = 10.0, epsilon = 1.0e-2, max_iter = 100, draw_curve = False):
    n, m = X.shape 
    w, w0 = Initialw(X, y)
    iteration = 0
    new_theta = np.zeros(w.shape)
    ob = []
    for i in range(max_iter):
        prev_theta = new_theta
        iteration = iteration +1
        theta_list = []
        theta_not_list = []
        for j in range (m):
            curr_X = np.delete(X,j, axis= 1)
            curr_w = np.delete(w,j, axis= 0)
            rk = y- np.dot(curr_X, curr_w)
            ak = np.dot(X[:, j], X[:, j])
            ck = np.dot(rk, X[:, j] )
            if ck< -lmda:
                w[j] = (ck+lmda)/ak
            elif abs(ck)<= lmda:
                w[j] = 0
            elif ck>lmda:
                w[j] = (ck-lmda)/ak
        w0 = (1/n)*(y - np.dot(X, w) )    
        objective_function = 0.5 * (np.dot((np.dot(X, w)+ w0 -y), (np.dot(X, w)+ w0 -y))) + lmda * np.sum(np.abs(w))
        ob.append(objective_function)
        new_theta = w.tolist()
        diff = abs(max(a - b for a, b in zip(new_theta, prev_theta)))
        if diff<= epsilon:
            draw_curve = True
            plt.plot(ob)
            plt.title("Iteration Vs Objective Function")
            plt.xlabel("Iteration")
            plt.ylabel("Objective Function")
            return w, w0
    draw_curve = True
    plt.plot(ob)
    plt.title("Iteration Vs Objective Function")
    plt.xlabel("Iteration")
    plt.ylabel("Objective Function")
    return w, w0
X, y, w, w0 = DataGenerator(n = 50, d = 75, k = 5, sigma = 10.0, w0 = 0.0, seed = 256)
theta_k, theta_not= lasso(X, y, lmda = 10.0, epsilon = 1.0e-2, max_iter = 100, draw_curve = False)
#For report 
non_zero = np.nonzero(theta_k)
# %%
def Evaluate(X, y, w, w0, w_true, w0_true):
    w1= w.tolist()
    w_true1 = w_true.tolist()
    count= 0
    for i in range(len(w)):
        if w1[i]!= 0 and w_true1[i]!= 0:
            count = count+1
        else:
            count = count
    if ((np.count_nonzero(w))!=0):
        precision_w = count/np.count_nonzero(w)
    else:
        precision_w = count/0.0000001
    recall_w = count/np.count_nonzero(w_true)
    sparsity_w = np.count_nonzero(w)
    pred = np.dot(X, w)
    rmse = np.sqrt(np.square(np.subtract(y, pred)).mean()) 
    return (rmse, sparsity_w, precision_w, recall_w)
# %%
# Problem 2.2
# TODO: apply your evaluation function to compute precision (of w), recall (of w), sparsity (of w) and training RMSE
X, y, w_true, w0_true = DataGenerator(n = 50, d = 75, k = 5, sigma = 1.0, w0 = 0.0, seed = 256)
wl, wl0= lasso(X, y, lmda = 10.0, epsilon = 1.0e-2, max_iter = 100, draw_curve = False)
rmse, sparsity_w, precision_w, recall_w = Evaluate(X, y, wl, wl0, w_true, w0_true)

#%%
# Problem 2.3
# TODO: compute a lasso solution path, draw the path(s) in a 2D plot
def LassoPath(X, y):
    y_bar = np.mean(y) 
    lmda_max_cal = np.dot((y-y_bar), X)
    lmda_max = max(lmda_max_cal)
    Lmda = np.linspace(0,lmda_max,50)
    theta_list = []
    theta_not_list = []
    for item in Lmda:
        theta, theta_not = lasso(X, y, lmda = item, epsilon = 1.0e-2, max_iter = 100, draw_curve = False)
        theta_list.append(theta)
        theta_not_list.append(theta_not)
    W = np.stack(theta_list).T
    W0 = np.stack(theta_not_list).T
    return (W, W0, Lmda)

#%%
# Problem 2.3
# TODO: evaluate a given lasso solution path, draw plot of precision/recall vs. lambda
def EvaluatePath(X, y, W, W0, w_true, w0_true, Lmda):
    RMSE = []
    Sparsity = []
    Precision = []
    Recall = []
    for item in Lmda:
        theta, theta_not = lasso(X, y, lmda = item, epsilon = 1.0e-2, max_iter = 100, draw_curve = False)
        rmse, sparsity_w, precision_w, recall_w = Evaluate(X, y, theta, theta_not, w_true, w0_true)
        RMSE.append(rmse)
        Sparsity.append(sparsity_w)
        Precision.append(precision_w)
        Recall.append(recall_w)
    return (RMSE, Sparsity, Precision, Recall)
X, y, w_true, w0_true = DataGenerator(n = 50, d = 75, k = 5, sigma = 1.0, w0 = 0.0, seed = 256)
W, W0, Lmda = LassoPath(X, y)
RMSE, Sparsity, Precision, Recall = EvaluatePath(X, y, W, W0, w_true, w0_true, Lmda)

#%%
# Problem 2.3
# TODO: draw lasso solution path and precision/recall vs. lambda curves
X, y, w_true, w0_true = DataGenerator(n=50, d=75, k=5, sigma=1.0)
W, W0, Lmda = LassoPath(X, y)
size,_ = W.shape
plt.figure(figsize = (12,8))
for i in range(size):
    if i == 0 or i == 1 or i == 2 or i == 3 or i == 4:
        plt.plot(Lmda, W[i], 'b')
    else:
        plt.plot(Lmda, W[i], 'r')
plt.xlabel('Lambdas')
plt.ylabel('Coefficients')
plt.title('Lasso Paths')
plt.legend()
plt.axis('tight')
#%%
plt.plot(Lmda.tolist(), Precision, label="precision")
plt.xlabel("lambda")
plt.ylabel("Precision")
axis = plt.gca()
axis.set_xlim(axis.get_xlim()[::-1])
plt.show()


#%%
# Problem 2.3
# TODO: try a larger std sigma = 10.0
X, y, w_true, w0_true = DataGenerator(n=50, d=75, k=5, sigma=10.0)
W, W0, Lmda = LassoPath(X, y)
size,_ = W.shape
plt.figure(figsize = (12,8))
for i in range(size):
    if i == 0 or i == 1 or i == 2 or i == 3 or i == 4:
        plt.plot(Lmda, W[i], 'b')
    else:
        plt.plot(Lmda, W[i], 'r')
plt.xlabel('Lambdas')
plt.ylabel('Coefficients')
plt.title('Lasso Paths')
plt.legend()
plt.axis('tight')
#%%
# Problem 2.4
# TODO: try another 5 different choices of (n,d) 
# draw lasso solution path and precision/recall vs. lambda curves, use them to estimate the lasso sample complexity
def which_lamda(R, P, L, i, j):
    R_P = []
    for m in range(len(R)):
        if R[m]== i and P[m]==j:
            R_P.append(L[m])
    return R_P
#%%
X1, y1, w_true1, w0_true1 = DataGenerator(n=50, d=150, k=5, sigma=1.0)
W1, W01, Lmda1 = LassoPath(X1, y1)
RMSE1, Sparsity1, Precision1, Recall1 = EvaluatePath(X1, y1, W1, W01, w_true1, w0_true1, Lmda1)

#%%
plt.plot(Recall1, Precision1)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title('Precision Vs Recall (n = 50;m = 150)')
plt.show()
R_P1= which_lamda(Recall1, Precision1, Lmda1, 1, 1)
size1,_ = W1.shape
plt.figure(figsize = (12,8))
for i in range(size1):
    if i == 0 or i == 1 or i == 2 or i == 3 or i == 4:
        plt.plot(Lmda1, W1[i], 'b')
    else:
        plt.plot(Lmda1, W1[i], 'r')
plt.xlabel('Lambdas')
plt.ylabel('Coefficients')
plt.title('Lasso Paths for (n = 50;m = 150)')
plt.legend()
plt.axis('tight')
#%%
X2, y2, w_true2, w0_true2 = DataGenerator(n=50, d=75, k=5, sigma=1.0)
W2, W02, Lmda2 = LassoPath(X2, y2)
RMSE2, Sparsity2, Precision2, Recall2 = EvaluatePath(X2, y2, W2, W02, w_true2, w0_true2, Lmda2)
#%%
plt.plot(Recall2, Precision2)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title('Precision Vs Recall (n = 50;m = 75)')
plt.show()
R_P2= which_lamda(Recall2, Precision2, Lmda2, 1, 1)
size2,_ = W2.shape
plt.figure(figsize = (12,8))
for i in range(size2):
    if i == 0 or i == 1 or i == 2 or i == 3 or i == 4:
        plt.plot(Lmda2, W2[i], 'b')
    else:
        plt.plot(Lmda2, W2[i], 'r')
plt.xlabel('Lambdas')
plt.ylabel('Coefficients')
plt.title('Lasso Paths for (n = 50;m = 75)')
plt.legend()
plt.axis('tight')
#%%
X3, y3, w_true3, w0_true3 = DataGenerator(n=50, d=1000, k=5, sigma=1.0)
W3, W03, Lmda3 = LassoPath(X3, y3)
RMSE3, Sparsity3, Precision3, Recall3 = EvaluatePath(X3, y3, W3, W03, w_true3, w0_true3, Lmda3)
#%%
plt.plot(Recall3, Precision3)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title('Precision Vs Recall (n = 50;m = 1000)')
plt.show()
R_P3= which_lamda(Recall3, Precision3, Lmda3, 1, 0.8333333333333334)
size3,_ = W3.shape
plt.figure(figsize = (12,8))
for i in range(size3):
    if i == 0 or i == 1 or i == 2 or i == 3 or i == 4:
        plt.plot(Lmda3, W3[i], 'b')
    else:
        plt.plot(Lmda3, W3[i], 'r')
plt.xlabel('Lambdas')
plt.ylabel('Coefficients')
plt.title('Lasso Paths for (n = 50;m = 1000)')
plt.legend()
plt.axis('tight')
#%%
X4, y4, w_true4, w0_true4 = DataGenerator(n=100, d=75, k=5, sigma=1.0)
W4, W04, Lmda4 = LassoPath(X4, y4)
RMSE4, Sparsity4, Precision4, Recall4 = EvaluatePath(X4, y4, W4, W04, w_true4, w0_true4, Lmda4)
#%%
plt.plot(Recall4, Precision4)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title('Precision Vs Recall (n = 100;m = 75)')
plt.show()
R_P4= which_lamda(Recall4, Precision4, Lmda4, 1, 1)
size4,_ = W4.shape
plt.figure(figsize = (12,8))
for i in range(size4):
    if i == 0 or i == 1 or i == 2 or i == 3 or i == 4:
        plt.plot(Lmda4, W4[i], 'b')
    else:
        plt.plot(Lmda4, W4[i], 'r')
plt.xlabel('Lambdas')
plt.ylabel('Coefficients')
plt.title('Lasso Paths for (n = 100;m = 75)')
plt.axis('tight')
#%%
X5, y5, w_true5, w0_true5 = DataGenerator(n=100, d=150, k=5, sigma=1.0)
W5, W05, Lmda5 = LassoPath(X5, y5)
RMSE5, Sparsity5, Precision5, Recall5 = EvaluatePath(X5, y5, W5, W05, w_true5, w0_true5, Lmda5)
#%%
plt.plot(Recall5, Precision5)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title('Precision Vs Recall (n = 100;m = 150)')
plt.show()
R_P5= which_lamda(Recall5, Precision5, Lmda5, 1, 1)
size5,_ = W5.shape
plt.figure(figsize = (12,8))
for i in range(size5):
    if i == 0 or i == 1 or i == 2 or i == 3 or i == 4:
        plt.plot(Lmda5, W5[i], 'b')
    else:
        plt.plot(Lmda5, W5[i], 'r')
plt.xlabel('Lambdas')
plt.ylabel('Coefficients')
plt.title('Lasso Paths for (n = 100;m = 150)')
plt.axis('tight')
# %%
X6, y6, w_true6, w0_true6 = DataGenerator(n=100, d=1000, k=5, sigma=1.0)
W6, W06, Lmda6 = LassoPath(X6, y6)
RMSE6, Sparsity6, Precision6, Recall6 = EvaluatePath(X6, y6, W6, W06, w_true6, w0_true6, Lmda6)
#%%
plt.plot(Recall6, Precision6)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title('Precision Vs Recall (n = 100;m = 1000)')
plt.show()
R_P6= which_lamda(Recall6, Precision6, Lmda6, 1, 1)
size6,_ = W6.shape
plt.figure(figsize = (12,8))
for i in range(size6):
    if i == 0 or i == 1 or i == 2 or i == 3 or i == 4:
        plt.plot(Lmda6, W6[i], 'b')
    else:
        plt.plot(Lmda6, W6[i], 'r')
plt.xlabel('Lambdas')
plt.ylabel('Coefficients')
plt.title('Lasso Paths for (n = 100;m = 1000)')
plt.axis('tight')
#%%
from scipy.sparse import csc_matrix
def lasso_sparse(X, y, lmda = 10.0, epsilon = 1.0e-2, max_iter = 100, draw_curve = False):
    n, m = X.shape 
    w, w0 = Initialw(X, y)
    new_theta = np.zeros(w.shape)
    for i in range(max_iter):
        prev_theta = new_theta
        theta_list = []
        theta_not_list = []
        for j in range (m):
            Selector1 = [x for x in range(X.shape[1]) if x != j]
            curr_X = X[:, Selector1]
            Selector2 = [x for x in range(w.shape[0]) if x != j]
            curr_w = w[Selector2]
            rk = y - curr_X.dot(curr_w)
            rk = sparse.csc_matrix(rk)
            ak = X[:, j].T.dot(X[:, j])
            ck = rk.dot(X[:, j])
            if ck< -lmda:
                ck_lambda = sparse.csc_matrix(np.ones((ck.shape[0], ck.shape[1] ))*lmda + ck)
                w[j] = ck_lambda/ak
            elif abs(ck)<= lmda:
                w[j] = 0
            elif ck>lmda:
                ck_lambda = sparse.csc_matrix(ck-np.ones((ck.shape[0], ck.shape[1] ))*lmda)
                w[j] = ck_lambda/ak
        w0 = (1/n)*(y - X.dot(w))    
        new_theta = w.tolist()
        diff = abs(max(a - b for a, b in zip(new_theta, prev_theta)))
        if diff<= epsilon:
            return w, w0
    return w, w0

#%%
# dense to sparse
from numpy import array
from scipy.sparse import csr_matrix
# create dense matrix
X, y, w, w0 = DataGenerator(n = 50, d = 75, k = 5, sigma = 10.0, w0 = 0.0, seed = 256)
S = sparse.csr_matrix(X)
w, w0 = lasso_sparse(S, y, lmda = 10.0, epsilon = 1.0e-2, max_iter = 100, draw_curve = False)

#%%
# Problem 2.5: predict reviews' star on Yelp
# data parser reading yelp data
def DataParser(Xfile, yfile, nfile, train_size = 30000, valid_size = 5000):

    # read X, y, feature names from file
    fName = open(nfile).read().splitlines()
    y = np.loadtxt(yfile, dtype=np.int)
    if Xfile.find('mtx') >= 0:
        # sparse data
        X = io.mmread(Xfile).tocsc()
    else:
        # dense data
        X = np.genfromtxt(Xfile, delimiter=",")

    # split training, validation and test set
    X_train = X[0 : train_size,:]
    y_train = y[0 : train_size]
    X_valid = X[train_size : train_size + valid_size,:]
    y_valid = y[train_size : train_size + valid_size]
    X_test = X[train_size + valid_size : np.size(X,0),:]
    y_test = y[train_size + valid_size : np.size(y,0)]

    return (X_train, y_train, X_valid, y_valid, X_test, y_test, fName)
#%%
X_train, y_train, X_valid, y_valid, X_test, y_test, fName = DataParser('star_data.mtx', 'star_labels.txt', 'star_features.txt', train_size = 30000, valid_size = 5000)
#%%
# Problem 2.5: predict reviews' star on Yelp
# TODO: evaluation funtion that computes the lasso path, evaluates the result, and draws the required plots
def Validation(X_train, y_train, X_valid, y_valid):
    lamda_max = np.max(np.abs(X_train.T.dot(y_train-np.mean(y_train))))
    Lmda = np.linspace(0.1*lamda_max,lamda_max,20)
    w_lasso = []
    w0_lasso= []
    for item in Lmda:
        theta, theta_not = lasso_sparse(X_train, y_train, lmda = item, epsilon = 1.0e-2, max_iter = 100, draw_curve = False)
        w_lasso.append(theta)
        w0_lasso.append(theta_not)
    return (w_lasso, w0_lasso, Lmda)
w_lasso, w0_lasso, Lmda = Validation(X_train, y_train, X_valid, y_valid)
#%%
W = np.stack(w_lasso).T
W0 = np.stack(w0_lasso).T
size,_ = W.shape
plt.figure(figsize = (12,8))
for i in range(size):
    plt.plot(Lmda, W[i])
plt.xlabel('Lambdas')
plt.ylabel('Coefficients')
plt.title('Lasso Paths')
plt.legend()
#plt.axis('tight')
#%%
# Problem 2.5: predict reviews' star on Yelp
# TODO: evaluation of your results
"""
# load Yelp data: change the address of data files on your own machine if necessary ('../data/' in the below)
from scipy.sparse.linalg import lsqr
X_train, y_train, X_valid, y_valid, X_test, y_test, fName = DataParser('../data/star_data.mtx', '../data/star_labels.txt', '../data/star_features.txt', 30000, 5000)

# evaluation
w_lasso, w0_lasso, lmda_best = Validation(X_train, y_train, X_valid, y_valid)
"""
#%%
#RMSE_LASSO 
RMSE_train = []

for i in range (len(w_lasso)):
    pred = X_train.dot(w_lasso[i])
    RMSE = np.sqrt(np.square(y_train-pred+w0_lasso[i]).mean()) 
    RMSE_train.append(RMSE)
plt.plot(RMSE_train, Lmda)
plt.xlabel("Lambda")
plt.ylabel("RMSE")
plt.title('Lambda Vs Train_Set_RMSE')
plt.show()
#%%  
RMSE_valid = []
for i in range (len(w_lasso)):
    pred = X_valid.dot(w_lasso[i])
    RMSE = np.sqrt(np.square(np.subtract(y_valid, pred)).mean()) 
    RMSE_valid.append(RMSE)
plt.plot(RMSE_valid, Lmda)
plt.xlabel("Lambda")
plt.ylabel("RMSE")
plt.title('Lambda Vs Validation_Set_RMSE')
plt.show()

best_lamda_cal1 = min(RMSE_valid)
best_lamda_cal2 = RMSE_valid.index(best_lamda_cal1)
best_lambda = Lmda[best_lamda_cal2]

pred_test = X_test.dot(w_lasso[best_lamda_cal2])
RMSE_test = np.sqrt(np.square(np.subtract(y_test, pred_test)).mean()) 

#%%
# print the top-10 features you found by lasso
idx = (-np.abs(w_lasso[best_lamda_cal2])).argsort()[0:10]
print('Lasso select features:')
for i in range(10):
    print(fName[idx[i]],w_lasso[best_lamda_cal2][idx[i]])


# %%
