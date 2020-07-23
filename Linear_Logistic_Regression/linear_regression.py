#%% load data
from sklearn import datasets
boston = datasets.load_boston()
print(boston.keys())

#%%
# summary of data
feature = boston.data
price = boston.target
print('data size = ', feature.shape)
print('target size = ', price.shape)
print('feature attributes: ', boston.feature_names)
print(boston.DESCR)

# %%
# more details of data
import pandas as pd
df_feature = pd.DataFrame(feature, columns = boston.feature_names)
df_target = pd.DataFrame(price, columns =['MEDV'])
df_boston = pd.concat([df_feature, df_target,], axis = 1)
#%%
df_boston.head()
#%%
df_boston.describe()


#%%
# 2.1 how does each feature relate to the price
import matplotlib.pyplot as plt
plt.figure()
fig,axes = plt.subplots(4, 4, figsize=(14,18))
fig.subplots_adjust(wspace=.4, hspace=.4)
img_index = 0
for i in range(boston.feature_names.size):
    row, col = i // 4, i % 4
    axes[row][col].scatter(feature[:,i], price)
    axes[row][col].set_title(boston.feature_names[i] + ' and MEDV')
    axes[row][col].set_xlabel(boston.feature_names[i])        
    axes[row][col].set_ylabel('MEDV (price)')
plt.show()
# %%
# 2.2 correlation matrix
import seaborn as sns
fig, ax = plt.subplots(figsize=(16, 10))
correlation = df_boston.corr()
sns.heatmap(correlation, annot = True, cmap = 'RdBu')
plt.show()
correlation

# %%
# train-test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(feature, price, test_size=0.3, random_state=8)
#%%
# 2.3 linear regression and ridge regression
import numpy as np
def least_square(X, y):
    #theta = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), y)
    theta = np.matmul(np.matmul(np.linalg.inv(np.matmul(X.T, X)), X.T), y)
    return theta

def ridge_reg(X, y, eta):
    n, m = X.shape
    I= np.identity(m)
    theta_r = np.matmul(np.matmul(np.linalg.inv(np.matmul(X.T, X) + (eta/2) * I), X.T), y)
    return theta_r

# apply linear regression
theta = least_square(X_train, y_train)
df_theta = pd.DataFrame(zip(boston.feature_names, theta),columns=['Feature','Coeff'])

# apply ridge regression
theta_r = ridge_reg(X_train, y_train, 15)
df_theta_r = pd.DataFrame(zip(boston.feature_names, theta_r),columns=['Feature','Coeff'])

#%%
# 2.4 evaluation
def pred_fn(X, theta):
    pred = np.dot(X, theta)
    return pred

def root_mean_square_error(pred, y):
   rmse = np.sqrt(np.square(np.subtract(y, pred)).mean()) 
   return rmse

#Introducing Bias Offset 
train_offset = np.ones((len(X_train), 1), dtype = np.float64)
test_offset = np.ones((len(X_test), 1), dtype=np.float64)
X_train1 = np.hstack((train_offset, X_train))
X_test1 = np.hstack((test_offset, X_test))

#Linear Regression
theta = least_square(X_train1, y_train)
LR_predicted_Y_from_trainset = pred_fn(X_train1, theta)
LR_rmse_trainset = root_mean_square_error(LR_predicted_Y_from_trainset, y_train)
LR_predicted_Y_from_testset = pred_fn(X_test1, theta)
LR_rmse_testset = root_mean_square_error(LR_predicted_Y_from_testset, y_test)

#Ridge Regression
theta_r = ridge_reg(X_train1, y_train, 15)
RR_predicted_Y_from_trainset = pred_fn(X_train1, theta_r)
RR_rmse_trainset = root_mean_square_error(RR_predicted_Y_from_trainset, y_train)
RR_predicted_Y_from_testset = pred_fn(X_test1, theta_r)
RR_rmse_testset = root_mean_square_error(RR_predicted_Y_from_testset, y_test)
print(LR_rmse_trainset, LR_rmse_testset, RR_rmse_trainset, RR_rmse_testset)

#%%
# 2.5 linear models of top-3 features
# linear regression using top-3 features
X_train_top3 = X_train1[:,[6,11,13]] #RM, PTRATIO, LSTAT
X_test_top3 = X_test1[:,[6,11,13]]


theta_top3 = least_square(X_train_top3, y_train)
LR_predicted_Y_from_trainset_top3 = pred_fn(X_train_top3, theta_top3)
LR_rmse_trainset_top3 = root_mean_square_error(LR_predicted_Y_from_trainset_top3, y_train)
LR_predicted_Y_from_testset_top3 = pred_fn(X_test_top3, theta_top3)
LR_rmse_testset_top3 = root_mean_square_error(LR_predicted_Y_from_testset_top3, y_test)
   
# ridge regression using top-3 features
theta_r_top3 = ridge_reg(X_train_top3, y_train, 15) #RM, PTRATIO, LSTAT
RR_predicted_Y_from_trainset_top3 = pred_fn(X_train_top3, theta_r_top3)
RR_rmse_trainset_top3 = root_mean_square_error(RR_predicted_Y_from_trainset_top3, y_train)
RR_predicted_Y_from_testset_top3 = pred_fn(X_test_top3, theta_r_top3)
RR_rmse_testset_top3 = root_mean_square_error(RR_predicted_Y_from_testset_top3, y_test)

print(LR_rmse_trainset_top3, LR_rmse_testset_top3, RR_rmse_trainset_top3, RR_rmse_testset_top3)


#%%

#Feature Engineering

def var(X):
    V= []
    for i in range (len(X)):
        V.append(((X[i]/np.mean(X))**2)/len(X))
    fin= np.array(V)
    return fin


Zn_Indus_train = np.add(X_train1[:, 2],X_train1[:, 3])
Zn_Indus_test = np.add(X_test1[:, 2], X_test1[:, 3])
PT_train = np.log(X_train1[:, 11])
PT_test = np.log(X_test1[:, 11])
tax_train = np.log(X_train1[:, 10])
tax_test = np.log(X_test1[:, 10])
var_age_train = var(X_train1[:, 7])
nox_train = np.sqrt(X_train1[:, 5])
var_age_test = var(X_test1[:, 7])
nox_test = np.sqrt(X_test1[:, 5])
B_test = np.log(X_test1[:, 12])
B_train = np.log(X_train1[:, 12])


R_test = np.sqrt(X_test1[:, 9])
R_train = np.sqrt(X_train1[:, 9])

C_test = np.sqrt(X_test1[:, 1])
C_train = np.sqrt(X_train1[:, 1])

Final_Matrix_train = np.column_stack(((X_train1[:,[0, 1, 4, 6, 8, 13]]), Zn_Indus_train, nox_train, var_age_train, B_train, tax_train, PT_train, R_train, C_train))
Final_Matrix_test = np.column_stack(((X_test1[:,[0, 1, 4, 6, 8, 13]]), Zn_Indus_test, nox_test, var_age_test, B_test, tax_test,PT_test, R_test, C_test))


#thetaF = ridge_reg(Final_Matrix_train, y_train, 15)
thetaF = least_square(Final_Matrix_train, y_train)
predF = pred_fn(Final_Matrix_test, thetaF)
rmseF = root_mean_square_error(predF, y_test)
print(rmseF)


#%%





            









# %%



# %%
