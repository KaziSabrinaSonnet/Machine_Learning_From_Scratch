#%%
#from multiprocessing import Pool
#from functools import partial
import numpy as np
#from numba import jit
import math
#%%
#TODO: loss of least square regression and binary logistic regression
'''
    pred() takes GBDT/RF outputs, i.e., the "score", as its inputs, and returns predictions.
    g() is the gradient/1st order derivative, which takes true values "true" and scores as input, and returns gradient.
    h() is the heassian/2nd order derivative, which takes true values "true" and scores as input, and returns hessian.
'''
class leastsquare(object):
    '''Loss class for mse. As for mse, pred function is pred=score.'''
    def pred(self,score):
        return score

    def g(self,true,score):
        gradient = -2*(true-score)
        return gradient

    def h(self,true,score):
        hessian = np.repeat(2, len(true))
        return hessian

class logistic(object):
    '''Loss class for log loss. As for log loss, pred function is logistic transformation.'''
    def pred(self,score):
        Pr1 = 1/(1+np.exp(-score))
        prdc = [1 if item >= 0.5 else 0 for item in Pr1]
        return np.array(prdc)

    def g(self,true,score):
        var1 = np.exp(score)
        var2 = np.exp(-score)
        gradient = (-true/(1+var1))+((1-true)/(1+var2))
        return gradient
    def h(self,true,score):
        var1 = np.exp(score)
        var2 = np.exp(-score)
        hessian = ((true*var1)/(1+var1)**2)+ (((1-true)*var2)/(1+var2)**2)
        return hessian
#%%
def bootstrap(train, target):
        samples = train.shape[0]
        index = np.random.choice(samples, samples, replace=True)
        return train[index], target[index]        
#%%
# TODO: class of Random Forest
class RF(object):
    '''
    Class of Random Forest
    
    Parameters:
        n_threads: The number of threads used for fitting and predicting.
        loss: Loss function for gradient boosting.
            'mse' for regression task and 'log' for classfication task.
            A child class of the loss class could be passed to implement customized loss.
        max_depth: The maximum depth d_max of a tree.
        min_sample_split: The minimum number of samples required to further split a node.
        lamda: The regularization coefficient for leaf score, also known as lambda.
        gamma: The regularization coefficient for number of tree nodes, also know as gamma.
        rf: rf*m is the size of random subset of features, from which we select the best decision rule.
        num_trees: Number of trees.
        
    '''

    def __init__(self, loss = 'mse',
        max_depth = 10, min_sample_split = 2, 
        lamda = 0.2, gamma = 0.1,
        rf = 0.5, num_trees = 20, feat = None):
        
        #self.n_threads = n_threads
        self.loss = loss
        self.max_depth = max_depth
        self.min_sample_split = min_sample_split
        self.lamda = lamda
        self.gamma = gamma
        self.rf = rf
        self.num_trees = num_trees
        self.feat = feat
        self.tree= []
    
    def fit(self, train, target):
        # train is n x m 2d numpy array
        # target is n-dim 1d array
        #TODO
        self.tree  = []
        print(f'Fitting {self.num_trees} number of trees for RF')
        for x in range(self.num_trees):
            print(f'Tree number: {x}')
            single_tree = Tree(rf= self.rf, loss= self.loss, max_depth=  self.max_depth, min_sample_split=self.min_sample_split,
                lamda=self.lamda, gamma = self.gamma, feat = self.feat)
            sample_x, sample_y = bootstrap(train, target)
            single_tree.fit(sample_x, sample_y)
            self.tree.append(single_tree)
        return self

    def predict(self, test):
        #TODO
        predicted_tree = np.array([tree.predict(test) for tree in self.tree])
        predicted_tree = np.swapaxes(predicted_tree, 0, 1)
        if self.loss == 'mse':
            y_pred = [np.mean(pt) for pt in predicted_tree]
        if self.loss == 'logistic':
            y_hat_score = np.mean(predicted_tree, axis=1)
            y_pred = logistic.pred(self, y_hat_score)
        
        return np.array(y_pred)
       
    
#%%
# TODO: class of GBDT
class GBDT(object):
    '''
    Class of gradient boosting decision tree (GBDT)
    
    Parameters:
        n_threads: The number of threads used for fitting and predicting.
        loss: Loss function for gradient boosting.
            'mse' for regression task and 'log' for classfication task.
            A child class of the loss class could be passed to implement customized loss.
        max_depth: The maximum depth D_max of a tree.
        min_sample_split: The minimum number of samples required to further split a node.
        lamda: The regularization coefficient for leaf score, also known as lambda.
        gamma: The regularization coefficient for number of tree nodes, also know as gamma.
        learning_rate: The learning rate eta of GBDT.
        num_trees: Number of trees.
    '''
    def __init__(self, loss = 'mse',
        max_depth = 3, min_sample_split = 2, 
        lamda = 0.2, gamma = 0.1,
        learning_rate = 0.1, num_trees = 100, rf=1):
        
        #self.n_threads = n_threads
        self.loss = loss
        self.max_depth = max_depth
        self.min_sample_split = min_sample_split
        self.lamda = lamda
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.num_trees = num_trees
        self.rf = rf
        self.boosting_tree = []

    def fit(self, train, target):
        # train is n x m 2d numpy array
        # target is n-dim 1d array
        #TODO

        self.boosting_tree  = []
        print(f'Fitting {self.num_trees} number of trees for GBDT')
        for x in range(self.num_trees):
            print(f'Tree number: {x}')
            single_tree = Tree(rf= self.rf, loss= self.loss, max_depth=  self.max_depth, min_sample_split=self.min_sample_split,
                lamda=self.lamda, gamma = self.gamma, feat = None, lr= self.learning_rate)
            single_tree.fit(train, target, list_prev_tree=self.boosting_tree)
            self.boosting_tree.append(single_tree)
        return self

    def predict(self, test):
        predicted_boosting_tree = np.array([tree.predict(test) for tree in self.boosting_tree])
        predicted_tree = np.swapaxes(predicted_boosting_tree, 0, 1)
        if self.loss == 'mse':
            y_pred = [np.sum(pt) for pt in predicted_tree]
        if self.loss == 'logistic':
            y_hat_score = np.sum(predicted_tree, axis=1)
            y_pred = logistic.pred(self, y_hat_score)
        return np.array(y_pred)
#%%
# TODO: class of a node on a tree
class TreeNode(object):
    '''
    Data structure that are used for storing a node on a tree.
    
    A tree is presented by a set of nested TreeNodes,
    with one TreeNode pointing two child TreeNodes,
    until a tree leaf is reached.
    
    A node on a tree can be either a leaf node or a non-leaf node.
    '''
    
    #TODO
    def __init__(self, feature=None, threshold=None, left_child=None, right_child=None, *, leaf_value=None):
        self.feature = feature
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.leaf_value = leaf_value
        #self.is_leaf = False
        
        #[X1, X2, index_y, value_y] = split(X)
        #self.left_child = TreeNode(X1)
        #self.right_child = TreeNode(X2)
    """    
    def forward(self, x):
        if x[index_y] < value_y:
            return self.left_child
    """    
    def is_leaf_node(self):
        return self.leaf_value is not None
#%%
# TODO: class of single tree
class Tree(object):
    '''
    Class of a single decision tree in GBDT

    Parameters:
        n_threads: The number of threads used for fitting and predicting.
        max_depth: The maximum depth of the tree.
        min_sample_split: The minimum number of samples required to further split a node.
        lamda: The regularization coefficient for leaf prediction, also known as lambda.
        gamma: The regularization coefficient for number of TreeNode, also know as gamma.
        rf: rf*m is the size of random subset of features, from which we select the best decision rule,
            rf = 0 means we are training a GBDT.
    '''
    
    def __init__(self, rf, loss, max_depth = 3, min_sample_split = 10,
                 lamda = 2, gamma = 0.3, feat = None, lr= 0.2):
        #self.n_threads = n_threads
        self.max_depth = max_depth
        self.min_sample_split = min_sample_split
        self.lamda = lamda
        self.gamma = gamma
        self.rf = rf
        self.int_member = 0
        self.feat = feat
        self.root = None
        self.loss = loss 
        self.lr = lr

    def fit(self, train_x, train_y, list_prev_tree=None):
        '''
        train is the training data matrix, and must be numpy array (an n_train x m matrix).
        g and h are gradient and hessian respectively.
        '''
        #TODO
        self.feat = int(train_x.shape[1] * self.rf)
        self.root = self.construct_tree(train_x, train_y, list_prev_tree=list_prev_tree)
        

    def predict(self,test):
        '''
        test is the test data matrix, and must be numpy arrays (an n_test x m matrix).
        Return predictions (scores) as an array.
        '''
        #TODO
        result = np.array([self.travel_tree(t, self.root) for t in test])
        return result

    def construct_tree(self, train_x, train_y, depth= 0, gain = 1, list_prev_tree=None):
        '''
        Tree construction, which is recursively used to grow a tree.
        First we should check if we should stop further splitting.
        The stopping conditions include:
            1. tree reaches max_depth $d_{max}$
            2. The number of sample points at current node is less than min_sample_split, i.e., $n_{min}$
            3. gain <= 0
        '''
        #TODO
        samples, features = train_x.shape
        labels = len(np.unique(train_y))
        if (depth >= self.max_depth or labels == 1 or samples < self.min_sample_split or gain<= 0):
            leaf = self.find_wk(train_x, train_y, self.loss, self.lamda, list_prev_tree)
            return TreeNode(leaf_value=leaf)
        feature_index = list(range(features))
        if self.rf != 1:
            feature_index = np.random.choice(features, self.feat, replace= False)
        best_feature, best_threshold, gain = self.find_best_decision_rule(train_x, train_y, feature_index, self.loss, self.lamda, self.gamma, self.lr, self.rf, list_prev_tree)
        left_child_index, right_child_index = self.split(train_x[:, best_feature], best_threshold) 
        left_child = self.construct_tree(train_x[left_child_index, :], train_y[left_child_index], depth+1, gain, list_prev_tree)
        right_child = self.construct_tree(train_x[right_child_index, :], train_y[right_child_index], depth+1, gain, list_prev_tree)

        return TreeNode(best_feature, best_threshold, left_child, right_child)
    

    def find_best_decision_rule(self, train_x, train_y, feature_index, loss, lamda, gamma, lr, rf, list_prev_tree=None):
        '''
        Return the best decision rule [feature, treshold], i.e., $(p_j, \tau_j)$ on a node j, 
        train is the training data assigned to node j
        g and h are the corresponding 1st and 2nd derivatives for each data point in train
        g and h should be vectors of the same length as the number of data points in train
        
        for each feature, we find the best threshold by find_threshold(),
        a [threshold, best_gain] list is returned for each feature.
        Then we select the feature with the largest best_gain,
        and return the best decision rule [feature, treshold] together with its gain.
        '''
        #TODO
        if rf !=1:
            X_selected = train_x[:, feature_index]
        else:
            X_selected = train_x
        threshold = list()
        for feature in range(X_selected.shape[1]):
            sort = (np.sort(X_selected[:, feature])).tolist()
            i= 0
            temp_threshold = []
            for item in range (len(sort)-1):
                temp_threshold.append((sort[i]+sort[i+1])/2)    
                i= i+1
            threshold.append(temp_threshold)
        Fin_G, Fin_T = self.find_threshold(X_selected, train_y, threshold, loss, lamda, gamma, lr, list_prev_tree)
        max_gain_index = Fin_G.index(max(Fin_G))
        best_feature = max_gain_index
        best_threshold = Fin_T[max_gain_index]
        return best_feature, best_threshold, max(Fin_G)
    
    def get_current_pred(self, feature_set, labels, list_prev_tree, bool_rf):
        tree_pred = np.repeat(0, len(labels))
        if bool_rf:
            return tree_pred
        for tree in list_prev_tree:
            tree_pred = np.add(tree_pred, tree.predict(feature_set))
        return tree_pred

    def find_threshold(self, X_selected, train_y, threshold, loss, lamda, gamma, lr, list_prev_tree=None):
        '''
        Given a particular feature $p_j$,
        return the best split threshold $\tau_j$ together with the gain that is achieved.
        '''
        Fin_G = list()
        Fin_T = list()
        #TODO 
        i= -1
        for feature in range (X_selected.shape[1]):
            xj = X_selected[:, feature].tolist()
            G = list()
            T =list()
            i = i+1
            for item in threshold[i]:
                list_left_index = [idx for idx in range(len(train_y)) if xj[idx] < item]
                list_right_index = [idx for idx in range(len(train_y)) if xj[idx] >= item]
                x_left = X_selected[list_left_index]
                y_left = train_y[list_left_index]
                x_right = X_selected[list_right_index]
                y_right = train_y[list_right_index]
                yhat_left = self.get_current_pred(x_left, y_left, list_prev_tree, self.rf != 1)
                yhat_right = self.get_current_pred(x_right, y_right, list_prev_tree, self.rf != 1)
                if self.loss == 'mse':
                    gjL = leastsquare.g(self, np.array(y_left), yhat_left)
                    gjL = np.sum(gjL)
                    hjL = leastsquare.h(self, np.array(y_left), yhat_left)
                    hjL = np.sum(hjL)
                    gjR = leastsquare.g(self, np.array(y_right), yhat_right)
                    gjR = np.sum(gjR)
                    hjR = leastsquare.h(self, np.array(y_right), yhat_right)
                    hjR = np.sum(hjR)
                if self.loss == 'logistic':
                    gjL = logistic.g(self, np.array(y_left), yhat_left)
                    gjL = np.sum(gjL)
                    hjL = logistic.h(self, np.array(y_left), yhat_left)
                    hjL = np.sum(hjL)
                    gjR = logistic.g(self, np.array(y_right), yhat_right)
                    gjR = np.sum(gjR)
                    hjR = logistic.h(self, np.array(y_right), yhat_right)
                    hjR = np.sum(hjR)
                Gain = 0.5* (((gjL**2)/(hjL+lamda)) + ((gjR**2)/(hjR+lamda))-(((gjL+gjR)**2)/(hjL+hjR+lamda)))-gamma
                G.append(Gain)
                T.append(item)
            max_gain = max(G)
            max_gain_index = G.index(max_gain)
            selected_threshold = T[max_gain_index]
            Fin_G.append(max_gain)
            Fin_T.append(selected_threshold)
        return Fin_G, Fin_T


    def find_wk(self, train_x, train_y, loss, lamda, list_prev_tree=None):
        yhat = self.get_current_pred(train_x, train_y, list_prev_tree, self.rf != 1)
        if self.loss == 'mse':
            gj = leastsquare.g(self, train_y, yhat)
            hj = leastsquare.h(self, train_y, yhat)
        if self.loss == 'logistic':
            gj = logistic.g(self, train_y, yhat)
            hj = logistic.h(self, train_y, yhat)
        Gj = np.sum(gj)
        Hj = np.sum(hj)
        Wk = -Gj/(Hj+lamda)
        return Wk

    def travel_tree(self, train_x, treenode):
        if treenode.is_leaf_node():
            return treenode.leaf_value
        elif train_x[treenode.feature] <= treenode.threshold:
            return self.travel_tree(train_x, treenode.left_child)
        else:
            return self.travel_tree(train_x, treenode.right_child)

    

    def split(self, train_x, best_threshold):
        left_child_index = np.argwhere(train_x <= best_threshold).flatten()
        right_child_index = np.argwhere(train_x > best_threshold).flatten()
        return left_child_index, right_child_index
    
#%%
# TODO: Evaluation functions (you can use code from previous homeworks)

def root_mean_square_error(pred, y):
   rmse = np.sqrt(np.square(np.subtract(y, pred)).mean()) 
   return rmse

# precision
def accuracy(pred, y):
    return np.sum(y==pred)/len(y)

# %%
list_train_metric = []
list_test_metric = []
#%%
#%%
# TODO: RF regression on boston house price dataset

# load data
from sklearn import datasets
boston = datasets.load_boston()
X = boston.data
y = boston.target

# train-test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=8)
clf = RF(loss='mse', num_trees=35)
clf.fit(X_train, y_train)
y_pred_train = clf.predict(X_train)
RMSE_train = root_mean_square_error(y_pred_train, y_train)
print ("Train RMSE:", RMSE_train)
list_train_metric.append(RMSE_train)

y_pred_test = clf.predict(X_test)
RMSE_test = root_mean_square_error(y_pred_test, y_test)
print ("Test RMSE:", RMSE_test)
list_test_metric.append(RMSE_test)
#%%
# TODO: GBDT regression on boston house price dataset
# load data
from sklearn import datasets
boston = datasets.load_boston()
X = boston.data
y = boston.target
# train-test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=8)
clf = GBDT(loss='mse', num_trees=3)
clf.fit(X_train, y_train)

y_pred_train = clf.predict(X_train)
RMSE_train = root_mean_square_error(y_pred_train, y_train)
print ("Train RMSE:", RMSE_train)
list_train_metric.append(RMSE_train)

y_pred_test = clf.predict(X_test)
RMSE_test = root_mean_square_error(y_pred_test, y_test)
print ("Test RMSE:", RMSE_test)
list_test_metric.append(RMSE_test)
#%%
#%%
# TODO: Least Square 
# load data
from sklearn import datasets
boston = datasets.load_boston()
X = boston.data
y = boston.target
# train-test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=8)

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
def pred_fn(X, theta):
    pred = np.dot(X, theta)
    return pred
def root_mean_square_error(pred, y):
   rmse = np.sqrt(np.square(np.subtract(y, pred)).mean()) 
   return rmse
#Linear Regression
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
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
a= list_train_metric[0]
b= list_test_metric[0]

labels = ['Least-square', 'Ridge-regression', 'Random-forest']
Train_RMSE = [round(LR_rmse_trainset, 2), round(RR_rmse_trainset, 2), round(a, 2)]
Test_RMSE = [round(LR_rmse_testset,2), round(RR_rmse_testset, 2), round(b, 2)]

x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, Train_RMSE, width, label='Train')
rects2 = ax.bar(x + width/2, Test_RMSE, width, label='Test')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('RMSE')
ax.set_title('Comparison of Regression Models')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

"""
def autolabel(rects):
    
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')


autolabel(rects1)
autolabel(rects2)
"""



fig.tight_layout()
plt.show()

#%%
# TODO: RF classification on credit-g dataset

# load data
from sklearn.datasets import fetch_openml
X, y = fetch_openml('credit-g', version=1, return_X_y=True, data_home='credit/')
y = np.array(list(map(lambda x: 1 if x == 'good' else 0, y)))

# train-test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=8)
clf = GBDT(loss='logistic', num_trees=30)
clf.fit(X_train, y_train)

y_pred_train = clf.predict(X_train)
acc_train = accuracy(y_train, y_pred_train)
print ("Train Accuracy:", acc_train)
list_train_metric.append(acc_train)

y_pred_test = clf.predict(X_test)
acc_test = accuracy(y_test, y_pred_test)
print ("Test Accuracy:", acc_test)
list_test_metric.append(acc_test)

#%%
# TODO: GBDT classification on breast cancer dataset

# load data
from sklearn import datasets
breast_cancer = datasets.load_breast_cancer()
X = breast_cancer.data
y = breast_cancer.target

# train-test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=8)
clf = GBDT(loss='logistic', num_trees=30)
clf.fit(X_train, y_train)

y_pred_train = clf.predict(X_train)
acc_train = accuracy(y_train, y_pred_train)
print ("Train Accuracy:", acc_train)
list_train_metric.append(acc_train)

y_pred_test = clf.predict(X_test)
acc_test = accuracy(y_test, y_pred_test)
print ("Test Accuracy:", acc_test)
list_test_metric.append(acc_test)

# %%
print('Training metric: ')
print(list_train_metric)

print('Test metric: ')
print(list_test_metric)
