#%%
from sklearn import datasets
digits = datasets.load_digits()

#%%
# summary of data
print('data size = ', digits.data.shape)
print('target size = ', digits.target.shape)
print(digits.DESCR)

# %%
#%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np

# show examples of dataset
plt.figure(figsize=(20,4))
for index, (image, label) in enumerate(zip(digits.data[0:5], digits.target[0:5])):
    plt.subplot(1, 5, index + 1)
    plt.imshow(np.reshape(image, (8,8)), cmap=plt.cm.gray)
    plt.title('Training: %i\n' % label, fontsize = 20)
#%%
# train-test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.25, random_state=8)
print (X_train[256], y_train[256])




# %%
# 3.2 batch gradient descent (GD) for Logistic regression
from numpy import linalg as LA
def LogisticRegression_GD(X_train, y_train, learning_rate):
    c = np.unique(y_train)
    cl = {i:v for v,i in enumerate(c)}
    W = np.zeros((X_train.shape[1], len(c))) 
    b = np.zeros(len(c)) 
    Y = np.eye(len(c))[np.vectorize(lambda c: cl[c])(y_train).reshape(-1)]
    X = np.insert(X_train, 0, 1, axis=1)
    loss = []
    diff_loss = 10**9
    cross_entropy_prev = 0
    while diff_loss> 1.0e-4:
        var1 = np.dot(X, W)
        var2 = np.exp(var1 - np.max(var1))
        if var2.ndim == 1:
            softmax = (var2 / np.sum(var2, axis=0)) * 10**(-5)
        else:
            softmax=  (var2 / np.array([np.sum(var2, axis=1)]).T) *10**(-5)
        penalty = 0
        for i in np.arange(0, W.shape[0]):
            for j in np.arange(0, W.shape[1]):
                penalty += (W[i][j] ** 2)
        penalty = np.sqrt(penalty)
        error = (Y - softmax)
        W += (learning_rate * np.dot(X_train.T, error)) - (learning_rate * (0.1/2) * W)
        b += learning_rate * np.mean(error, axis=0)
        cross_entropy_new = (- np.mean(np.sum(Y * np.log(softmax) +(1 - Y) * np.log(1 - softmax), axis=1))) +(0.1)*penalty
        loss.append(cross_entropy_new) 
        diff_loss = cross_entropy_new  - cross_entropy_prev
        cross_entropy_prev = cross_entropy_new
    return W, b, loss
#%%
W1, b1, loss1 = LogisticRegression_GD(X_train, y_train, 5.0e-3)
#%%
def Calculate_Accuracy_1(X, y):
    X = np.insert(X, 0, 1, axis=1)
    cs = np.unique(y_train)
    W1, b1, loss1 = LogisticRegression_GD(X_train, y_train, 5.0e-3)
    Z = np.dot(X, W1.T).reshape(-1,len(cs))
    var1 = np.exp(Z-(np.amax(Z, axis=1).reshape(-1, 1)))/ np.sum(np.exp(Z-(np.amax(Z, axis=1).reshape(-1, 1))), axis=1).reshape(-1,1)
    var2 = np.vectorize(lambda c: cs[c])(np.argmax(var1, axis=1))
    return np.mean(var2 == y)

def Calculate_Accuracy_2(X, y):
    X = np.insert(X, 0, 1, axis=1)
    cs = np.unique(y_train)
    W2, b2, loss2= LogisticRegression_GD(X_train, y_train, 1.0e-2)
    Z = np.dot(X, W1.T).reshape(-1,len(cs))
    var1 = np.exp(Z-(np.amax(Z, axis=1).reshape(-1, 1)))/ np.sum(np.exp(Z-(np.amax(Z, axis=1).reshape(-1, 1))), axis=1).reshape(-1,1)
    var2 = np.vectorize(lambda c: cs[c])(np.argmax(var1, axis=1))
    return np.mean(var2 == y)

def Calculate_Accuracy_3(X, y):
    X = np.insert(X, 0, 1, axis=1)
    cs = np.unique(y_train)
    W3, b3, loss3 = LogisticRegression_GD(X_train, y_train, 5.0e-2)
    Z = np.dot(X, W1.T).reshape(-1,len(cs))
    var1 = np.exp(Z-(np.amax(Z, axis=1).reshape(-1, 1)))/ np.sum(np.exp(Z-(np.amax(Z, axis=1).reshape(-1, 1))), axis=1).reshape(-1,1)
    var2 = np.vectorize(lambda c: cs[c])(np.argmax(var1, axis=1))
    return np.mean(var2 == y)

# %%
s11 = Calculate_Accuracy_1(X_train, y_train)
s12 = Calculate_Accuracy_1(X_test, y_test)
s22 = Calculate_Accuracy_2(X_train, y_train)
s23 = Calculate_Accuracy_2(X_test, y_test)
s33 = Calculate_Accuracy_3(X_train, y_train)
s34 = Calculate_Accuracy_3(X_test, y_test)
#%%
print ("For learning rate = 5.0e-3, Train_Set_Accuracy is {} and Test set Accuracy is {}".format(s11, s12))
print ("For learning rate = 1.0e-2, Train_Set_Accuracy is {} and Test set Accuracy is {}".format(s22,s23))
print ("For learning rate = 5.0e-2, Train_Set_Accuracy is {} and Test set Accuracy is {}".format (s33, s34))


# %%

W1, b1, loss1 = LogisticRegression_GD(X_train, y_train, 5.0e-3)
W2, b2, loss2= LogisticRegression_GD(X_train, y_train, 1.0e-2)
W3, b3, loss3 = LogisticRegression_GD(X_train, y_train, 5.0e-2)
# %%
fig = plt.figure(figsize=(8,6))
plt.plot(np.arange(len(loss1)), loss1)
plt.plot(np.arange(len(loss2)), loss2)
plt.plot(np.arange(len(loss3)), loss3)
plt.title("Development of loss during training")
plt.xlabel("Number of iterations")
plt.ylabel("Loss")
plt.legend(['Learning Rate:5.0e-3', 'Learning Rate:1.0e-2', 'Learning Rate:5.0e-2'], loc='lower right')

# %%
# 3.3 stochastic gradient descent (SGD) for Logistic regression

def LogisticRegression_SGD(X, y, batch_size, lr=1.0e-2, eta=2.0e-1, eps = 1.0e-4, max_epoch=500):
    
    #TODO: initialization
    notstop = True
    epoch = 0
    loss = []    

    # optimization loop	
    while notstop and epoch < max_epoch:
        
        #TODO: SGD of each epoch
        
        # half lr if not improving in 10 epochs
        if epoch > 10:
            if loss[epoch - 10] <= loss[epoch] - eps:
                lr *= 0.5
                print 'reduce learning rate to', lr
        
        # stop if not improving in 20 epochs
        if epoch > 20:
            if loss[epoch - 20] <= loss[epoch] - eps or abs(loss[epoch] - loss[epoch-1]) <= eps:                
                notstop = False
                break
            
        epoch += 1
        
        #TODO: W and b

    return (W, b, loss)