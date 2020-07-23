#%%
from sklearn import datasets
digits = datasets.load_digits()
#%%
# summary of data
print('data size = ', digits.data.shape)
print('target size = ', digits.target.shape)
print(digits.DESCR)
#%%
%matplotlib inline
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
#print X_train[256], y_train[256]

#%%
from numpy import linalg as LA
# 3.2 batch gradient descent (GD) for Logistic regression
def LogisticRegression_GD(X_train, y_train, learning_rate):
    c = np.unique(y_train)
    cl = {i:v for v,i in enumerate(c)}
    X = np.insert(X_train, 0, 1, axis=1)
    W = np.zeros(shape=(len(c), X.shape[1]))
    Y = np.eye(len(c))[np.vectorize(lambda c: cl[c])(y_train).reshape(-1)]
    var1 = np.dot(X, W.T).reshape(-1, len(c))
    exp = np.exp(var1-np.max(var1, axis=1).reshape((-1,1)))
    norms = np.sum(exp, axis=1).reshape((-1,1))
    softmax = exp / norms
    cross_entropy = (-np.mean(Y *np.log(softmax+1e-6)))+ (0.1/2)*LA.norm(W, 'fro')
    prev_entropy = cross_entropy
    loss= []
    loss.append(prev_entropy)
    while True:
        e = Y - softmax
        u = (learning_rate*np.dot(e.T, X)) - (learning_rate*(0.1/2)*W)
        W += u
        var1 = np.dot(X, W.T).reshape(-1, len(c))
        exp = np.exp(var1-np.max(var1, axis=1).reshape((-1,1)))
        norms = np.sum(exp, axis=1).reshape((-1,1))
        softmax = exp / norms
        cross_entropy = (- np.mean(Y * np.log(softmax+1e-6)))+ (0.1/2)*LA.norm(W, 'fro')
        new_entropy = cross_entropy
        loss.append(new_entropy)
        diff_loss = new_entropy- prev_entropy
        if diff_loss< 1.0e-4:
            break
        prev_entropy = new_entropy
    return W, b, loss
#%%
W1, b1, loss1 = LogisticRegression_GD(X_train, y_train, 5.0e-3)

# %%
