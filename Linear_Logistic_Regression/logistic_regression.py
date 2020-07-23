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
def calculate_penalty(W):
    penalty = 0
    for i in np.arange(0, W.shape[0]):
        for j in np.arange(0, W.shape[1]):
            penalty += (W[i][j] ** 2)
    penalty = np.sqrt(penalty)
    return penalty

def softmax(X, W, b, c):
    Z = np.dot(X, W.T).reshape(-1, len(c)) + b
    exp_values = np.exp(Z-(np.amax(Z, axis=1).reshape(-1, 1)))
    soft_max = exp_values/np.sum(exp_values, axis=1).reshape(-1,1)

    #soft_max= np.exp(Z-(np.amax(Z, axis=1).reshape(-1, 1)))/ np.sum(np.exp(Z-(np.amax(Z, axis=1).reshape(-1, 1))), axis=1).reshape(-1,1)
    #soft_max= np.exp(Z) / np.sum(np.exp(Z), axis=1).reshape(-1,1)
    return soft_max


def calculate_loss_function(soft_max, Y, l, penalty):
    return -1 * ((np.sum(Y * np.log(soft_max))/Y.shape[0])+ (l*penalty))

# 3.2 batch gradient descent (GD) for Logistic regression

def LogisticRegression_GD(X_train, y_train, learning_rate):
    c = np.unique(y_train)
    cl = {i:v for v,i in enumerate(c)}
    X = X_train
    W = np.random.random((len(c), X.shape[1]))
    b = np.random.random(len(c)).reshape(1, -1)
    Y = np.eye(len(c))[np.vectorize(lambda c: cl[c])(y_train).reshape(-1)]
    l = 0.1/2
    loss = []
    soft_max = softmax(X, W, b, c)
    penalty = calculate_penalty(W)
    new_F = calculate_loss_function(soft_max, Y, l, penalty)
    loss.append(new_F)
    iteration= 0
    while True:
        loss.append(new_F)
        e = Y - soft_max
        de = np.mean(e, axis=0)
        u = learning_rate*(np.dot(e.T, X)- l*W)
        W += u
        b += learning_rate * de
        prev_F = new_F
        soft_max = softmax(X, W, b, c)
        penalty = calculate_penalty(W)
        new_F = calculate_loss_function(soft_max, Y, l, penalty)
        difference_loss = abs(new_F - prev_F)
        if difference_loss < 1.0e-4:
            break
        prev_F = new_F
        iteration = iteration+1
        print(f'iteration: {str(iteration)}')
        print(f'new F: {str(new_F)}')
        print(f'prev F: {str(prev_F)}')
        print(f'difference: {str(difference_loss)}')
    b= W[:, 0]
    return W, loss, b, iteration

# %%
W1, loss1, b1, iteration1 = LogisticRegression_GD(X_train, y_train, 0.05)
#W2, loss2 , b2, iteration2 = LogisticRegression_GD(X_train, y_train, 1.0e-2)
#W3, loss3, b3,  iteration3 = LogisticRegression_GD(X_train, y_train, 5.0e-2)



# %%
