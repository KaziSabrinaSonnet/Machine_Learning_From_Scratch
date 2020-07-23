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

#%%
def calculate_penalty(W):
    penalty = 0
    for i in np.arange(0, W.shape[0]):
        for j in np.arange(0, W.shape[1]):
            penalty += (W[i][j] ** 2)
    penalty = np.sqrt(penalty)
    return penalty

def softmax(X, W, c):
    """
    Z = np.dot(X, W.T).reshape(-1, len(c))
    soft_max = np.exp(Z) / np.sum(np.exp(Z), axis=1).reshape(-1,1)
    """
    Z = np.dot(X, W.T).reshape(-1, len(c))
    exp_values = np.exp(Z-(np.amax(Z, axis=1).reshape(-1, 1))) + 1.0e-64
    #exp_values = np.exp(Z-(np.amax(Z, axis=1).reshape(-1, 1)))
    soft_max = exp_values/np.sum(exp_values, axis=1).reshape(-1,1)
    
    return soft_max


def calculate_loss_function(soft_max, Y, l, penalty):
    return -1 * (np.mean(Y * np.log(soft_max)) + (l * penalty))
    #return -1 * np.mean(Y * np.log(soft_max))


# 3.2 batch gradient descent (GD) for Logistic regression
def LogisticRegression_GD(X_train, y_train, learning_rate):
    np.random.seed(42) 

    X_train = np.insert(X_train, 0, 1, axis=1)
    c = np.unique(y_train)
    cl = {i:v for v,i in enumerate(c)}
    y_train = np.eye(len(c))[np.vectorize(lambda c: cl[c])(y_train).reshape(-1)]

    W = np.zeros(shape=(len(c), X_train.shape[1]))

    l = 0.1/2
    loss = []
    iteration = 0

    while True:
        soft_max = softmax(X_train, W, c)
        penalty = calculate_penalty(W)
        loss_value = calculate_loss_function(soft_max, y_train, l, penalty)
        loss.append(loss_value)

        print(f'iteration: {str(iteration)}')
        print(f'loss_value: {str(loss_value)}')

        if len(loss) >=2 and abs(loss[len(loss)-2] - loss[len(list_loss)-1]) < 1.0e-4:
            break
        iteration += 1

        e = y_train - soft_max
        u = learning_rate * np.dot(e.T, X_train) - learning_rate * l * W
        W += u
        
    b= W[:, 0]
    return W, b, loss
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
    Z = np.dot(X, W2.T).reshape(-1,len(cs))
    var1 = np.exp(Z-(np.amax(Z, axis=1).reshape(-1, 1)))/ np.sum(np.exp(Z-(np.amax(Z, axis=1).reshape(-1, 1))), axis=1).reshape(-1,1)
    var2 = np.vectorize(lambda c: cs[c])(np.argmax(var1, axis=1))
    return np.mean(var2 == y)

def Calculate_Accuracy_3(X, y):
    X = np.insert(X, 0, 1, axis=1)
    cs = np.unique(y_train)
    W3, b3, loss3 = LogisticRegression_GD(X_train, y_train, 5.0e-2)
    Z = np.dot(X, W3.T).reshape(-1,len(cs))
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

# %%
W2, b2, loss2= LogisticRegression_GD(X_train, y_train, 1.0e-2)

# %%
W3, b3, loss3 = LogisticRegression_GD(X_train, y_train, 5.0e-2)
# %%
fig = plt.figure(figsize=(8,6))
plt.plot(np.arange(len(loss1)), loss1)
#plt.plot(np.arange(len(loss2)), loss2)
plt.title("Development of loss during training")
plt.xlabel("Number of iterations")
plt.ylabel("Loss")
plt.legend(['Learning Rate:5.0e-3'], loc='lower right')




# %%
