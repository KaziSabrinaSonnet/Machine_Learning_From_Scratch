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
    #penalty = np.sqrt(penalty)
    return penalty

def softmax(X, W, c):
    var1 = np.dot(X, W.T).reshape(-1, len(c))
    exp = np.exp(var1-np.max(var1, axis=1).reshape((-1,1)))
    norms = np.sum(exp, axis=1).reshape((-1,1))
    softmax = exp / norms
    return softmax


def calculate_loss_function(soft_max, Y, l, penalty):
    return (-1* ((np.sum(Y * np.log(soft_max+1e-6))/Y.shape[0]))) + (l*penalty)
    #return -1 * ((np.mean(Y * np.log(soft_max+1e-6)))+ (l*penalty))

# 3.2 batch gradient descent (GD) for Logistic regression

def LogisticRegression_GD(X_train, y_train, learning_rate):
    c = np.unique(y_train)
    cl = {i:v for v,i in enumerate(c)}
    X = np.insert(X_train, 0, 1, axis=1)
    W = np.zeros(shape=(len(c), X.shape[1]))
    Y = np.eye(len(c))[np.vectorize(lambda c: cl[c])(y_train).reshape(-1)]
    l = 0.1/2
    loss = []
    soft_max = softmax(X, W, c)
    penalty = calculate_penalty(W)
    #penalty = LA.norm(W, 'fro')
    prev_F = calculate_loss_function(soft_max, Y, l, penalty)
    loss.append(prev_F)
    iteration =0
    while True:
        e = Y - soft_max
        #de = np.mean(e, axis=0)
        #u = (learning_rate/Y.shape[0]*np.dot(e.T, X))- (learning_rate *l *W)
        u= ((learning_rate/Y.shape[0]) * np.dot(e.T, X)) - (learning_rate*0.1*W)
        W += u
        soft_max = softmax(X, W, c)
        penalty = calculate_penalty(W)
        #penalty = LA.norm(W, 'fro')
        new_F = calculate_loss_function(soft_max, Y, l, penalty)
        loss.append(new_F)
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
#%%
Train = [s33, s11, s22]
Test = [s34,  s12, s23]

#%%
# evaluation of different learning rate
learning_rate = [5.0e-2, 5.0e-3, 1.0e-2]
cl = ['darkgreen', 'cyan', 'red']
fig, ax = plt.subplots(figsize=(10, 8))
for i in range(len(learning_rate)):
    
    print ('---------------------------------------')
    print ("learning rate = {} ".format(learning_rate[i]))
    
    W, b, loss_GD = LogisticRegression_GD(X_train, y_train, learning_rate[i])
    
    #TODO
    print ("Training precision is:{}".format(Train[i]))

    #TODO
    print ("Testing precision is:{}".format(Test[i]))
    
    
    plt.plot(loss_GD, c = cl[i], ls = '-', marker = 'o', label = 'batch gradient descent (lr = ' + str(learning_rate[i]) + ')')

plt.grid()
plt.legend()
plt.xlabel('iteration')
plt.ylabel('loss')
#%%
def get_mini_batches(X, y, batch_size):
    random_idxs = np.random.choice(len(y), len(y), replace=False)
    X_shuffled = X[random_idxs,:]
    y_shuffled = y[random_idxs]
    mini_batches = [(X_shuffled[i:i+batch_size,:], y_shuffled[i:i+batch_size]) for
                   i in range(0, len(y), batch_size)]
    return mini_batches

# %%
def LogisticRegression_SGD(X, y, batch_size, lr=1.0e-2, eta=2.0e-1, eps = 1.0e-4, max_epoch=500):
    
    #TODO: initialization
    notstop = True
    epoch = 0
    loss = []  

    X = np.insert(X, 0, 1, axis=1)
    c = np.unique(y)
    cl = {i:v for v,i in enumerate(c)}
    y = np.eye(len(c))[np.vectorize(lambda c: cl[c])(y).reshape(-1)]
    
    W = np.zeros(shape=(len(c), X.shape[1]))
    l = 0.1/2

    # optimization loop	
    while notstop and epoch < max_epoch:
        
        #TODO: SGD of each epoch
        mini_batches = get_mini_batches(X, y, batch_size)
        loss_epoch = 0.0
        for mb in mini_batches:
            X_i = mb[0]
            y_i = mb[1]

            if y_i.shape[0] < batch_size:
                continue

            soft_max = softmax(X_i, W, c)
            e = y_i - soft_max
            u = ((lr/y_i.shape[0]) * np.dot(e.T, X_i)) - (lr * 0.1 * W)
            W += u

            penalty = calculate_penalty(W)
            loss_epoch += calculate_loss_function(soft_max, y_i, l, penalty)
        loss.append(loss_epoch)
        
        print(f'epoch: {epoch}')
        print(f'loss: {loss_epoch}')
        
        # half lr if not improving in 10 epochs
        if epoch > 10:
            if loss[epoch - 10] <= loss[epoch] - eps:
                lr *= 0.5
                print(f'reduce learning rate to {lr}')
        
        # stop if not improving in 20 epochs
        if epoch > 20:
            if loss[epoch - 20] <= loss[epoch] - eps or abs(loss[epoch] - loss[epoch-1]) <= eps:                
                notstop = False
                break
            
        epoch += 1
        
        #TODO: W and b
    b = W[:, 0]
    W = W[:, 1:]
    return (W, b, loss)

# %%
def Calculate_Precision_1(X, y):
    X = np.insert(X, 0, 1, axis=1)
    cs = np.unique(y_train)
    W1, b1, loss1 = LogisticRegression_SGD(X_train, y_train, 10, lr=0.01, eta=2.0e-1, eps = 1.0e-4, max_epoch=500)
    W = np.hstack((b1.reshape(-1, 1), W1))
    Z = np.dot(X, W.T).reshape(-1,len(cs))
    var1 = np.exp(Z-(np.amax(Z, axis=1).reshape(-1, 1)))/ np.sum(np.exp(Z-(np.amax(Z, axis=1).reshape(-1, 1))), axis=1).reshape(-1,1)
    var2 = np.vectorize(lambda c: cs[c])(np.argmax(var1, axis=1))
    return np.mean(var2 == y)
def Calculate_Precision_2(X, y):
    X = np.insert(X, 0, 1, axis=1)
    cs = np.unique(y_train)
    W2, b2, loss2 = LogisticRegression_SGD(X_train, y_train, 50, lr=0.01, eta=2.0e-1, eps = 1.0e-4, max_epoch=500)
    W = np.hstack((b2.reshape(-1, 1), W2))
    Z = np.dot(X, W.T).reshape(-1,len(cs))
    var1 = np.exp(Z-(np.amax(Z, axis=1).reshape(-1, 1)))/ np.sum(np.exp(Z-(np.amax(Z, axis=1).reshape(-1, 1))), axis=1).reshape(-1,1)
    var2 = np.vectorize(lambda c: cs[c])(np.argmax(var1, axis=1))
    return np.mean(var2 == y)
def Calculate_Precision_3(X, y):
    X = np.insert(X, 0, 1, axis=1)
    cs = np.unique(y_train)
    W3, b3, loss3 = LogisticRegression_SGD(X_train, y_train, 100, lr=0.01, eta=2.0e-1, eps = 1.0e-4, max_epoch=500)
    W = np.hstack((b3.reshape(-1, 1), W3))
    Z = np.dot(X, W.T).reshape(-1,len(cs))
    var1 = np.exp(Z-(np.amax(Z, axis=1).reshape(-1, 1)))/ np.sum(np.exp(Z-(np.amax(Z, axis=1).reshape(-1, 1))), axis=1).reshape(-1,1)
    var2 = np.vectorize(lambda c: cs[c])(np.argmax(var1, axis=1))
    return np.mean(var2 == y)
#%%
b11 = Calculate_Precision_1(X_train, y_train)
b12 = Calculate_Precision_1(X_test, y_test)
b22 = Calculate_Precision_2(X_train, y_train)
b23 = Calculate_Precision_2(X_test, y_test)
b33 = Calculate_Precision_3(X_train, y_train)
b34 = Calculate_Precision_3(X_test, y_test)
print ("For batch = 10 and lr = 0.01, Train_Set_Accuracy is {} and Test set Accuracy is {}".format(b11, b12))
print ("For batch = 50 and lr = 0.01, Train_Set_Accuracy is {} and Test set Accuracy is {}".format(b22,b23))
print ("For batch = 100 and lr = 0.01, Train_Set_Accuracy is {} and Test set Accuracy is {}".format (b33, b34))
#%%
Train1 = [b11, b22, b33]
Test1 = [b12, b23, b34]
#%%
# evaluation of different batch size
bs = [10, 50, 100]
cl = ['green', 'blue', 'orange']
# TODO: different learning rate for different batch size
lr = [0.01, 0.01, 0.01]
fig, ax = plt.subplots(figsize=(10, 8))

for i in range(len(bs)):
    print ('---------------------------------------')
    print ('batch_size = {}'.format(bs[i]))
    W, b, loss_SGD = LogisticRegression_SGD(X_train, y_train, bs[i], lr[i], eta = 2.0e-1, eps = 1.0e-4, max_epoch = 500)
    
      
    #TODO
    print ("Training precision is:{}".format(Train1[i]))

    #TODO
    print ("Testing precision is:{}".format(Test1[i]))
    
    plt.plot(loss_SGD, c = cl[i], ls = '-', marker = 'o', label = 'stochastic gradient descent (batch_size = ' + str(bs[i]) + ')')

plt.grid()
plt.legend()
plt.xlabel('epoch')
plt.ylabel('loss')


# %%
W1, b1, loss1 = LogisticRegression_SGD(X_train, y_train, 10, lr=0.01, eta=2.0e-1, eps = 1.0e-4, max_epoch=500)
W2, b2, loss2 = LogisticRegression_SGD(X_train, y_train, 50, lr=0.01, eta=2.0e-1, eps = 1.0e-4, max_epoch=500)
W3, b3, loss3 = LogisticRegression_SGD(X_train, y_train, 100, lr=0.01, eta=2.0e-1, eps = 1.0e-4, max_epoch=500)
W11, b11, loss11 = LogisticRegression_GD(X_train, y_train, 5.0e-3)
W22, b22, loss22= LogisticRegression_GD(X_train, y_train, 1.0e-2)
W33, b33, loss33 = LogisticRegression_GD(X_train, y_train, 5.0e-2)



# %%
