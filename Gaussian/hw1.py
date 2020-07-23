
#%%
import numpy as np
import matplotlib.pyplot as plt

# This is similar to the code from lecture 1
# to sample from a 2D Gaussian Distribution
mean = [0,0]
cov = [[1, 1.5], [1.5, 5]]
# Fix random seed to get consistent results
np.random.seed(1024)
X = np.random.multivariate_normal(mean, cov, 1000)
fig, ax = plt.subplots(figsize=(10, 10))
# c='r', dot color is red
# s=10.0, dot size is 10
# alpha=0.3, dot opacity is 0.3
ax.scatter(X[:,0], X[:,1], c='r', s=10.0, alpha=0.3, label="2D-Gaussian")
ax.grid()
ax.legend(loc = 0)
# Set x/y axis limits
ax.set_xlim([-10, 10])
ax.set_ylim([-10, 10])
#fig.show()
#%%
# TODO: Report the estimate mean and covariance
# of the sampled points
a= np.cov(X)
b= np.mean(X)
print(a)
print(b)
#%%
# TODO: Plot the histogram for the x-coordinates of X
# and y-coordinates of X respectively.
# You can use the plt.hist() function
fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)
n1, bins1, patches1 = axs[0].hist(x= X[:,0], bins=8, color='b', alpha= 0.7, rwidth=0.85)
n2, bins2, patches2 = axs[1].hist(x= X[:,1], bins=8, color='g', alpha= 0.7, rwidth=0.85)
axs[0].grid(axis='y', alpha=0.75)
axs[1].grid(axis='y', alpha=0.75)
axs[0].set_xlabel('Value')
axs[1].set_xlabel('Value')
axs[0].set_ylabel('Frequency')
axs[1].set_ylabel('Frequency')
axs[0].title.set_text('Histogram of x-coordinates of X')
axs[1].title.set_text('Histogram of y-coordinates of X')
fig.set_dpi(100)
#%%
# TODO: Are the x-coordinates of X samples from
# some Gaussian distribution?
# If so, estimate the mean and variance.
# Do the same for the y-coordinates.

#Yes they are both normally distributed
x_sample_var= np.var(X[:, 0])
y_sample_var= np.var(X[:, 1])
x_sample_mean= np.mean(X[:, 0])
y_sample_mean= np.mean(X[:, 1])
print(x_sample_var, y_sample_var, x_sample_mean, y_sample_mean)


# TODO: Generate a new 2D scatter plot of 1000 points,
# such that the x-coordinates(y-coordinates) of all the points
# are samples from a 1D Gaussian distribution 
# using the estimated mean and variance based on the x-coordinates(y-coordinates) of X.

m = [x_sample_mean,y_sample_mean]
c = [[x_sample_var, 0], [0, y_sample_var]]
sx = np.random.normal(x_sample_mean, x_sample_var, 1000)
sy = np.random.normal(y_sample_mean, y_sample_var, 1000)
fig, ax = plt.subplots(figsize=(10, 10))
# c='r', dot color is red
# s=10.0, dot size is 10
# alpha=0.3, dot opacity is 0.3
ax.scatter(sx, sy, c='r', s=10.0, alpha=0.3, label="2D-Gaussian")
ax.grid()
ax.legend(loc = 0)
# Set x/y axis limits
ax.set_xlim([-10, 10])
ax.set_ylim([-10, 10])
fig.show()


# %%
# Back to the original X
fig, ax = plt.subplots(figsize=(10, 10))
# c='r', dot color is red
# s=10.0, dot size is 10
# alpha=0.3, dot opacity is 0.3
ax.scatter(X[:,0], X[:,1], c='r', s=10.0, alpha=0.3, label="2D-Gaussian")
ax.grid()
ax.legend(loc = 0)
ax.set_xlim([-10, 10])
ax.set_ylim([-10, 10])

# TODO: Plot a line segment with x = [-10, 10]
# and y = 3x + 1 onto the 2D-Gaussian plot.
# The np.linspace() function may be helpful.
x = np.linspace(-10,10)
y = 3*x + 1
ax.plot(x,y)
fig.show()

#%%
# TODO: Project X onto line y=3x + 1
# and plot the projected points on the 2D space.
# You need to remove this line and assign the projected points to X_proj.
X_p = []
m= 3
for i in range (len(X)):
    b= (3*(X[:, 1][i])+(X[:, 0][i]))/3
    xp = (3*(b-1))/10
    yp= ((9*b)+1)/10
    X_p.append([xp, yp])
X_proj = np.array(X_p)
fig, ax = plt.subplots(figsize=(10, 10))
ax.scatter(X_proj[:,0], X_proj[:,1], c='b', s=10.0, alpha=1.0, label="Projected 2D-Gaussian")
ax.grid()
ax.legend(loc = 0)
ax.set_xlim([-10, 10])
ax.set_ylim([-10, 10])
# You can also add the line from the previous plot for verification.
fig.show()

#%%
# Here we only plot 30 points to check the correctness
fig, ax = plt.subplots(figsize=(10, 10))
ax.scatter(X_proj[:30,0], X_proj[:30,1], c='b', s=10.0, alpha=1.0, label="Projected 2D-Gaussian")
ax.scatter(X[:30,0], X[:30,1], c='r', s=10.0, alpha=1.0, label="2D-Gaussian")
ax.plot(x, y, c= 'g')
ax.grid()
ax.legend(loc = 0)
ax.set_xlim([-10, 10])
ax.set_ylim([-10, 10])
# You can also add the line from the previous plot for verification.
fig.show()
#%%
# TODO: Draw the histogram of the x-coordinates
# of the projected points.
# Are the x-coordinates of the projected points
# samples from some Gaussian distribution?
# If so, estimate the mean and variance.


plt.figure(figsize=[8,8])
n, bins, patches = plt.hist(x=X_proj[:,0], bins=8, color='#0504aa',alpha=0.7, rwidth=0.85)
plt.grid(axis='y', alpha=0.75)
plt.xlabel('Value',fontsize=15)
plt.ylabel('Frequency',fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.ylabel('Frequency',fontsize=15)
plt.title('Histogram of the x-coordinates of the projected points',fontsize=15)
plt.show()

#%%
#Yes Gaussian Distribution
x_sample_var= np.var(X_proj[:, 0])
x_sample_mean= np.mean(X_proj[:, 0])
print(x_sample_mean, x_sample_var)

# %%
