import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
from sklearn.linear_model import LogisticRegression



def gradient_descent_p4(X, y, alpha, lamda, T):
    '''
    Inputs:
    numpy matrix X of shape (m, n)
    output vector y of shape (m, )
    scalar learning rate alpha
    regularization strength parameter lambda
    number of iterations T

    Output:
    a vector theta of shape (n + 1; 1).
    theta should be the logistic regression parameter vector found by executing
    the gradient descent algorithm for T iterations on the given inputs.'''

    m = len(X)
    n = len(X[0])
    X = np.c_[np.ones((m, 1)), X]  # place a vector of ones in first column
    theta = np.random.rand(1, n + 1)  # Initial theta

    # # normalize the data
    # for r in range(len(X)):
    #     mean = np.mean(X[r])
    #     range_diff = np.max(X[r]) - np.min(X[r])
    #     for c in range(len(X[r])):
    #         X[r][c] = (X[r][c] - mean)/range_diff

    cost_lst = []
    for k in range(T):
        for j in range(n+1):
            summation = theta[0].dot(X[0]) - y[0]
            for i in range(1, m):
                summation += (theta[0].dot(X[i]) - y[i])*X[i][j]
            if j == 0:
                theta[0][j] = theta[0][j] - alpha * (1/m) * summation
            else:
                theta[0][j] = theta[0][j] - alpha * \
                    (1/m) * summation + (lamda/m)*theta[0][j]

        summation_cost = 0
        for i in range(m):
            summation_cost += (theta.dot(X[i]) - y[i]) ** 2
        cost = summation_cost / (2 * m)
        cost_lst.append(cost)

    _, ax = plt.subplots()
    ax.set_ylabel('J(Theta), cost')
    ax.set_xlabel('T, (itearations)')
    _ = ax.plot(range(T), cost_lst, 'b.')
    plt.show()
    return theta


# Example cases for Problem 4
m = 200  # number of examples
n = 5  # number of features (x0, x1, ..., xn) where x0 is 1
alpha = 0.01  # learning rate
lam = 0.5  # regularization parameter
T = 200  # number of iterations
X = np.random.rand(m, n)

y = np.zeros((m, 1))
for i in range(len(y)):
    if np.random.randint(0, 2) == 1:
        y[i] = 1

theta = gradient_descent_p4(X, y, alpha, lam, T)
print(theta)


# Problem 5
def gradient_descent_p5(X, y, alpha, lamda, T):
    '''
    Inputs:
    numpy matrix X of shape (m, n)
    output vector y of shape (m, )
    scalar learning rate alpha
    regularization strength parameter lambda
    number of iterations T

    Output:
    a vector theta of shape (n + 1; 1).
    theta should be the logistic regression parameter vector found by executing
    the gradient descent algorithm for T iterations on the given inputs.'''

    m = len(X)
    n = len(X.loc[0])
    X = np.c_[np.ones((m, 1)), X]  # place a vector of ones in first column
    theta = np.random.rand(1, n + 1)  # Initial theta

    cost_lst = []
    for k in range(T):
        for j in range(n+1):
            # print(theta[0], X[0], y.loc[0])
            if y.loc[0] == 'M':
                yj = 1
            else:
                yj = 0
            summation = theta[0].dot(X[0]) - yj
            for i in range(1, m):
                if y.loc[1] == 'M':
                    yi = 1
                else:
                    yi = 0
                summation += (theta[0].dot(X[i]) - yi)*X[i][j]
            if j == 0:
                theta[0][j] = theta[0][j] - alpha * (1/m) * summation
            else:
                theta[0][j] = theta[0][j] - alpha * \
                    (1/m) * summation + (lamda/m)*theta[0][j]

        summation_cost = 0
        for i in range(m):
            if y.loc[1] == 'M':
                yi = 1
            else:
                yi = 0
            summation_cost += (theta.dot(X[i]) - yi) ** 2
        cost = summation_cost / (2 * m)
        cost_lst.append(cost)

    _, ax = plt.subplots()
    ax.set_ylabel('J(Theta), cost')
    ax.set_xlabel('T, itearations)')
    _ = ax.plot(range(T), cost_lst, 'b.')
    # plt.show()
    return theta


# Example cases for Problem 5
alpha = 0.1  # learning rate
lam = 0.5  # regularization parameter
T = 50  # number of iterations

df = pd.read_csv(r'C:\Users\hocke\OneDrive\Desktop\EricZacharia\03-Education\02-GraduateSchool\01-UChicago\01-Quarters\04-Spring2021\MPCS53111-MachineLearning\Homework\hw3\wdbc.data')
base_names = ['radius', 'texture', 'perimeter', 'area', 'smooth', 'compact', 'concav',
              'conpoints', 'symmetry', 'fracdim']
names = ['m' + name for name in base_names]
names += ['s' + name for name in base_names]
names += ['e' + name for name in base_names]
names = ['id', 'class'] + names
df.columns = names
df['color'] = pd.Series([(0 if x == 'M' else 1) for x in df['class']])
y = df['class']

X = df.drop(['id', 'class', 'color'], axis=1)

theta = gradient_descent_p5(X, y, alpha, lam, T)
print(theta)

# probelm 6
# X.isnull().sum()
log_model = LogisticRegression(max_iter=100000)
log_model.fit(X, y)
print(log_model.score(X, y))


# problem 7
my_color_map = mpl.colors.ListedColormap(['r', 'g'], 'mycolormap')

c1 = 'mradius'
c2 = 'mtexture'

clf = LogisticRegression(solver='lbfgs')
clf.fit(df[[c1, c2]], df['color'])

plt.scatter(df[c1], df[c2], c=df['color'], cmap=my_color_map)
plt.xlabel(c1)
plt.ylabel(c2)

x = np.linspace(df[c1].min(), df[c1].max(), 1000)
y = np.linspace(df[c2].min(), df[c2].max(), 1000)
xx, yy = np.meshgrid(x, y)
predicted_prob = clf.predict_proba(
    np.hstack((xx.ravel().reshape(-1, 1), yy.ravel().reshape(-1, 1))
              ))[:, 1]
predicted_prob = predicted_prob.reshape(xx.shape)

plt.contour(xx, yy, predicted_prob, [0.5], colors=['b'])
# plt.show()
