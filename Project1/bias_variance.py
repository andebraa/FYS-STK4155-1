import numpy as np
import matplotlib.pyplot as plt
from bootstrap import bootstrap
from functions import OLS, Ridge
from cross_validation import cross_validation

# initial data
n = 50                 # number of data points
maxdegree = 14
noise = 0.1
#n_folds = 5               # number of folds
n_bootstrap = 100
lmbda = 0
seed = 7053

# Bootstrap OLS
polydegree_ols, MSE_bootstrap_test_ols, MSE_bootstrap_train_ols, \
bias_bootstrap_ols, variance_bootstrap_ols = \
            bootstrap(n, maxdegree, n_bootstrap, noise, method=OLS, seed=seed)

# Bootstrap Ridge
polydegree_b, MSE_bootstrap_test, MSE_bootstrap_train, bias_bootstrap, \
variance_bootstrap = \
            bootstrap(n, maxdegree, n_bootstrap, noise, method=Ridge, seed=seed)

n = 20
#bootstrap Ridge low num bootstraps
polydegree_b_low, MSE_bootstrap_test_low, MSE_bootstrap_train_low, bias_bootstrap_low, \
variance_bootstrap_low = \
            bootstrap(n, maxdegree, n_bootstrap, noise, method=Ridge, seed=seed)

# Bootstrap OLS low num bootstraps
polydegree_ols_low, MSE_bootstrap_test_ols_low, MSE_bootstrap_train_ols_low, \
bias_bootstrap_ols_low, variance_bootstrap_ols_low = \
            bootstrap(n, maxdegree, n_bootstrap, noise, method=OLS, seed=seed)


# plt.plot(polydegree_ols, bias_bootstrap_ols, label='OLS bias')
plt.style.use('ggplot')
#plt.plot(polydegree_b, MSE_bootstrap_test, label='Ridge MSE')
plt.plot(polydegree_b, bias_bootstrap, label='Ridge bias, 50 bootstraps')
#plt.plot(polydegree_b, variance_bootstrap, label='Ridge var')
plt.plot(polydegree_b_low, bias_bootstrap_low, label='Ridge bias, 20 bootstraps')
plt.xlabel('Model complexity', size=12)
plt.ylabel('Error', size=12)
plt.title('Bias of Ridge regression with different \n amount of bootstraps', size=18)
plt.legend()
plt.show()

# #plt.plot(polydegree_ols, MSE_bootstrap_test_ols, label='OLS MSE')
# plt.plot(polydegree_ols, bias_bootstrap_ols, label='OLS bias')
# #plt.plot(polydegree_ols, variance_bootstrap_ols, label='OLS var')
# plt.plot(polydegree_ols_low, bias_bootstrap_ols_low, label='ols bias with 10 bootstraps')
# plt.xlabel('Model complexity', size=12)
# plt.ylabel('Error', size=12)
# plt.title('Bias-variance trade-off', size=18)
# plt.legend()
# plt.show()


plt.plot(polydegree_ols, variance_bootstrap_ols, label='OLS var')
plt.plot(polydegree_b, variance_bootstrap, 'o', label='Ridge var')
plt.xlabel('Model complexity', size=12)
plt.ylabel('variance', size=12)
plt.title('Bias-variance trade-off', size=18)
plt.legend()
plt.show()
