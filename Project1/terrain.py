import numpy as np
import functions as f
import matplotlib.pyplot as plt
from imageio import imread
from matplotlib import  cm
from bootstrap import bootstrap
from mpl_toolkits.mplot3d import Axes3D
from cross_validation import cross_validation
from sklearn.preprocessing import StandardScaler

# Load the terrain
filename = 'SRTM_data_Kamloops.tif'
terrain = imread(filename)

# Show the terrain
# plt.figure()
# plt.title('Terrain over Minneapolis')
# plt.imshow(terrain, cmap='gray')
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.show()

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

# Initial data
n = 40
maxdegree = 23
n_bootstrap = 100
n_folds = 5
noise = 0.1
method = f.Ridge
lmbda = 0

# polydegree, MSE_bootstrap_test, MSE_bootstrap_train, bias_bootstrap, \
# variance_bootstrap = bootstrap(n, maxdegree, n_bootstrap, noise, method, \
#                                datatype='Terrain', filename=filename)

polydegree_cv, MSE_mean, MSE_best, R2Score_skl, R2Score_mean, beta_best, \
best_degree, MSE_mean_sklearn, best_degree_sklearn, beta_best_sklearn \
    = cross_validation(n, maxdegree, noise, n_folds, method, lmbda=lmbda, \
                       datatype='Terrain', filename=filename)


plt.style.use("ggplot")

plt.plot(polydegree_cv, MSE_mean, label=f"MSE mean")
plt.plot(polydegree_cv, MSE_mean_sklearn, label=f"MSE sklearn")
plt.xlabel('Model complexity')
plt.title('Error in terrain data')
plt.ylabel('MSE')
plt.legend()
plt.show()

print(best_degree, best_degree_sklearn)
print(f"cunt {MSE_best}, {MSE_mean}")
print(f"twat {MSE_mean_sklearn}")


# Normalize data
scaler = StandardScaler()                   # removes the mean and scales each feature/variable to unit variance
scaler.fit(terrain)                         # compute the mean and std to be used for later scaling
terrain_scaled = scaler.transform(terrain)  # perform standardization by centering and scaling

# Fixing a set of points
terrain_scaled = terrain_scaled[:n, :n]

# Create mesh of image pixel
x = np.sort(np.linspace(0, 1, terrain_scaled.shape[0]))
y = np.sort(np.linspace(0, 1, terrain_scaled.shape[1]))
x, y = np.meshgrid(x, y)

X = f.design_matrix(x, y, best_degree)
z_tilde = X @ beta_best
z_tilde = z_tilde.reshape(x.shape[0], x.shape[1])
print(f.MSE(terrain_scaled, z_tilde))

X_sklearn = f.design_matrix(x, y, best_degree_sklearn)
z_tilde_sklearn = X_sklearn @ beta_best_sklearn
z_tilde_sklearn = z_tilde_sklearn.reshape(x.shape[0], x.shape[1])
print(f.MSE(terrain_scaled, z_tilde_sklearn))


plt.subplot(311)
plt.title('Actual terrain data')
plt.imshow(terrain_scaled, cmap='gist_earth')

plt.subplot(312)
plt.title('Ridge regression with cross validation')
plt.imshow(z_tilde, cmap='gist_earth')

plt.subplot(313)
plt.title('Sklear ridge with cross validation')
plt.imshow(z_tilde_sklearn, cmap='gist_earth')
plt.show()


fig, ax = plt.subplots(3, 1)
fig.suptitle('terrain data, using ridge and cross validation')
cp1 = ax[0].contour(x, y, terrain_scaled)
ax[0].set_title('raw terrain data')
#fig.colorbar(cp1)

cp2 = ax[1].contour(x, y, z_tilde)
ax[1].set_title('our implementation')
#fig.colorbar(cp2)

cp3 = ax[2].contour(x, y, z_tilde_sklearn)
ax[2].set_title('scikit learn')

fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
fig.colorbar(cp3, cax=cbar_ax)
plt.show()
