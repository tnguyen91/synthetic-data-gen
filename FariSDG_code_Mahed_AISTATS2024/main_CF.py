from CF_GAN import train_cfgan
import argparse
import utils
from sklearn.metrics import mean_squared_error
import numpy as np
from data_loading import Data_Loading_Law
import matplotlib.pyplot as plt
import pickle
from scipy.stats import gaussian_kde
#import seaborn as sns
#from plotutils import *

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=50, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="learning rate")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=5, help="dimensionality of the latent space")
parser.add_argument("--lam", type=float, default=0.004, help="cf regularisation")
parser.add_argument("--n_critic", type=int, default=5, help="number of training steps for discriminator per iter")
parser.add_argument("--clip_value", type=float, default=0.01, help="lower and upper clip value for disc. weights")
parser.add_argument("--sensitive_dim", type=int, default=8, help="dimension of sensitive attributes")
parser.add_argument("--seed", type=int, default=0, help="seed")

opt = parser.parse_args()
print(opt)
np.random.seed(opt.seed)
# Creating training and test datasets
X_train, y_train, X_test, y_test, df_train, A, norm_fac = Data_Loading_Law(opt.seed, norm=True)

X_train_unaware = X_train[:, 8:]  # Removing race columns
# X_train_unaware = X_train_unaware[:, :-2]  # removing gender

X_test_unaware = X_test[:, 8:]  # removing gender
# X_test_unaware = X_test_unaware[:, :-2]   # removing gender

X_in = np.concatenate((X_train, y_train.reshape(-1, 1)), axis=1)
X_fair = train_cfgan(X_in, opt,opt.seed)
X_fair = X_fair.numpy()
y_fair = X_fair[:, -1]
X_fair = X_fair[:, 0:12]

# linearregression or randomforestReg
model_type = 'linearregression'
model_full = utils.supervised_model_training(X_train, y_train, model_type,opt.seed)
model_unaware = utils.supervised_model_training(X_train_unaware, y_train, model_type,opt.seed)
model_fair = utils.supervised_model_training(X_fair, y_fair, model_type,opt.seed)

# Evaluation
predictions_full = model_full.predict(X_test)
RMSE_full = np.sqrt(mean_squared_error(y_test, predictions_full))

predictions_unaware = model_unaware.predict(X_test_unaware)
RMSE_unaware = np.sqrt(mean_squared_error(y_test, predictions_unaware))

predictions_fair = model_fair.predict(X_test)
RMSE_fair = np.sqrt(mean_squared_error(y_test, predictions_fair))

print("RMSE of full model is", RMSE_full)
print("RMSE of unaware model is", RMSE_unaware)
print("RMSE of fair model is", RMSE_fair)

# Training counterfactual simulator

Race_dict = {"Amerindian": np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
             "Asian": np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
             "Black": np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
             "Hispanic": np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]),
             "Mexican": np.array([0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
             "Other": np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]),
             "Puertorican": np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]),
             "White": np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
             }

with open('k_est.pkl', 'rb') as file:
    # Call load method to deserialze
    k_est = pickle.load(file)

sigma = 0.05
N = len(k_est)
k_est = k_est.reshape(-1,1) + np.random.normal(0, sigma, [N,1])

# Figure 1 Black vs White
y_full_f = []
y_full_cf = []
y_full_acf = []
y_unaware_f = []
y_unaware_cf = []
y_unaware_acf = []
y_fair_f = []
y_fair_cf = []
y_fair_acf = []
norm_y = norm_fac[0]
count1 = 0
count2 = 0
count3 = 0

for i in range(X_test.shape[0]):
    if np.array_equal(X_test[i, 0:8], Race_dict["White"]) and count1 < 250:
        count1 += 1
        a = np.concatenate((Race_dict["Black"], X_test[i, 10:12]))
        x_cf, y_cf = utils.cf_sample(a, k_est[i], norm_fac)
        aa = np.concatenate((Race_dict["Asian"], X_test[i, 10:12]))
        x_acf, y_acf = utils.cf_sample(aa, k_est[i], norm_fac)

        y_full_f.append(norm_y * model_full.predict(X_test[i, :].reshape(1, -1)))
        y_full_cf.append(norm_y * model_full.predict(x_cf))
        y_full_acf.append(norm_y * model_full.predict(x_acf))

        y_unaware_f.append(norm_y * model_unaware.predict(X_test_unaware[i, :].reshape(1, -1)))
        y_unaware_cf.append(norm_y * model_unaware.predict(x_cf[:, 8:]))
        y_unaware_acf.append(norm_y * model_unaware.predict(x_acf[:, 8:]))

        y_fair_f.append(norm_y * model_fair.predict(X_test[i, :].reshape(1, -1)))
        y_fair_cf.append(norm_y * model_fair.predict(x_cf))
        y_fair_acf.append(norm_y * model_fair.predict(x_acf))

    elif np.array_equal(X_test[i, 0:8], Race_dict["Black"]) and count2 < 250:
        count2 += 1
        a = np.concatenate((Race_dict["White"], X_test[i, 10:12]))
        x_cf, y_cf = utils.cf_sample(a, k_est[i], norm_fac)

        y_full_cf.append(norm_y * model_full.predict(X_test[i, :].reshape(1, -1)))
        y_full_f.append(norm_y * model_full.predict(x_cf))

        y_unaware_cf.append(norm_y * model_unaware.predict(X_test_unaware[i, :].reshape(1, -1)))
        y_unaware_f.append(norm_y * model_unaware.predict(x_cf[:, 8:]))

        y_fair_cf.append(norm_y * model_fair.predict(X_test[i, :].reshape(1, -1)))
        y_fair_f.append(norm_y * model_fair.predict(x_cf))

    elif np.array_equal(X_test[i, 0:8], Race_dict["Black"]) and count3 < 250:
        count3 = 0
        a = np.concatenate((Race_dict["White"], X_test[i, 10:12]))
        x_cf, y_cf = utils.cf_sample(a, k_est[i], norm_fac)

        y_full_acf.append(norm_y * model_full.predict(X_test[i, :].reshape(1, -1)))
        y_full_f.append(norm_y * model_full.predict(x_cf))

        y_unaware_acf.append(norm_y * model_unaware.predict(X_test_unaware[i, :].reshape(1, -1)))
        y_unaware_f.append(norm_y * model_unaware.predict(x_cf[:, 8:]))

        y_fair_acf.append(norm_y * model_fair.predict(X_test[i, :].reshape(1, -1)))
        y_fair_f.append(norm_y * model_fair.predict(x_cf))

fig, ax = plt.subplots(nrows=2, ncols=3)
xs = np.linspace(-3, 3, 400)
xf = np.linspace(-2.5, 2.5, 400)
bw = 0.4

y_full_f = gaussian_kde(np.array(y_full_f).reshape(-1), bw)
y_full_cf = gaussian_kde(np.array(y_full_cf).reshape(-1), bw)
ax[0][0].plot(xs, y_full_f(xs), label="factual")
ax[0][0].plot(xs, y_full_cf(xs), label="counterfactual")
ax[0][0].set_title("Full model")
ax[0][0].set_ylabel("White/Black")
# ax[0][0].legend()

y_unaware_f = gaussian_kde(np.array(y_unaware_f).reshape(-1), bw)
y_unaware_cf = gaussian_kde(np.array(y_unaware_cf).reshape(-1), bw)
ax[0][1].plot(xs, y_unaware_f(xs), label="factual")
ax[0][1].plot(xs, y_unaware_cf(xs), label="counterfactual")
ax[0][1].set_title("Unaware model")
# ax[0][1].legend()

y_fair_f = gaussian_kde(np.array(y_fair_f).reshape(-1), bw)
y_fair_cf = gaussian_kde(np.array(y_fair_cf).reshape(-1), bw)
ax[0][2].plot(xf, y_fair_f(xf), label="factual")
ax[0][2].plot(xf, y_fair_cf(xf), label="counterfactual")
ax[0][2].set_title("Fair model")
# ax[0][2].legend()

y_full_acf = gaussian_kde(np.array(y_full_acf).reshape(-1), bw)
ax[1][0].plot(xs, y_full_f(xs), label="factual")
ax[1][0].plot(xs, y_full_acf(xs), label="counterfactual")
# ax[1][0].legend()
ax[1][0].set_ylabel("White/Asian")
ax[1][0].set_xlabel('FYA')

y_unaware_acf = gaussian_kde(np.array(y_unaware_acf).reshape(-1), bw)
l1 = ax[1][1].plot(xs, y_unaware_f(xs), label="factual")
l2 = ax[1][1].plot(xs, y_unaware_acf(xs), label="counterfactual")
ax[1][1].set_xlabel('FYA')
# ax[1][1].legend()

y_fair_acf = gaussian_kde(np.array(y_fair_acf).reshape(-1), bw)
ax[1][2].plot(xf, y_fair_f(xf), label="factual")
ax[1][2].plot(xf, y_fair_acf(xf), label="counterfactual")
ax[1][2].set_xlabel('FYA')
ax[1][2].legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.tight_layout()

plt.show()
