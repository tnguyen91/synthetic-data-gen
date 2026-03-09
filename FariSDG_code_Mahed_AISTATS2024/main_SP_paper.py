from spgan import train_spgan
import numpy as np
import data_loading
import utils
import argparse
import copy
#import pdb
#pdb.set_trace()


parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=50, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0005, help="learning rate")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=40, help="dimensionality of the latent space")
parser.add_argument("--lam", type=float, default=0.01, help="regularisation coef")
parser.add_argument("--n_critic", type=int, default=20, help="number of training steps for discriminator per iter")
parser.add_argument("--clip_value", type=float, default=0.01, help="lower and upper clip value for disc. weights")
parser.add_argument("--sample_interval", type=int, default=400, help="interval betwen image samples")
opt = parser.parse_args()
print(opt)

X_in = data_loading.Data_Loading_Adult(1)
X_gen = train_spgan(X_in, opt)
x_s = X_gen[:, :-1]
y_s = X_gen[:, -1] > 0.5

trainX, trainY, testX, testY = data_loading.Data_Loading_Adult()

### Check difference

model_1 = utils.supervised_model_training(trainX, trainY, 'randomforest')
trainX1 = copy.deepcopy(trainX)
testX1 = copy.deepcopy(testX)
trainX1[:,60:62] = 0
testX1[:,60:62] = 0

model_2 = utils.supervised_model_training(trainX1, trainY, 'randomforest')

acc_1 = utils.metrics.accuracy_score(testY, model_1.predict(testX))
acc_2 = utils.metrics.accuracy_score(testY, model_2.predict(testX1))
print(acc_1,acc_2)


# Based on gender
test_X_female = testX[np.where(testX[:, 60] > 0)]
testY_female = testY[np.where(testX[:, 60] > 0)]
test_X_male = testX[np.where(testX[:, 61] > 0)]
testY_male = testY[np.where(testX[:, 61] > 0)]


model_name = 'randomforest'
trainY = trainY.astype(int)
model = utils.supervised_model_training(trainX, trainY, model_name)
pre_r, rec_r, auc_r = utils.model_test(model, testX, testY, model_name)

model_s = utils.supervised_model_training(x_s, y_s, model_name)
per_s, rec_s, auc_s = utils.model_test(model_s, testX, testY, model_name)
print("The performance on real data is:", pre_r, rec_r, auc_r, "\nThe performance on synthetic data is:", per_s, rec_s,
      auc_s)


# Computing FTU fairness of the model
test_X_mail2 = copy.deepcopy(test_X_female)
test_X_mail2[:, 60] = 0
test_X_mail2[:, 61] = 1
test_X_female2 = copy.deepcopy(test_X_male)
test_X_female2[:, 61] = 0
test_X_female2[:, 60] = 1

test_X_male_com = np.concatenate((test_X_male, test_X_mail2), axis=0)
test_X_female_com = np.concatenate((test_X_female2, test_X_female), axis=0)

FTU_m = np.abs(np.mean(model_s.predict(test_X_male_com)) - np.mean(model_s.predict(test_X_female_com)))
print("FTU is:", FTU_m)


# Computing DP fairness of the model

DP_m = np.abs(np.mean(model_s.predict(test_X_male)) - np.mean(model_s.predict(test_X_female)))
print("DP is:", DP_m)



'''
based on gender
print((acc_r1 - acc_s1)/acc_r1)

auc_r, apr_r = utils.model_test(model, test_X_female, testY_female, model_name)
auc_s, apr_s = utils.model_test(model_s, test_X_female, testY_female, model_name)
print(auc_r, apr_r, auc_s, apr_s)

auc_r, apr_r = utils.model_test(model, test_X_male, testY_male, model_name)
auc_s, apr_s = utils.model_test(model_s, test_X_male, testY_male, model_name)
print(auc_r, apr_r, auc_s, apr_s)
'''
