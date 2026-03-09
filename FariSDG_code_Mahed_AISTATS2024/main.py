from cwgan import train_wgan
import numpy as np
import data_loading
import utils

X_in = data_loading.Data_Loading_Adult(1)
X_gen = train_wgan(X_in)
x_s = X_gen[:, :-1]
y_s = X_gen[:, -1] > 0.5

trainX, trainY, testX, testY = data_loading.Data_Loading_Adult()

# Based on gender
test_X_female = testX[np.where(testX[:, 64] > 0)]
testY_female = testY[np.where(testX[:, 64] > 0)]
test_X_male = testX[np.where(testX[:, 65] > 0)]
testY_male = testY[np.where(testX[:, 65] > 0)]

# Based on race
test_X_R1 = testX[np.where(testX[:, 59] > 0)]
test_Y_R1 = testY[np.where(testX[:, 59] > 0)]
test_X_R2 = testX[np.where(testX[:, 60] > 0)]
test_Y_R2 = testY[np.where(testX[:, 60] > 0)]
test_X_R3 = testX[np.where(testX[:, 61] > 0)]
test_Y_R3 = testY[np.where(testX[:, 61] > 0)]
test_X_R4 = testX[np.where(testX[:, 62] > 0)]
test_Y_R4 = testY[np.where(testX[:, 62] > 0)]
test_X_R5 = testX[np.where(testX[:, 63] > 0)]
test_Y_R5 = testY[np.where(testX[:, 63] > 0)]

model_name = 'logisticregression'
model = utils.supervised_model_training(trainX, trainY, model_name)
acc_r, auc_r, apr_r = utils.model_test(model, testX, testY, model_name)

model_s = utils.supervised_model_training(x_s, y_s, model_name)
acc_s, auc_s, apr_s = utils.model_test(model_s, testX, testY, model_name)
print("The performance on real data is:", acc_r, auc_r, apr_r, "\nThe performance on synthetic data is:", acc_s, auc_s,
      apr_s)
print("\n", (acc_r - acc_s)/acc_r)

acc_r1, auc_r1, apr_r1 = utils.model_test(model, test_X_R1, test_Y_R1, model_name)
acc_s1, auc_s1, apr_s1 = utils.model_test(model_s, test_X_R1, test_Y_R1, model_name)
print("The performance on real data is:", acc_r1, auc_r1, apr_r1, "\nThe performance on synthetic data is:", acc_s1,
      auc_s1, apr_s1)
print("\n", (acc_r1 - acc_s1)/acc_r1)

acc_r1, auc_r1, apr_r1 = utils.model_test(model, test_X_R2, test_Y_R2, model_name)
acc_s1, auc_s1, apr_s1 = utils.model_test(model_s, test_X_R2, test_Y_R2, model_name)
print("The performance on real data is:", acc_r1, auc_r1, apr_r1, "\nThe performance on synthetic data is:", acc_s1,
      auc_s1, apr_s1)
print("\n", (acc_r1 - acc_s1)/acc_r1)

acc_r1, auc_r1, apr_r1 = utils.model_test(model, test_X_R3, test_Y_R3, model_name)
acc_s1, auc_s1, apr_s1 = utils.model_test(model_s, test_X_R3, test_Y_R3, model_name)
print("The performance on real data is:", acc_r1, auc_r1, apr_r1, "\nThe performance on synthetic data is:", acc_s1,
      auc_s1, apr_s1)
print("\n", (acc_r1 - acc_s1)/acc_r1)

acc_r1, auc_r1, apr_r1 = utils.model_test(model, test_X_R4, test_Y_R4, model_name)
acc_s1, auc_s1, apr_s1 = utils.model_test(model_s, test_X_R4, test_Y_R4, model_name)
print("The performance on real data is:", acc_r1, auc_r1, apr_r1, "\nThe performance on synthetic data is:", acc_s1,
      auc_s1, apr_s1)
print("\n", (acc_r1 - acc_s1)/acc_r1)

acc_r1, auc_r1, apr_r1 = utils.model_test(model, test_X_R5, test_Y_R5, model_name)
acc_s1, auc_s1, apr_s1 = utils.model_test(model_s, test_X_R5, test_Y_R5, model_name)
print("The performance on real data is:", acc_r1, auc_r1, apr_r1, "\nThe performance on synthetic data is:", acc_s1,
      auc_s1, apr_s1)
print("\n", (acc_r1 - acc_s1)/acc_r1)
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
