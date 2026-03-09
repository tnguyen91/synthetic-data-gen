import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def Data_Loading_Adult(gen=0):
    # data loading
    Data1 = pd.read_csv('data/adult.data', header=None, delimiter=",")
    Data2 = pd.read_csv('data/adult.test', header=None, delimiter=",", skiprows=1)

    # Merge two datasets
    df = pd.concat((Data1, Data2), axis=0)

    # Define column names
    df.columns = ["Age", "WorkClass", "fnlwgt", "Education", "EducationNum",
                  "MaritalStatus", "Occupation", "Relationship", "Race", "Gender",
                  "CapitalGain", "CapitalLoss", "HoursPerWeek", "NativeCountry", "Income"]

    # Index reset due to merging
    df = df.reset_index()
    df = df.drop("index", axis=1)

    # Label define
    Y = np.ones([len(df), ])

    # Set >50K as 1 and <=50K as 0
    Y[df["Income"].index[df["Income"] == " <=50K"]] = 0
    Y[df["Income"].index[df["Income"] == " <=50K."]] = 0

    # Drop feature which can directly infer label
    df.drop("Income", axis=1, inplace=True, )

    df.drop("fnlwgt", axis=1, inplace=True, )
    df.drop("EducationNum", axis=1, inplace=True, )
    df.drop("CapitalGain", axis=1, inplace=True, )
    df.drop("CapitalLoss", axis=1, inplace=True, )

    # Transform the type from string to float
    df.Age = df.Age.astype(float)
    df.HoursPerWeek = df.HoursPerWeek.astype(float)

    # One hot encoding for categorical features
    df = pd.get_dummies(df, columns=["WorkClass", "Education", "MaritalStatus", "Occupation", "Relationship",
                                     "Race", "Gender", "NativeCountry"])

    # Treat data as numpy array
    X = np.asarray(df)

    # Normalization with Minmax Scaler
    for i in range(len(X[0, :])):
        X[:, i] = X[:, i] - np.min(X[:, i])
        X[:, i] = X[:, i] / (np.max(X[:, i]) + 1e-8)

    # Divide the data into train, valid, and test set (1/3 each)
    idx = np.random.permutation(len(Y))

    # train
    trainX = X[idx[:(2 * int(len(Y) / 3))], :]
    trainY = Y[idx[:(2 * int(len(Y) / 3))]]

    # test
    testX = X[idx[(2 * int(len(Y) / 3)):], :]
    testY = Y[idx[(2 * int(len(Y) / 3)):]]

    if gen == 0:
        # Return train, valid, and test sets
        return trainX, trainY, testX, testY
    else:
        return np.concatenate((trainX, trainY.reshape(-1, 1)), axis=1)


def Data_Loading_Law(seed, norm=False):
    # Only race as sensitive attribute, include gender as a feature
    df = pd.read_csv("data/law_data.csv", index_col=0)
    df = pd.get_dummies(df, columns=["race"], prefix="", prefix_sep="")

    df["male"] = df["sex"].map(lambda x: 1 if x == 2 else 0)
    df["female"] = df["sex"].map(lambda x: 1 if x == 1 else 0)
    df = df.drop(axis=1, columns=["sex"])
    df["LSAT"] = df["LSAT"].astype(int)

    df_train, df_test = train_test_split(df, random_state=seed, test_size=0.2)
    A = [
        "Amerindian",
        "Asian",
        "Black",
        "Hispanic",
        "Mexican",
        "Other",
        "Puertorican",
        "White",
    ]
    X_train = np.hstack(
        (
            df_train[A],
            np.array(df_train["UGPA"]).reshape(-1, 1),
            np.array(df_train["LSAT"]).reshape(-1, 1),
            np.array(df_train["male"]).reshape(-1, 1),
            np.array(df_train["female"]).reshape(-1, 1),
        ))

    y_train = df_train["ZFYA"]
    y_train = pd.Series.to_numpy(y_train)
    X_test = np.hstack(
        (
            df_test[A],
            np.array(df_test["UGPA"]).reshape(-1, 1),
            np.array(df_test["LSAT"]).reshape(-1, 1),
            np.array(df_test["male"]).reshape(-1, 1),
            np.array(df_test["female"]).reshape(-1, 1),
        ))
    norm_fac = []
    y_test = df_test["ZFYA"]
    y_test = pd.Series.to_numpy(y_test)
    norm_y = max(abs(y_train))
    y_train = y_train / max(abs(y_train))
    y_test = y_test / norm_y
    norm_fac.append(norm_y)
    if norm:
        X_mean = np.mean(X_train[:, 8:10], axis=0)
        norm_fac.append(X_mean)
        X_train[:, 8:10] = (X_train[:, 8:10] - X_mean)
        fac = np.abs(X_train[:, 8:10]).max(axis=0)
        norm_fac.append(fac)
        X_train[:, 8:10] = X_train[:, 8:10] / fac
        X_test[:, 8:10] = (X_test[:, 8:10] - X_mean)
        X_test[:, 8:10] = X_test[:, 8:10] / fac
    return X_train, y_train, X_test, y_test, df_train, A, norm_fac


def Data_Loading_GC(gen=0, seed=0):
    # data loading
    Data1 = pd.read_csv('data/Training_GC.csv', delimiter=",")
    Data1 = Data1[
        ['Sex...Marital.Status', 'Creditability', 'Account.Balance', 'Duration.of.Credit..month.', 'Credit.Amount']]
    Data1 = Data1.rename(columns={"Creditability": "label", "Sex...Marital.Status": "gender"})
    Data2 = pd.read_csv('data/Test_GC.csv', delimiter=",")
    Data2 = Data2[
        ['Sex...Marital.Status', 'Creditability', 'Account.Balance', 'Duration.of.Credit..month.', 'Credit.Amount']]
    Data2 = Data2.rename(columns={"Creditability": "label", "Sex...Marital.Status": "gender"})
    Data1 = Data1.loc[Data1['gender'] < 3]
    Data2 = Data2.loc[Data2['gender'] < 3]

    # Merge two datasets
    df = pd.concat((Data1, Data2), axis=0)

    # Index reset due to merging
    df = df.reset_index()
    df = df.drop("index", axis=1)

    # Label define
    Y = np.ones([len(df), ])

    # Set >50K as 1 and <=50K as 0
    Y[df["label"].index[df["label"] == 0]] = 0
    Y[df["label"].index[df["label"] == 1]] = 1

    # Drop feature which can directly infer label
    df.drop("label", axis=1, inplace=True, )

    # One hot encoding for categorical features
    df = pd.get_dummies(df, columns=["Account.Balance", 'gender'])
    print(df)
    # Treat data as numpy array
    X = np.asarray(df)
    print(X)

    # Normalization with Minmax Scaler
    for i in range(len(X[0, :])):
        X[:, i] = X[:, i] - np.min(X[:, i])
        X[:, i] = X[:, i] / (np.max(X[:, i]))

    # Divide the data into train, valid, and test set (1/3 each)
    np.random.seed(seed=seed)
    idx = np.random.permutation(len(Y))

    # train
    trainX = X[idx[:(2 * int(len(Y) / 3))], :]
    trainY = Y[idx[:(2 * int(len(Y) / 3))]]

    # test
    testX = X[idx[(2 * int(len(Y) / 3)):], :]
    testY = Y[idx[(2 * int(len(Y) / 3)):]]

    if gen == 0:
        # Return train, valid, and test sets
        return trainX, trainY, testX, testY
    else:
        return np.concatenate((trainX, trainY.reshape(-1, 1)), axis=1)


def Data_Loading_FC(gen=0, seed=1):
    # data loading
    df = pd.read_csv('data/crx.data', delimiter=",")
    df.columns = ["A1", "A2", "A3", "A4", "A5",
                  "A6", "A7", "A8", "A9", "A10",
                  "A11", "A12", "A13", "A14", "A15", "label"]

    # Index reset due to merging
    df = df.reset_index()
    df = df.drop("index", axis=1)

    # Label define
    # Set >50K as 1 and <=50K as 0
    df.loc[df["A4"] == 'u', 'A4'] = 0
    df.loc[df["A4"] == 'y', 'A4'] = 1
    df.loc[df["A4"] == 'l', 'A4'] = 0
    df.loc[df["A4"] == 't', 'A4'] = 0

    df = df.drop(df[df["A1"] == '?'].index)
    df = df.drop(df[df["A2"] == '?'].index)
    df = df.drop(df[df["A3"] == '?'].index)
    df = df.drop(df[df["A4"] == '?'].index)
    df = df.drop(df[df["A5"] == '?'].index)
    df = df.drop(df[df["A6"] == '?'].index)
    df = df.drop(df[df["A7"] == '?'].index)
    df = df.drop(df[df["A8"] == '?'].index)
    df = df.drop(df[df["A9"] == '?'].index)
    df = df.drop(df[df["A10"] == '?'].index)
    df = df.drop(df[df["A11"] == '?'].index)
    df = df.drop(df[df["A12"] == '?'].index)
    df = df.drop(df[df["A13"] == '?'].index)
    df = df.drop(df[df["A14"] == '?'].index)
    df = df.drop(df[df["A15"] == '?'].index)
    df = df.drop(df[df["label"] == '?'].index)
    Y = np.ones([len(df), ])
    Y[df["label"] == '-'] = 0
    Y[df["label"] == '+'] = 1
    l = sum((df["label"] == '+') & (df["A4"] == 1))
    np.random.seed(seed=seed)
    Y[(df["label"] == '+') & (df["A4"] == 1)] = np.random.choice(a=[0, 1], size=l, p=[0.5, 0.5])
    # Drop feature which can directly infer label
    df.drop("label", axis=1, inplace=True, )

    # One hot encoding for categorical features
    df = pd.get_dummies(df, columns=["A1", "A4", "A5",
                                     "A6", "A7", "A9", "A10",
                                     "A12", "A13"])
    print(df)

    df['A2'] = df['A2'].astype(float)
    df['A3'] = df['A3'].astype(float)
    df['A8'] = df['A8'].astype(float)
    df['A11'] = df['A11'].astype(float)
    df['A14'] = df['A14'].astype(float)
    df['A15'] = df['A15'].astype(float)

    # Treat data as numpy array
    X = np.asarray(df)

    # Normalization with Minmax Scaler
    for i in range(len(X[0, :])):
        if i in [8, 9]:
            X[:, i] = X[:, i] - np.min(X[:, i])
            X[:, i] = X[:, i] / (np.max(X[:, i]))
            continue
        X[:, i] = X[:, i] - np.min(X[:, i])
        X[:, i] = X[:, i] / (np.max(X[:, i]) + 1e-8)

    # Divide the data into train, valid, and test set (1/3 each)

    idx = np.random.permutation(len(Y))

    # train
    trainX = X[idx[:(2 * int(len(Y) / 3))], :]
    trainY = Y[idx[:(2 * int(len(Y) / 3))]]

    # test
    testX = X[idx[(2 * int(len(Y) / 3)):], :]
    testY = Y[idx[(2 * int(len(Y) / 3)):]]
    print(sum(trainY))
    if gen == 0:
        # Return train, valid, and test sets
        return trainX, trainY, testX, testY
    else:
        return np.concatenate((trainX, trainY.reshape(-1, 1)), axis=1)
