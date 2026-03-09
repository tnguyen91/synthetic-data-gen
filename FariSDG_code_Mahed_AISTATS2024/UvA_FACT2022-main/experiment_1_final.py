#!/usr/bin/env python
# coding: utf-8

# # Reproduction of Adult dataset experiments
# 
# In this notebook we reproduce the results from Table 2 of the DECAF paper. We compare various methods for generating debiased data using the DECAF model against synthetic data generated using benchmark models GAN, WGAN-GP and FairGAN. As described in the paper we run all experiments (as implemented in this notebook) 10 times and avarage the results.

# In[8]:


import zlib
import sys
import pickle
import numpy as np

repNum = float(sys.argv[1])


# In[ ]:


import numpy
import random as rd

with open("random_draw_Final/data_"+str(repNum)+'.txt','w') as f:
    f.write(str([rd.random(),numpy.random.random()]))



# In[6]:


from sklearn.metrics import precision_score, recall_score, roc_auc_score
from sklearn.metrics import confusion_matrix


from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


from data import load_adult, preprocess_adult
from metrics import DP, FTU, FTU_alt, FTU_prob
from train import train_decaf, train_fairgan, train_vanilla_gan, train_wgan_gp


# ## Loading data

# In[ ]:


dataset = load_adult()
dataset.head()


# Preprocess the data next in order to make it suitable for training models on.

# In[ ]:


dataset = preprocess_adult(dataset)
dataset.head()


# Split the dataset into train and test folds. Test fold size is 2000.

# In[ ]:


# Split data into train and testing sets
dataset_train, dataset_test = train_test_split(dataset, test_size=2000,
                                               stratify=dataset['income'])

print('Size of train set:', len(dataset_train))
print('Size of test set:', len(dataset_test))


# In[ ]:


results = {}


# In[ ]:


results['split'] = (dataset_train, dataset_test)


# ### Defining the DAG
# 
# We need to define a DAG which captures the biases of the dataset. As described in the DECAF paper normally a causal discovery algorithm is used. In this notebook we simply copy the DAG which as described in the Zhang et al. paper which is the one also used in the DECAF paper.

# In[ ]:


# Define DAG for Adult dataset
dag = [
    # Edges from race
    ['race', 'occupation'],
    ['race', 'income'],
    ['race', 'hours-per-week'],
    ['race', 'education'],
    ['race', 'marital-status'],

    # Edges from age
    ['age', 'occupation'],
    ['age', 'hours-per-week'],
    ['age', 'income'],
    ['age', 'workclass'],
    ['age', 'marital-status'],
    ['age', 'education'],
    ['age', 'relationship'],
    
    # Edges from sex
    ['sex', 'occupation'],
    ['sex', 'marital-status'],
    ['sex', 'income'],
    ['sex', 'workclass'],
    ['sex', 'education'],
    ['sex', 'relationship'],
    
    # Edges from native country
    ['native-country', 'marital-status'],
    ['native-country', 'hours-per-week'],
    ['native-country', 'education'],
    ['native-country', 'workclass'],
    ['native-country', 'income'],
    ['native-country', 'relationship'],
    
    # Edges from marital status
    ['marital-status', 'occupation'],
    ['marital-status', 'hours-per-week'],
    ['marital-status', 'income'],
    ['marital-status', 'workclass'],
    ['marital-status', 'relationship'],
    ['marital-status', 'education'],
    
    # Edges from education
    ['education', 'occupation'],
    ['education', 'hours-per-week'],
    ['education', 'income'],
    ['education', 'workclass'],
    ['education', 'relationship'],
    
    # All remaining edges
    ['occupation', 'income'],
    ['hours-per-week', 'income'],
    ['workclass', 'income'],
    ['relationship', 'income'],
]

def dag_to_idx(df, dag):
    """Convert columns in a DAG to the corresponding indices."""

    dag_idx = []
    for edge in dag:
        dag_idx.append([df.columns.get_loc(edge[0]), df.columns.get_loc(edge[1])])

    return dag_idx

# Convert the DAG to one that can be provided to the DECAF model
dag_seed = dag_to_idx(dataset, dag)
print(dag_seed)


# It's also necessary to define edges we want to remove from the DAG in order to meet the various fairness criteria described in the paper.

# In[ ]:


def create_bias_dict(df, edge_map):
    """
    Convert the given edge tuples to a bias dict used for generating
    debiased synthetic data.
    """
    bias_dict = {}
    for key, val in edge_map.items():
        bias_dict[df.columns.get_loc(key)] = [df.columns.get_loc(f) for f in val]
    
    return bias_dict

# Bias dictionary to satisfy FTU
bias_dict_ftu = create_bias_dict(dataset, {'income': ['sex']})
print('Bias dict FTU:', bias_dict_ftu)

# Bias dictionary to satisfy DP
bias_dict_dp = create_bias_dict(dataset, {'income': [
    'occupation', 'hours-per-week', 'marital-status', 'education', 'sex',
    'workclass', 'relationship']})
print('Bias dict DP:', bias_dict_dp)

# Bias dictionary to satisfy CF
bias_dict_cf = create_bias_dict(dataset, {'income': [
    'marital-status', 'sex']})
print('Bias dict CF:', bias_dict_cf)


# ## Experiments
# 
# We have loaded and preprocessed the data and we are ready to run the experiments. For each experiment we train a generative model, sample synthetic data from the trained model and then obtain metrics by training and evaluating a downstream multi-layer perceptron using the test fold we generated in the previous section. We use the MLP model from `sklearn` with default parameters which matches the settings described in Appendix D of the paper.

# In[7]:


def eval_model(dataset_train, dataset_test,classifer="MLP"):
    """Helper function that prints evaluation metrics."""

    X_train, y_train = dataset_train.drop(columns=['income']), dataset_train['income']
    X_test, y_test = dataset_test.drop(columns=['income']), dataset_test['income']

    if classifer == "MLP":
        clf = MLPClassifier(max_iter=1000)
    elif classifer == "LR":
        clf = LogisticRegression(max_iter=1000)
    else:
        raise Exception("Unknown classifier")
        
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_pred_proba = clf.predict_proba(X_test)
    
  
    
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    auroc = roc_auc_score(y_test, y_pred_proba[:,1])
    
    aurocOLD = roc_auc_score(y_test, y_pred)

    
    dp = DP(clf, X_test)
    ftu = FTU(clf, X_test)
    
    ftu_alt = FTU_alt(clf,X_test)
    
    ftu_prob = FTU_prob(clf,X_test)
    conf = confusion_matrix(y_test, y_pred)

    X_train_alt = X_train.assign(sex=0)
    X_test_alt = X_test.assign(sex=0)
    
    if classifer == "MLP":
        clf1 = MLPClassifier(max_iter=1000)
    elif classifer == "LR":
        clf1 = LogisticRegression(max_iter=1000)
    else:
        raise Exception("Unknown classifier")
    clf1.fit(X_train_alt, y_train)
    
    y_pred_alt = clf1.predict(X_test_alt)
    
    y_pred_alt = y_pred_alt.astype(float)                                                 
    y_pred = y_pred.astype(float)                                                 
                                                                                    
    ftu_m = np.mean(abs(y_pred-y_pred_alt)) 
    
    return {'precision': precision, 'recall': recall, 'auroc': auroc, "aurocOLD": aurocOLD,
            'dp': dp, 'ftu': ftu, "ftu_alt":ftu_alt,"ftu_m":ftu_m, "ftu_prob":ftu_prob, "confMatrix":conf}


# ### Original dataset
# 
# As a benchmark we want to first train the downstream model on the original dataset.

# In[ ]:


res = eval_model(dataset_train, dataset_test)
res1 = eval_model(dataset_train, dataset_test,"LR")


# In[ ]:


results['original'] = {"MLP":res,"LR":res1}
print('original done')


# In the following sections we train various models in order to reproduce the results from Table 2 of the DECAF paper.

# ### GAN

# In[ ]:


synth_data = train_vanilla_gan(dataset_train)
synth_data.head()


# In[ ]:


res = eval_model(synth_data, dataset_test)
res1 = eval_model(synth_data, dataset_test,"LR")


# In[ ]:


results['GAN'] = {"MLP":res,"LR":res1}
print('GAN done')


# ### WGAN-GP

# In[ ]:


synth_data = train_wgan_gp(dataset_train)
synth_data.head()


# In[ ]:


res = eval_model(synth_data, dataset_test)
res1 = eval_model(synth_data, dataset_test,"LR")


# In[ ]:


results['WGAN-GP'] = {"MLP":res,"LR":res1}
print('WGAN-GP done')


# ### FairGAN

# In[ ]:


synth_data = train_fairgan(dataset_train)
synth_data.head()


# In[ ]:


res = eval_model(synth_data, dataset_test)
res1 = eval_model(synth_data, dataset_test,"LR")


# In[ ]:


results['FairGAN'] = {"MLP":res,"LR":res1}
print('FairGAN done')


# ### DECAF

# #### DECAF-ND

# In[ ]:


synth_data = train_decaf(dataset_train, dag_seed)
synth_data.head()


# In[ ]:


res = eval_model(synth_data, dataset_test)
res1 = eval_model(synth_data, dataset_test,"LR")


# In[ ]:


results['DECAF-ND'] = {"MLP":res,"LR":res1}
print('DECAF-ND done')


# #### DECAF-FTU

# In[ ]:


synth_data = train_decaf(dataset_train, dag_seed, biased_edges=bias_dict_ftu)
synth_data.head()


# In[ ]:


res = eval_model(synth_data, dataset_test)
res1 = eval_model(synth_data, dataset_test,"LR")


# In[ ]:


results['DECAF-FTU'] = {"MLP":res,"LR":res1}
print('DECAF-FTU done')


# #### DECAF-CF

# In[ ]:


synth_data = train_decaf(dataset_train, dag_seed, biased_edges=bias_dict_cf)
synth_data.head()


# In[ ]:


res = eval_model(synth_data, dataset_test)
res1 = eval_model(synth_data, dataset_test,"LR")


# In[ ]:


results['DECAF-CF'] = {"MLP":res,"LR":res1}
print('DECAF-CF done')


# #### DECAF-DP

# In[ ]:


synth_data = train_decaf(dataset_train, dag_seed, biased_edges=bias_dict_dp)
synth_data.head()


# In[ ]:


res = eval_model(synth_data, dataset_test)
res1 = eval_model(synth_data, dataset_test,"LR")


# In[ ]:


results['DECAF-DP'] = {"MLP":res,"LR":res1}
print('DECAF-DP done')


# #### SpGan

# In[ ]:


from spgan import spgan


# In[ ]:


from importlib import reload


# In[ ]:


reload(spgan)


# In[ ]:


class optC:
    pass


# In[ ]:


import pandas as pd
def helper(lam):
    opt.lam = lam
    res = spgan.train_spgan(dataset_train.to_numpy(),opt)
    q1 = pd.DataFrame(res)
    q1.columns = dataset_train.columns
    q1['income'] = q1['income']>0.5
    res = eval_model(q1, dataset_test)
    res1 = eval_model(q1, dataset_test,"LR")
    return {"MLP":res,"LR":res1}


# ### Run at multiple lambdas

# In[ ]:


opt = optC()
opt.latent_dim = 40
opt.batch_size = 64
opt.lr = 0.0005
opt.n_epochs = 500
opt.n_critic = 20
opt.clip_value = 0.01


# In[ ]:


results["spgan-0.0"] = helper(0.0)
results["spgan-0.001"] = helper(0.001)
results["spgan-0.002"] = helper(0.002)
results["spgan-0.003"] = helper(0.003)
results["spgan-0.004"] = helper(0.004)
results["spgan-0.005"] = helper(0.005)
results["spgan-0.006"] = helper(0.006)
results["spgan-0.007"] = helper(0.007)
results["spgan-0.008"] = helper(0.008)
results["spgan-0.009"] = helper(0.009)
results["spgan-0.01"] = helper(0.1)
print('SPGAN done')


# ## Ifgan

# In[5]:


from ifgan import ifgan_final as ifgan


# In[ ]:


import pandas as pd
import pickle
import zlib
def helper_ifgan(lam):
    opt.lam = lam
    res, s = ifgan.train_ifgan(dataset_train.to_numpy(),opt,storeStats=True)
    stats = zlib.compress(pickle.dumps(s,protocol=5))
    q1 = pd.DataFrame(res)
    q1.columns = dataset_train.columns
    q1['income'] = q1['income']>0.5
    res = eval_model(q1, dataset_test)
    res1 = eval_model(q1, dataset_test,"LR")
    return {"MLP":res,"LR":res1, "stats":stats}


# In[ ]:


results["ifgan-0"] = helper_ifgan(0.0)
results["ifgan-10-4"] = helper_ifgan(10**(-4))
results["ifgan-10-3"] = helper_ifgan(10**(-3))
results["ifgan-10-2"] = helper_ifgan(10**(-2))
results["ifgan-10-1"] = helper_ifgan(10**(-1))
results["ifgan-10-0"] = helper_ifgan(10**(0))
results["ifgan-10+1"] = helper_ifgan(10**(1))
print('IFGAN done')


# ## Ifgan-CF

# In[5]:


from ifgan import ifgan_mod as ifganMod


# In[ ]:


import pandas as pd
import pickle
import zlib
def helper_ifgan_mod(lam):
    opt.lam = lam
    res,s = ifganMod.train_ifgan(dataset_train.to_numpy(),opt,storeStats=True)
    stats = zlib.compress(pickle.dumps(s,protocol=5))
    q1 = pd.DataFrame(res)
    q1.columns = dataset_train.columns
    q1['income'] = q1['income']>0.5
    res = eval_model(q1, dataset_test)
    res1 = eval_model(q1, dataset_test,"LR")
    return {"MLP":res,"LR":res1, "stats":stats}


# In[ ]:


results["ifganCF-0"] = helper_ifgan(0.0)
results["ifganCF-10-4"] = helper_ifgan(10**(-4))
results["ifganCF-10-3"] = helper_ifgan(10**(-3))
results["ifganCF-10-2"] = helper_ifgan(10**(-2))
results["ifganCF-10-1"] = helper_ifgan(10**(-1))
results["ifganCF-10-0"] = helper_ifgan(10**(0))
results["ifganCF-10+1"] = helper_ifgan(10**(1))
print('IFGAN-cf done')


# ## Save data

# In[ ]:


import pickle


# In[ ]:


with open("resultsFinal/data_"+str(repNum)+'.p','wb') as f:
    pickle.dump(results,f)

