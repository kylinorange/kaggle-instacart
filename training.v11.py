
# coding: utf-8

# ![train](http://cliparting.com/wp-content/uploads/2016/06/Train-clipart-for-kids-free-free-clipart-images.gif)

# In[1]:


import numpy as np
np.random.seed(1019)

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().magic('matplotlib inline')

import xgboost

import sklearn
from sklearn.model_selection import train_test_split 

import sys, os, gc, types
import time
from subprocess import check_output


# In[2]:


sys.path.append('./utils')

from training import cv, train
from plotting import plot_importance
from data import Data


# In[3]:


root_paths = [
    "/data/kaggle-instacart",
    "/Users/jiayou/Dropbox/珺珺的程序/Kaggle/Instacart",
    "/Users/jiayou/Dropbox/Documents/珺珺的程序/Kaggle/Instacart"
]
root = None
for p in root_paths:
    if os.path.exists(p):
        root = p
        break


# In[ ]:





# # Hyper-Parameter Search

# In[4]:


name = 'v11-r0'
down_sample = None
test_size = 0.01
num_searches = 1
boosting_rounds = 3500
stopping_rounds = None
aug = True

xgb_params_search = {
#     "learning_rate"    : lambda: int(10**np.random.uniform(-2, -1) * 1e4) / 1e4,
#     "max_depth"        : lambda: np.random.randint(12, 13),
#     "subsample"        : [0.5],
    "tree_method"      : ["hist"],
    "seed"             : list(range(10000))
}


# -----------

# In[5]:


dtrain, dval = Data.dtrain_dval(down_sample=down_sample, test_size=test_size, aug=aug)
gc.collect()

print(dtrain.num_row(), dtrain.num_col())
print(dval.num_row(), dval.num_col())


# In[6]:


def get_params(default, search):
    np.random.seed(int(time.time()))
    p = dict(default)
    for k, gen in search.items():
        v = None
        if type(gen) == list:
            v = gen[np.random.randint(0, len(gen))]
        elif type(gen) == types.LambdaType:
            v = gen()
        p[k] = v
    return p

def print_params(params, keys):
    print()
    print(["{} = {}".format(k, params[k]) for k in keys])
    print()


# In[7]:


xgb_params_default = {
    "booster"          : "gbtree",
    "tree_method"      : "auto",
    "learning_rate"    : 0.1,
    "max_depth"        : 6,
    "min_child_weight" : 10, # hessian weight
    "subsample"        : 0.7,
    "colsample_bytree" : 0.9,
        
    "objective"        : "reg:logistic",
    "eval_metric"      : "logloss",
    
    "min_split_loss"   : 0.7, # ?
    "reg_alpha"        : 2e-05,
    "reg_lambda"       : 10
#     "grow_policy"      : ["lossguide"]
}


# In[8]:


def lr_schedule(env):
    bst, r = env.model, env.iteration
    if r == 300:
        bst.set_param('learning_rate', 0.05)
    elif r == 500:
        bst.set_param('learning_rate', 0.01)
        
def save_checkpoint(env):
    bst, r = env.model, env.iteration
    if r % 300 == 0:
        bst.save_model(os.path.join(root, 'train-checkpoint.bst'))


# In[9]:


results = []
for i in range(num_searches):
    xgb_params = get_params(default=xgb_params_default, search=xgb_params_search)
    print_params(xgb_params, keys=xgb_params_search.keys())
    
    h = {}
    callbacks = [xgboost.callback.record_evaluation(h)]
    callbacks.append(save_checkpoint)
    if stopping_rounds is not None:
        callbacks.append(xgboost.callback.early_stop(stopping_rounds=stopping_rounds))
    
    bst = train(
        xgb_params, dtrain, num_boost_round=boosting_rounds,
        evals=[(dtrain, 'train'), (dval, 'val')],
        callbacks=callbacks)
    
    bst.save_model(os.path.join(root, 'train-{}-n{}.bst'.format(name, i)))
    results.append([xgb_params, h])
    
    _, axes = plt.subplots(nrows=1, ncols=5, figsize=(25,30))
    measures = ['weight', 'gain', 'cover']
    for i in range(3):
        plot_importance(bst, height=1, ax=axes[2*i], importance_type=measures[i], title=measures[i])
    plt.show()


# ----

# In[ ]:


# Save search results
params = []
histories = []
for i in range(num_searches):
    p = dict(results[i][0])
    h = pd.DataFrame({
        'train-logloss': results[i][1]['train']['logloss'],
        'val-logloss': results[i][1]['val']['logloss']    
    })
    
    p['search_id'] = i
    p['boost_rounds'] = h.shape[0]
    p['last_val-logloss'] = h['val-logloss'][len(h) - 1]
    p['last_train-logloss'] = h['train-logloss'][len(h) - 1]
    params.append(p)
    
    h['search_id'] = i
    h['boost_round'] = range(h.shape[0])
    histories.append(h)
    
p = pd.DataFrame(params)
p.to_csv(os.path.join(root, 'train-{}-params.csv'.format(name)), index=False)

h = pd.concat(histories)
h.to_csv(os.path.join(root, 'train-{}-histories.csv'.format(name)), index=False)


# In[ ]:





# In[ ]:


plt.figure(figsize=(18, 12))
plt.ylim((0.23, 0.25))
for i in range(num_searches):
    plt.plot(h['boost_round'][h.search_id == i], h['val-logloss'][h.search_id == i])
    plt.plot(h['boost_round'][h.search_id == i], h['train-logloss'][h.search_id == i], '--')
plt.show()


# In[ ]:


p.sort_values(by='last_val-logloss')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




