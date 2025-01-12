import sys
sys.path.insert(1, '/home/chene5/ENVI_new_copy/scenvi')
from utils import *
from ENVI import *

import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns

import sklearn.neighbors
import scipy.sparse
import umap.umap_ as umap

st_data = sc.read_h5ad('/data/peer/chene5/scenvi/st_data.h5ad')
sc_data = sc.read_h5ad('/data/peer/chene5/scenvi/sc_data.h5ad')


# train test split of spatial data
split = st_data.shape[0] // 10 * 8

train_idx = np.random.choice(st_data.shape[0], split, replace=False)
test_idx = np.setdiff1d(np.arange(st_data.shape[0]), train_idx)

train_st_data, test_st_data = st_data[train_idx], st_data[test_idx]




envi_model = ENVI(spatial_data = train_st_data, sc_data = sc_data)


i,j = np.random.randint(0,envi_model.spatial_data.obsm['scaled_niche'].shape[1],2)
print("between random actual scaled niches", S2(envi_model.spatial_data.obsm['scaled_niche'][i],envi_model.spatial_data.obsm['scaled_niche'][j], 1e-2, True))


envi_model.start_train()
# first two should be neg, last two should be pos


# test compute
test_st_data.obsm["scaled_niche"], min, max = compute_niche(test_st_data, envi_model.n_niche_genes, envi_model.k_nearest, envi_model.spatial_key, envi_model.batch_key)

# test inference
envi_model.infer_niche_x(test_st_data, "spatial")
envi_model.infer_niche_x(envi_model.spatial_data, "spatial")


pretrain_mean_train_OT, pretrain_train_OTs = envi_model.batched_S2(envi_model.spatial_data.obsm['scaled_niche'],envi_model.spatial_data.obsm['inferred_scaled_niche'], 1e-2, True)
pretrain_mean_test_OT, pretrain_test_OTs = envi_model.batched_S2(test_st_data.obsm['scaled_niche'],test_st_data.obsm['inferred_scaled_niche'], 1e-2, True)
print("pretrain_mean_train_OT", pretrain_mean_train_OT)
print("pretrain_mean_test_OT", pretrain_mean_test_OT)



print("training")
envi_model.continue_train(training_steps = 10000)



# test inference
envi_model.infer_niche_x(test_st_data, "spatial")
envi_model.infer_niche_x(envi_model.spatial_data, "spatial")
envi_model.infer_niche_x(envi_model.sc_data, "sc")

posttrain_mean_train_OT, posttrain_train_OTs = envi_model.batched_S2(envi_model.spatial_data.obsm['scaled_niche'],envi_model.spatial_data.obsm['inferred_scaled_niche'], 1e-2, True)
posttrain_mean_test_OT, posttrain_test_OTs = envi_model.batched_S2(test_st_data.obsm['scaled_niche'],test_st_data.obsm['inferred_scaled_niche'], 1e-2, True)
print("posttrain_mean_train_OT", posttrain_mean_train_OT)
print("posttrain_mean_test_OT", posttrain_mean_test_OT)


# save anndata
envi_model.spatial_data.write('/data/peer/chene/scenvi/train_st_data.h5ad')
test_st_data.write('/data/peer/chene/scenvi/test_st_data.h5ad')
envi_model.sc_data.write('/data/peer/chene/scenvi/sc_data.h5ad')












#%%
num_genes = 10
eps=1e-6


a = 2*(np.random.rand(10, num_genes) - 0.5) / sqrt(num_genes)
b = np.random.permutation(a) / sqrt(num_genes)
c = (a + 0.01 )/ sqrt(num_genes)
d = (a + 0.1)/ sqrt(num_genes)
e = 2*(np.random.rand(10, num_genes) - 0.5)/ sqrt(num_genes)


d1 = S2(a,b, eps, True)
d2 = S2(a,c, eps, True)
d3 = S2(a,d, eps, True)
d4 = S2(a,e, eps, True)
print("epsilon = ", eps)
d1, d2, d3, d4

#%%
# import scanpy as sc
# import matplotlib.pyplot as plt
# import seaborn as sns
# import umap.umap_ as umap

# # to prevent out of memory error -> just don't run multiple at the same time
# # import os
# # os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"
# # os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"]=".XX"
# # os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"]="platform"


# st_data = sc.read_h5ad('/data/peer/chene/scenvi/st_data.h5ad')
# sc_data = sc.read_h5ad('/data/peer/chene/scenvi/sc_data.h5ad')

# envi_model = ENVI(spatial_data = st_data, sc_data = sc_data) # using ENVI from ENVI.py

# envi_model.train(training_steps = 2)

# envi_model.latent_rep()
# envi_model.infer_niches()
# print("done")
# # envi_model.sc_data.obs['niches']
# %%
