# %% [markdown]
# # Installation
# Install required packages for deepCOLOR analysis

# %%
import os
os.chdir('/home/tiisaishima/packages/deepcolor')


# %% [markdown]
# # Preprocessing data
# Although deepCOLOR estimate the spatial distribution of single cells based on raw count matrix of single cell and spatail transcriptomes, it requires normalized expression for downstream analysis such as DEG detection. Also, we exclude lowly expressed cells and spots from folowing analysis.
# 

# %%
import torch
import scanpy as sc
import numpy as np
np.random.seed(1)
torch.manual_seed(1)

# %%
sc_adata = sc.read_h5ad('data/tutorial/scc_sc.h5ad')
sp_adata = sc.read_h5ad('data/tutorial/scc_sp.h5ad')

# %% [markdown]
# You need to align gene names for both sinlge cell and spatial `AnnData` objects. We also excluede genes lowly expressed.

# %%
import numpy as np
sc_adata = sc_adata[:, sc_adata.layers['count'].toarray().sum(axis=0) > 10]
sp_adata = sp_adata[:, sp_adata.layers['count'].toarray().sum(axis=0) > 10]
common_genes = np.intersect1d(sc_adata.var_names, sp_adata.var_names)
sc_adata = sc_adata[:, common_genes]
sp_adata = sp_adata[:, common_genes]

# %% [markdown]
# # Estimate Spatial distribution
# Estimation of spatial distribution is conduted for each singel cell. For this process, we strongly recommend you to conduct the caluculation in GPU availabel environments.

# %%
# conduct spatial estimation
import deepcolor
import importlib
importlib.reload(deepcolor)
sc_adata, sp_adata = deepcolor.estimate_spatial_distribution(sc_adata, sp_adata, param_save_path='data/tutorial/opt_params.pt', first_epoch=500, second_epoch=500)


# %% [markdown]
# Here, we will display imputed expression patterns, combining estiamted spatial distribution of single cells with its expression profiles.

# %%
from matplotlib import pyplot as plt
sp_adata = deepcolor.calculate_imputed_spatial_expression(sc_adata, sp_adata)
sc.pl.spatial(sp_adata, color="CD8A", layer='imputed_exp')


# %%
sc.pl.spatial(sp_adata, color="CD8A", layer='count')

# %% [markdown]
# The first panel represents the imputed CD8A expression, while the second panel represents original expression. It is shown that imputed expression is more smooth than original one.

# %% [markdown]
# While deepCOLOR estimate spatial distributions in single cell level, it can estimate the spatial distribution of spcified cell popluation by aggregating the distribution of belongin cells.

# %%
sp_adata = deepcolor.calculate_clusterwise_distribution(sc_adata, sp_adata, 'level3_celltype')
sc.pl.spatial(sp_adata, color='CD8_EM')

# %% [markdown]
# # Colocalization based ligand receptor interaction

# %% [markdown]
# Using ligand-target regulatory potentianl derived from NicheNet, we can estimate the ligand activity between spatially proximal cells.

# %%
import pandas as pd
importlib.reload(deepcolor)
lt_df = pd.read_csv('data/tutorial/ligand_target_df.csv', index_col=0)
fig, coexp_cc_df = deepcolor.calculate_proximal_cell_communications(sc_adata, 'level1_celltype', lt_df, ["PDC", "CD1C", "ASDC", "MDSC", "Mac", "MDSC"], celltype_sample_num=500, ntop_genes=4000, each_display_num=3, role="sender", edge_thresh=1)
fig

# %% [markdown]
# # Colocalization based clustering
# Clustering cell pairs with high coloalization scores, we can obtain cell population pairs characterized by its colocalization.

# %%
dual_adata = deepcolor.analyze_pair_cluster(sc_adata, sp_adata, ["Epithelial"], ["Fibroblast"], "level1_celltype", max_pair_num=30000)
sc_adata.obs['coloc_cluster'] = 'None'
sc_adata.obs.loc[dual_adata.obs.cell1_obsname, 'coloc_cluster'] = dual_adata.obs.leiden.values
sc_adata.obs.loc[dual_adata.obs.cell2_obsname, 'coloc_cluster'] = dual_adata.obs.leiden.values
pair_adata = sc_adata[sc_adata.obs['coloc_cluster'] != 'None']
sc.pl.umap(pair_adata, color=['level1_celltype', 'coloc_cluster'])

# %%
sc_adata.write_h5ad('data/tutorial/processed_scc_sc.h5ad')


