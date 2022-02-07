import torch
import numpy as np
from .exp import VaeSmExperiment
import scanpy as sc
import pandas as pd


def define_exp(
        x_fname, s_fname,
        model_params = {
            'x_dim': 100,
            'z_dim': 10,
            'enc_z_h_dim': 50, 'enc_d_h_dim': 50, 'dec_z_h_dim': 50,
            'num_enc_z_layers': 2, 'num_enc_d_layers': 2,
            'num_dec_z_layers': 2
        },
        lr=0.001, val_ratio=0.01, test_ratio=0.01, batch_ratio=0.05, num_workers=2, device='auto'):
    x = torch.tensor(np.loadtxt(x_fname))
    s = torch.tensor(np.loadtxt(s_fname))
    model_params['x_dim'] = x.size()[1]
    model_params['s_dim'] = s.size()[1]
    x_batch_size = int(x.size()[0] * batch_ratio)
    s_batch_size = int(s.size()[0] * batch_ratio)
    vaesm_exp = VaeSmExperiment(model_params, lr, x, s, test_ratio, 100, 100, num_workers, validation_ratio=val_ratio, device=device)
    return(vaesm_exp)


# defining useful function
def select_slide(adata, s, s_col='sample', s_sub=None):
    if s_sub == None:
        slide = adata[adata.obs[s_col].isin([s]), :]
    else:
        slide = adata[adata.obs[s_col].isin([s_sub]), :]
    s_keys = list(slide.uns['spatial'].keys())
    s_spatial = s_keys[0]
    slide.uns['spatial'] = {s_spatial: slide.uns['spatial'][s]}
    return slide


def conduct_umap(adata, key):
    sc.pp.neighbors(adata, use_rep=key, n_neighbors=30)
    sc.tl.umap(adata)
    return(adata)

def convert2array(mat):
    if isinstance(mat, np.ndarray):
        return(mat)
    else:
        return(mat.toarray())


def plot_mapped_sc(sc_adata, mapping, ax):
    embed = sc_adata.obsm['X_umap']
    ax.scatter(embed[:, 0], embed[:, 1], c='gray', s=5)
    ax.scatter(embed[:, 0], embed[:, 1], c=mapping, s=30 * mapping / np.max(mapping))

def calculate_roc_df(pred, target):
    stats_df = pd.DataFrame({'pred': pred, 'target': target})
    stats_df['pos_target'] = (stats_df['target'] > 0).astype(int)
    stats_df['neg_target'] = (stats_df['target'] == 0).astype(int)
    stats_df['pos_target'] = stats_df['pos_target'] / stats_df['pos_target'].sum()
    stats_df['neg_target'] = stats_df['neg_target'] / stats_df['neg_target'].sum()
    stats_df = stats_df.groupby('pred', as_index=False).sum()
    stats_df = stats_df.sort_values('pred', ascending=False)
    stats_df['tpr'] = np.cumsum(stats_df['pos_target'])
    stats_df['fpr'] = np.cumsum(stats_df['neg_target'])
    return(stats_df)


def calculate_auc(pred, target):
    stats_df = pd.DataFrame({'pred': pred, 'target': target})
    stats_df['norm_target'] = stats_df['target'].div(stats_df['target'].sum())
    stats_df['ompr'] = stats_df['pred'].rank(method='average').div(stats_df.shape[0])
    auc = (stats_df['norm_target'] * stats_df['ompr']).sum()
    return(auc)
    
def calculate_recall(pred, target, q):
    thresh = np.quantile(pred, 1 - q)
    target = target / np.sum(target)
    recall = np.sum(target[pred > thresh])
    return(recall)

def process_each_ensembl(symbol, info_df):
    val = info_df.loc[symbol]['ensembl']
    if type(val) == dict:
        return([val['gene']])
    elif type(val) == list:
        ensembls = [d['gene'] for d in val]
        return(ensembls)
    else:
        return([])
        


def cut_unmapped(adata, q):
    contrib_vec = np.sort(adata.obsm['map2sp'].sum(axis=1))
    cum_contrib_vec = np.cumsum(contrib_vec)
    cum_contrib_vec = cum_contrib_vec / np.max(cum_contrib_vec)
    map_thresh = np.max(contrib_vec[cum_contrib_vec < q])
    adata = adata[adata.obsm['map2sp'].sum(axis=1) > map_thresh]
    return(adata)
    

def make_celltype_coloc(sc_adata, celltypes, celltype_label, thresh=2.0):
    mapped_cells = cut_unmapped(sc_adata, 0.05).obs_names
    map_vec = sc_adata.obs_names.isin(mapped_cells).astype(int)
    p_mat = sc_adata.obsm['map2sp'] / np.sum(sc_adata.obsm['map2sp'], axis=1).reshape((-1, 1))
    p_mat = p_mat * map_vec.reshape((-1, 1))
    coloc_mat = p_mat @ p_mat.transpose()
    coloc_mat = coloc_mat * p_mat.shape[1]
    bcoloc_mat = (coloc_mat > thresh).astype(int)
    celltype_coloc_props = np.array(
        [[
            np.sum(coloc_mat[sc_adata.obs[celltype_label] == celltype1][:, sc_adata.obs[celltype_label] == celltype2]) /
            (np.sum(sc_adata.obs[celltype_label] == celltype1) * np.sum(sc_adata.obs[celltype_label] == celltype2)) 
            for celltype2 in celltypes]
         for celltype1 in celltypes])
    return(celltype_coloc_props)

def make_df_col_category(df, col, categories):
    df[col] = pd.Categorical(df[col], categories=categories)
    return(df)

def categolize_method(methods):
    methods_cat = pd.Categorical(methods, categories=['scoloc', 'cell2loc', 'tangram'])
    return(methods_cat)


def make_edge_df(sc_adata, large_celltype_label, sub_sample=True, tot_size=5000, exclude_reverse=True, edge_thresh=1):
    if sub_sample:
        tot_size = 5000
        sub_sc_adata = sc_adata[np.random.choice(sc_adata.obs_names, tot_size, replace=False)]
    else:
        sub_sc_adata = sc_adata
    p_mat = sub_sc_adata.obsm['map2sp'] / np.sum(sub_sc_adata.obsm['map2sp'], axis=1).reshape((-1, 1))
    coloc_mat = p_mat @ p_mat.transpose()
    coloc_mat = np.log2(coloc_mat) + np.log2(p_mat.shape[1])
    thresh = edge_thresh
    ## thresh = np.quantile(coloc_mat, 0.8)
    high_coloc_index = np.argwhere(coloc_mat > thresh)
    if exclude_reverse:
        high_coloc_index = high_coloc_index[high_coloc_index[:, 0] < high_coloc_index[:, 1]]
    ocell1_types = sub_sc_adata.obs[large_celltype_label].iloc[high_coloc_index[:, 0]].values
    ocell2_types = sub_sc_adata.obs[large_celltype_label].iloc[high_coloc_index[:, 1]].values
    high_coloc_index = high_coloc_index[ocell1_types != ocell2_types]
    cell1_types = ocell1_types[ocell1_types != ocell2_types]
    cell2_types = ocell2_types[ocell1_types != ocell2_types]
    edge_idx = np.arange(cell1_types.shape[0])
    orig_edge_df = pd.DataFrame({'edge': edge_idx, 'cell1': sub_sc_adata.obs_names[high_coloc_index[:, 0]], 'cell2': sub_sc_adata.obs_names[high_coloc_index[:, 1]], 'cell1_type': cell1_types, 'cell2_type': cell2_types}, index=edge_idx)
    return(orig_edge_df)


def calc_signature_score(adata, orig_genes,label,  min_count=100):
    genes = orig_genes[np.isin(orig_genes, adata.var_names)]
    tot_vec = np.array(adata[:, genes].layers['count'].sum(axis=0))
    genes = genes[tot_vec > min_count]
    sc.tl.score_genes(adata, genes, score_name=label)
    return(adata)


def trancate_ext_val(vec, q=0.01):
    high_val = np.quantile(vec, 1 - q)
    vec[vec > high_val] = high_val
    return(vec)

def make_count_vec(X, axis):
    count_vec = np.array(X.sum(axis=axis)).reshape(-1)
    return(count_vec)
