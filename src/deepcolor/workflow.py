import anndata 
import torch
import scanpy as sc
import pandas as pd
import numpy as np
from .exp import VaeSmExperiment
from plotnine import *
import plotly.graph_objects as go
import plotly.io as pio
from .commons import make_edge_df
from matplotlib import colors

def safe_toarray(x):
    if type(x) != np.ndarray:
        return x.toarray()
    else:
        return x

def make_inputs(sc_adata, sp_adata, layer_name):
    x = torch.tensor(safe_toarray(sc_adata.layers[layer_name]))
    s = torch.tensor(safe_toarray(sp_adata.layers[layer_name]))
    if (x - x.int()).norm() > 0:
        try:
            raise ValueError('target layer of sc_adata should be raw count')
        except ValueError as e:
           print(e) 
    if (s - s.int()).norm() > 0:
        try:
            raise ValueError('target layer of sp_adata should be raw count')
        except ValueError as e:
           print(e) 
    return x, s

def optimize_deepcolor(vaesm_exp, lr, x_batch_size, s_batch_size, first_epoch, second_epoch):
    print(f'Loss: {vaesm_exp.evaluate()}')
    print('Start first opt')
    vaesm_exp.mode_change('sc')
    vaesm_exp.initialize_optimizer(lr)
    vaesm_exp.initialize_loader(x_batch_size, s_batch_size)
    vaesm_exp.train_total(first_epoch)
    print('Done first opt')
    print(f'Loss: {vaesm_exp.evaluate()}')
    print('Start second opt')
    vaesm_exp.mode_change('sp')
    vaesm_exp.initialize_optimizer(lr)
    vaesm_exp.initialize_loader(x_batch_size, s_batch_size)
    vaesm_exp.train_total(second_epoch)
    print('Done second opt')
    print(f'Loss: {vaesm_exp.evaluate()}')
    return vaesm_exp


def conduct_umap(adata, key):
    sc.pp.neighbors(adata, use_rep=key, n_neighbors=30)
    sc.tl.umap(adata)
    return(adata)


def extract_mapping_info(vaesm_exp, sc_adata, sp_adata):
    with torch.no_grad():
        xz, qxz, xld, p, sld, theta_x, theta_s = vaesm_exp.vaesm(vaesm_exp.xedm.x.to(vaesm_exp.device))
    sc_adata.obsm['X_zl'] = qxz.loc.detach().cpu().numpy()
    sc_adata.obsm['lambda'] = xld.detach().cpu().numpy()
    p_df = pd.DataFrame(p.detach().cpu().numpy().transpose(), index=sc_adata.obs_names, columns=sp_adata.obs_names)
    sc_adata.obsm['map2sp'] = p_df.values
    sp_adata.obsm['map2sc'] = p_df.transpose().values
    sc_adata.obsm['p_mat'] = sc_adata.obsm['map2sp'] / np.sum(sc_adata.obsm['map2sp'], axis=1).reshape((-1, 1))
    return sc_adata, sp_adata


def estimate_spatial_distribution(
        sc_adata, sp_adata, param_save_path, layer_name='count', first_epoch=500, second_epoch=500, lr=0.001, val_ratio=0.01, test_ratio=0.01, device=None, num_workers=1,
        x_batch_size=1000, s_batch_size=100, 
        model_params = {
            "x_dim": 100,
            "s_dim": 100,
            "xz_dim": 10, "sz_dim": 10,
            "enc_z_h_dim": 50, "dec_z_h_dim": 50, "map_h_dim": 50,
            "num_enc_z_layers": 2, "num_dec_z_layers": 2
        }
    ):
    if device == None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # make data set 
    sp_adata.obs_names_make_unique()
    sc_adata.obs_names_make_unique()
    x, s = make_inputs(sc_adata, sp_adata, layer_name)
    model_params['x_dim'] = x.size()[1]
    model_params['s_dim'] = s.size()[1]
    vaesm_exp = VaeSmExperiment(model_params, lr, x, s, test_ratio, 100, 100, num_workers, validation_ratio=val_ratio, device=device)
    vaesm_exp = optimize_deepcolor(vaesm_exp, lr, x_batch_size, s_batch_size, first_epoch, second_epoch)
    torch.save(vaesm_exp.vaesm.state_dict(), param_save_path)
    sc_adata.uns['param_save_path'] = param_save_path
    sp_adata.uns['param_save_path'] = param_save_path
    sp_adata.uns['layer_name'] = layer_name
    sc_adata, sp_adata = extract_mapping_info(vaesm_exp, sc_adata, sp_adata)
    return sc_adata, sp_adata


def calculate_clusterwise_distribution(sc_adata, sp_adata, cluster_label):
    p_mat = sp_adata.obsm['map2sc']
    celltypes = np.sort(np.unique(sc_adata.obs[cluster_label]))
    try:
        raise ValueError('some of cluster names in `cluster_label` is overlapped with `sp_adata.obs.columns`')
    except ValueError as e:
        print(e)
    cp_map_df = pd.DataFrame({
        celltype: np.sum(p_mat[:, sc_adata.obs[cluster_label] == celltype], axis=1)
        for celltype in celltypes}, index=sp_adata.obs_names)
    sp_adata.obs = pd.concat([sp_adata.obs, cp_map_df], axis=1)
    return sp_adata

def calculate_imputed_spatial_expression(sc_adata, sp_adata):
    sc_norm_mat = sc_adata.layers['count'].toarray() / np.sum(sc_adata.layers['count'].toarray(), axis=1).reshape((-1, 1))
    sp_adata.layers['imputed_exp'] = np.matmul(
        sp_adata.obsm['map2sc'], sc_norm_mat)
    return sp_adata

def estimate_colocalization(sc_adata):
    p_mat = sc_adata.obsm['p_mat']
    coloc_mat = p_mat @ p_mat.transpose()
    coloc_mat = np.log2(coloc_mat) + np.log2(p_mat.shape[1])
    sc_adata.obsp['colocalization'] = coloc_mat
    return sc_adata

def make_coloc_mat(sc_adata):
    p_mat = sc_adata.obsm['p_mat']
    coloc_mat = p_mat @ p_mat.transpose()
    coloc_mat = np.log2(coloc_mat) + np.log2(p_mat.shape[1])
    return coloc_mat


def make_high_coloc_index(sc_adata, celltype_label):
    coloc_mat = make_coloc_mat(sc_adata) 
    thresh = 1
    high_coloc_index = np.argwhere(coloc_mat > thresh)
    high_coloc_index = high_coloc_index[high_coloc_index[:, 0] < high_coloc_index[:, 1]]
    ocell1_types = sc_adata.obs[celltype_label].iloc[high_coloc_index[:, 0]].values
    ocell2_types = sc_adata.obs[celltype_label].iloc[high_coloc_index[:, 1]].values
    high_coloc_index = high_coloc_index[ocell1_types != ocell2_types]
    return high_coloc_index

def make_cell_umap_df(cell, edge_df, sc_adata):
    cell_adata = sc_adata[edge_df[cell]]
    cell_umap_df = pd.DataFrame(cell_adata.obsm['position'] * 0.9, columns=['X', 'Y'], index=edge_df.index)
    cell_umap_df['edge'] = edge_df.index
    return(cell_umap_df)

def make_edge_vis_df(sc_adata, celltype_label, total_edge_num, edge_thresh=1):
    orig_edge_df = make_edge_df(sc_adata, celltype_label, edge_thresh=edge_thresh)
    sub_edge_df = orig_edge_df.loc[np.random.choice(orig_edge_df.index, total_edge_num)]
    tot_edge_df = pd.concat([
        make_cell_umap_df(cell, sub_edge_df, sc_adata)
        for cell in ['cell1', 'cell2']], axis=0)
    return tot_edge_df

def visualize_colocalization_network(sc_adata, sp_adata, celltype_label, spatial_cluster, celltype_sample_num=500, total_edge_num=5000,  color_dict=None, edge_thresh=1):
    # resample celltypes
    sc_adata = sc_adata[sc_adata.obs.groupby(celltype_label).sample(celltype_sample_num, replace=True).index]
    sc_adata.obs_names_make_unique()
    # determine cell thetas and positions
    thetas = 2 * np.pi * np.arange(sc_adata.shape[0]) / sc_adata.shape[0]
    x = np.cos(thetas)
    y = np.sin(thetas)
    pos_mat = np.column_stack((x, y))
    sc_adata.obsm['position'] = pos_mat
    # map max leiden
    p_mat = sc_adata.obsm['p_mat']
    sc_adata.obs['max_map'] = sp_adata[p_mat.argmax(axis=1)].obs[spatial_cluster].astype(str).values
    # sample cell pairs based on colocalization
    tot_edge_df = make_edge_vis_df(sc_adata, celltype_label, total_edge_num, edge_thresh=edge_thresh)
    # visualize
    cells_df = pd.DataFrame({
        'X': sc_adata.obsm['position'][:, 0] * np.random.uniform(0.9, 1.1, size=sc_adata.shape[0]), 
        'Y': sc_adata.obsm['position'][:, 1] * np.random.uniform(0.9, 1.1, size=sc_adata.shape[0]), 
        'celltype': sc_adata.obs[celltype_label]})
    # determine even odds groups
    groups = sc_adata.obs[celltype_label].unique()
    gidxs = np.arange(groups.shape[0])
    even_groups = groups[gidxs % 2 == 0]
    odd_groups = groups[gidxs % 2 == 1]
    even_cells_df = cells_df.query('celltype in @even_groups')
    odd_cells_df = cells_df.query('celltype in @odd_groups')
    celltype_df = cells_df.groupby('celltype', as_index=False).mean()
    add_df = pd.DataFrame({
        'X': sc_adata.obsm['position'][:, 0] * np.random.uniform(1.1, 1.3, size=sc_adata.shape[0]), 
        'Y': sc_adata.obsm['position'][:, 1] * np.random.uniform(1.1, 1.3, size=sc_adata.shape[0]),
        'celltype': sc_adata.obs['max_map'].astype(str)})
    g = ggplot(add_df, aes(x='X', y='Y', color='celltype')) + geom_point(size=0.5)  +\
         geom_point(even_cells_df, size=0.1, color='#60C2CB') + \
          geom_point(odd_cells_df, size=0.1, color='#D05C54')  + \
          geom_line(tot_edge_df, aes(group='edge'), color='black', size=0.1, alpha=0.05) + \
              geom_text(celltype_df, aes(label='celltype'), color='black')  
    if not color_dict == None:
        g = g + scale_color_manual(color_dict)
    return g


# spcify top expression
def make_top_values(mat, top_fraction = 0.1, axis=0):
    top_mat = mat > np.quantile(mat, 1 - top_fraction, axis=axis, keepdims=True)
    return(top_mat)


def make_top_act_ligands(cell_type, coexp_count_df, topn=3):
    d = coexp_count_df.loc[cell_type].max(axis=0).sort_values(ascending=False)[:topn]
    return(d.index)


def make_coexp_cc_df(ligand_adata, edge_df, role):
    sender = edge_df.cell1 if role == "sender" else edge_df.cell2
    receiver = edge_df.cell2 if role == "sender" else edge_df.cell1
    coexp_df = pd.DataFrame(
        ligand_adata[sender].X *
        ligand_adata[receiver].layers['activity'],
        columns=ligand_adata.var_names, index=edge_df.index
    )
    coexp_df['cell2_type'] = edge_df['cell2_type']
    coexp_df['cell1_type'] = edge_df['cell1_type']
    coexp_cc_df = coexp_df.groupby(['cell2_type', 'cell1_type']).sum()
    coexp_cc_df = coexp_cc_df.reset_index().melt(id_vars=['cell1_type', 'cell2_type'], var_name='ligand', value_name='coactivity')
    return coexp_cc_df

def calculate_proximal_cell_communications(sc_adata, celltype_label, lt_df, target_cellset, celltype_sample_num=500, ntop_genes=4000, each_display_num=3, role="sender", edge_thresh=1, target_color_dict=None):
    # subsample data
    sc_adata = sc_adata[sc_adata.obs.groupby(celltype_label).sample(celltype_sample_num, replace=True).index]
    sc_adata.obs_names_make_unique()
    celltypes = sc_adata.obs.loc[:, celltype_label].unique()
    # make edge_df
    edge_df = make_edge_df(sc_adata, celltype_label, sub_sample=False, exclude_reverse=False, edge_thresh=edge_thresh)
    # select edge df with cell1 as target
    edge_df = edge_df.loc[edge_df.cell1_type.isin(target_cellset)]
    edge_df = edge_df.loc[~edge_df.cell2_type.isin(target_cellset)]
    # select genes
    sc.pp.highly_variable_genes(sc_adata, n_top_genes=ntop_genes)
    sc_adata = sc_adata[:, sc_adata.var.highly_variable]
    common_genes = np.intersect1d(lt_df.index, sc_adata.var_names)
    lt_df = lt_df.loc[common_genes]
    sc_adata = sc_adata[:, common_genes]
    lt_df = lt_df.loc[:, lt_df.columns.isin(sc_adata.var_names)]
    # make normalization
    ligands = lt_df.columns
    lt_df = lt_df.div(lt_df.sum(axis=0), axis=1)
    ligand_adata = sc_adata[:, ligands]
    top_exps = make_top_values(sc_adata.X.toarray(), axis=1, top_fraction=0.01)
    ligand_adata.layers['activity'] = make_top_values(top_exps @ lt_df)
    ligand_adata.X = make_top_values(safe_toarray(ligand_adata.X))
    # make base coexp df
    coexp_cc_df = make_coexp_cc_df(ligand_adata, edge_df, role)
    sub_coexp_cc_df = coexp_cc_df.sort_values('coactivity', ascending=False).groupby('cell2_type', as_index=False).head(n=each_display_num)
    # plotting configurrations
    tot_list = list(sub_coexp_cc_df.ligand.unique()) + list(celltypes)
    ligand_pos_dict = pd.Series({
        ligand: i
        for i, ligand in enumerate(sub_coexp_cc_df.ligand.unique())
    })
    celltype_pos_dict = pd.Series({
        celltype: i + sub_coexp_cc_df.ligand.unique().shape[0]
        for i, celltype in enumerate(celltypes)
    })
    senders = sub_coexp_cc_df.cell1_type.values if role == "sender" else sub_coexp_cc_df.cell2_type.values
    receivers = sub_coexp_cc_df.cell2_type.values if role == "sender" else sub_coexp_cc_df.cell1_type.values
    sources = pd.concat([ligand_pos_dict.loc[sub_coexp_cc_df.ligand.values], celltype_pos_dict.loc[senders]])
    targets = pd.concat([celltype_pos_dict.loc[receivers], ligand_pos_dict.loc[sub_coexp_cc_df.ligand.values]])
    values = pd.concat([sub_coexp_cc_df['coactivity'], sub_coexp_cc_df['coactivity']])
    labels = pd.concat([sub_coexp_cc_df['cell1_type'], sub_coexp_cc_df['cell1_type']])
    # colors = pd.Series(target_color_dict)[labels]
    fig = go.Figure(data=[go.Sankey(node=dict(label=tot_list),
        link=dict(
            source=sources,
            target=targets,
            value=values,
            # color=colors,
            label=labels))])
    fig.update_layout(font_family="Courier New")
    return fig, coexp_cc_df


def make_dual_zl(sc_adata, high_coloc_index):
    dual_zl_add = sc_adata.obsm['X_zl'][high_coloc_index[:, 0]] + sc_adata.obsm['X_zl'][high_coloc_index[:, 1]]
    dual_zl_prod = sc_adata.obsm['X_zl'][high_coloc_index[:, 0]] * sc_adata.obsm['X_zl'][high_coloc_index[:, 1]]
    dual_zl_prod = np.sign(dual_zl_prod) * np.sqrt(np.abs(dual_zl_prod))
    dual_zl = np.concatenate([dual_zl_add, dual_zl_prod], axis=1)
    return dual_zl_add


def setup_dual_adata(dual_zl, sc_adata, high_coloc_index):
    dual_adata = anndata.AnnData(dual_zl)
    dual_adata.obsm['X_zl'] = dual_zl
    dual_adata.obs['cell1_celltype'] = sc_adata.obs["large_class"].values[high_coloc_index[:, 0]]
    dual_adata.obs['cell2_celltype'] = sc_adata.obs["large_class"].values[high_coloc_index[:, 1]]
    dual_adata.obs['cell1_obsname'] = sc_adata.obs_names[high_coloc_index[:, 0]]
    dual_adata.obs['cell2_obsname'] = sc_adata.obs_names[high_coloc_index[:, 1]]
    dual_adata.obs_names = dual_adata.obs['cell1_obsname'] + dual_adata.obs['cell2_obsname']
    cell_min = dual_adata.obs[['cell1_celltype', 'cell2_celltype']].astype(str).values.min(axis=1)
    cell_max = dual_adata.obs[['cell1_celltype', 'cell2_celltype']].astype(str).values.max(axis=1)
    dual_adata.obs['dual_celltype'] = cell_min + '/' + cell_max
    return dual_adata


def analyze_pair_cluster(sc_adata, sp_adata, cellset1, cellset2, celltype_label, max_pair_num=30000):
    contributions = np.sum(sc_adata.obsm['map2sp'], axis=1)
    sc_adata.obs['large_class'] = 'None'
    annot1 = ','.join(cellset1)
    annot2 = ','.join(cellset2)
    sc_adata.obs['large_class'][sc_adata.obs[celltype_label].isin(cellset1)] = annot1
    sc_adata.obs['large_class'][sc_adata.obs[celltype_label].isin(cellset2)] = annot2
    sc_adata = sc_adata[sc_adata.obs['large_class'].isin([annot1, annot2])]
    high_coloc_index = make_high_coloc_index(sc_adata, "large_class")
    if high_coloc_index.shape[0] > max_pair_num:
        high_coloc_index = high_coloc_index[np.random.randint(0, high_coloc_index.shape[0], size=max_pair_num)]
    dual_zl = make_dual_zl(sc_adata, high_coloc_index)
    dual_adata = setup_dual_adata(dual_zl, sc_adata, high_coloc_index)
    dual_adata = conduct_umap(dual_adata, 'X_zl')
    sc.tl.leiden(dual_adata, resolution=0.1)
    dual_adata.uns['large_class1'] = annot1
    dual_adata.uns['large_class2'] = annot2
    return dual_adata
