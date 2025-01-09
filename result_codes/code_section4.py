# -*- coding: utf-8 -*-
"""
Section 4 - Phen association with cell types or embryo times
"""

import scanpy as sc
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
import os
import warnings; warnings.simplefilter('ignore')
import anndata
from scipy.stats import pearsonr
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster, leaves_list, optimal_leaf_ordering
from sklearn.metrics.pairwise import pairwise_distances
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from scipy.stats import mstats, kstest, ttest_ind, t
from matplotlib.colors import ListedColormap
from scipy.stats import kurtosis,  skew


def n_clust_sorted_tree(dist, cut_height, label):
    
    linkage_matrix = linkage(dist, method='ward')
    
    new_link_matrix=optimal_leaf_ordering(linkage_matrix, dist)

    
    dn=dendrogram(new_link_matrix,
               color_threshold=cut_height,
               leaf_rotation=90,
               leaf_font_size=10,
               above_threshold_color='gray')
    

    # Plotting dendrogram
    plt.xlabel(label)
    plt.ylabel('Distance')
    plt.xticks([])  # Set text labels.
    plt.show()    
    
    optimal_clusters = fcluster(new_link_matrix, cut_height, criterion='distance')
    num_clusters = len(np.unique(optimal_clusters))
    ordered_indices = leaves_list(new_link_matrix)

    return num_clusters, optimal_clusters, ordered_indices, new_link_matrix


def n_clust(dist, cut_height):
        
    linkage_matrix = linkage(dist, method='ward')
    
    plt.figure(figsize=(10, 6), dpi=600)
    
    dn=dendrogram(linkage_matrix,
               color_threshold=cut_height,
               leaf_rotation=90,
               leaf_font_size=10,
               above_threshold_color='gray')
    

    # Plotting dendrogram
    plt.xlabel('NMF components')
    plt.ylabel('Distance')
    plt.xticks([])  # Set text labels.
    plt.show()    
    
    optimal_clusters = fcluster(linkage_matrix, cut_height, criterion='distance')
    num_clusters = len(np.unique(optimal_clusters))
    ordered_indices = leaves_list(linkage_matrix)

    return num_clusters, optimal_clusters, ordered_indices


def quality_control(adata):
    #we calculate QC metrics
    sc.pp.calculate_qc_metrics(adata, percent_top=None, log1p=False, inplace=True)
    #we eliminate genes that are not expressed in at least 3 cells
    sc.pp.filter_genes(adata, min_cells=3)
    #we eliminate cells that do not achieve 1000 UMIs 
    sc.pp.filter_cells(adata, min_counts=1000)
    
    return adata
    
"""
path_save_data, path_dev, path_phen, path_sim and path_pleio
are the path that you chose after download the needed files
"""

path_save_data='YOUR_PATH_TO_SAVE_DATA'

path_dev='PATH_WHERE_IS_DOWNLOADED_THE_DATA'
path_phen='PATH_WHERE_IS_DOWNLOADED_THE_DATA'
path_sim='PATH_WHERE_IS_DOWNLOADED_THE_DATA'
path_pleio='PATH_WHERE_IS_DOWNLOADED_THE_DATA'
path_sec2='PATH_WHERE_IS_DOWNLOADED_THE_DATA'

#1.) Developmental data: cells x genes matrix

adata = anndata.read(path_dev+'packer2019.h5ad')
adata #(cell x genes)

adata=quality_control(adata)
adata  

N_cell=adata.obs.shape
N_cell=N_cell[0]
genes_id_sc_data=adata.var['gene_id'].to_numpy()
N_genes=len(genes_id_sc_data) 
print("Number of cells:", N_cell)
print('Number of genes:', N_genes)


data_past=adata.X.toarray()
time_all_cells= adata.obs['embryo_time'].to_numpy() 
cell_type=adata.obs['cell_type'].to_numpy() 
cell_subtype=adata.obs['cell_subtype'].to_numpy()

del adata

union_cell_type_all_cells=[]
for i in range(len(cell_type)):
    inner_list=cell_type[i]+ ' ' + cell_subtype[i] 
    union_cell_type_all_cells.append(inner_list)
    
del inner_list
union_cell_type_all_cells=np.array(union_cell_type_all_cells)

del cell_type, cell_subtype

cell_type_without_slash=[]
for i in range(len(union_cell_type_all_cells)):
    cell_type_without_slash.append(union_cell_type_all_cells[i].replace(' ', ' - '))
    
cell_type_all_cells=[]
for i in range(len(cell_type_without_slash)):
    cell_type_all_cells.append(cell_type_without_slash[i].replace('_', ' '))

del cell_type_without_slash, union_cell_type_all_cells


#2.) We read the data
#2.1.) We read commmon genes
f=open(path_dev+'genes_id.txt', 'r')
txt = f.read()
genes = txt.split('\n')
del txt, f
genes=np.delete(genes, len(genes)-1)
genes=np.array(genes)

#2.2.) We read commmon phenotypes
f=open(path_phen+'phenotypes_description.txt', 'r')
txt = f.read()
phen = txt.split('\n')
del txt, f
phen=np.delete(phen, len(phen)-1)
phen=np.array(phen)

#2.3.) We read phenotype matrices
W=np.loadtxt(path_phen+'W.txt')
H=np.loadtxt(path_phen+'H.txt')


#2.4.) We read developmental matrices
m_types=np.loadtxt(path_dev+'m_types.txt')
n_coord_with_cells=len(np.where(m_types>0)[0])
n_cells_per_type=np.sum(m_types, axis=1)
n_cells_per_time=np.sum(m_types, axis=0)

#2.5) We read cell_types and times
f=open(path_dev+'cell_types.txt', 'r')
txt = f.read()
cell_types = txt.split('\n')
del txt, f
cell_types=np.delete(cell_types, len(cell_types)-1)
cell_types=np.array(cell_types)

f=open(path_dev+'time.txt', 'r')
txt = f.read()
time = txt.split('\n')
del txt, f
time=np.delete(time, len(time)-1)
time=np.array(time, dtype=float)


#We read the archive with the broad classification of cell types
f=open(path_dev+'final_broad_class_cell_type.txt', 'r')
txt = f.read()
big_classification_of_our_cells = txt.split('\n')
del txt, f
big_classification_of_our_cells=np.delete(big_classification_of_our_cells, len(big_classification_of_our_cells)-1)
big_classification_of_our_cells=np.array(big_classification_of_our_cells)



#2.6.) We read cell types enriched by pleio in sec 2
f=open(path_sec2+'cell_types_enriched_pleio_sec2.txt', 'r')
txt = f.read()
cell_types_enriched_pleio_sec2 = txt.split('\n')
del txt, f
cell_types_enriched_pleio_sec2=np.delete(cell_types_enriched_pleio_sec2, len(cell_types_enriched_pleio_sec2)-1)
cell_types_enriched_pleio_sec2=np.array(cell_types_enriched_pleio_sec2)


#3.) We create a cellxgene matrix with the matching genes with phenotypic space
cell_genes_data=np.zeros((int(N_cell), len(genes)))
for i in range(len(genes)):
    ind_gene=np.where(genes_id_sc_data==genes[i])[0]
    cell_genes_data[:, i]=data_past[:, int(ind_gene)]


#4.) We compute the total of genes expressed in each of the cells of the scRNAseq matrix
n_genes_total_per_cel_ind=np.zeros(N_cell)
for i in range(N_cell):
    n_genes_total_per_cel_ind[i]=len(np.where(cell_genes_data[i, :]>0)[0])
    
#n_genes_total_por_cel_ind=n_genes_total_por_cel_ind/np.max(n_genes_total_por_cel_ind)
n_genes_total_per_cel_ind=n_genes_total_per_cel_ind/len(genes)

X=n_genes_total_per_cel_ind.reshape(-1, 1)


#5.) We create the list of gene index expressed in each cell
list_index_genes_per_single_cell=[]
for i in range(N_cell):
    list_index_genes_per_single_cell.append(np.where(cell_genes_data[i, :]>0)[0])



# #6.)Link phenotype-cell types and phenotypes-times

# cell_type_ks_test=[]
# cell_type_p_value=[]

# embryo_times_ks_test=[]
# embryo_times_p_value=[]

    
 
# for p in range(len(W[0, :])):
#     chosen_phen=p

#     #In each cell we search for the expressed genes
#     #Having those index, we add the associted weights to the specific phenotype
    
#     n_genes_specific_per_cel_ind = np.zeros(N_cell)
#     for i in range(N_cell):
#         #We have the list of gene expressed in each singel cell
#         indices = list_index_genes_per_single_cell[i]
#         for x in range(len(indices)):
#             n_genes_specific_per_cel_ind[i]=n_genes_specific_per_cel_ind[i]+W[int(indices[x])][p]
     
#     sum_weights_per_phen=np.sum(W[:, p])
#     #We normalize the weights in each phenotype
#     n_genes_specific_per_cel_ind=n_genes_specific_per_cel_ind/sum_weights_per_phen
        

#     #Linear regression (using all the cells)
#     y=n_genes_specific_per_cel_ind
    
#     model = LinearRegression()
#     model.fit(X, y)
    
#     y_pred = model.predict(X)
    
#     residuals = y - y_pred
     

#     #KS TEST (cell type)
#     res_specific=[]
#     res_bulk=[]
#     ks=[]
#     p_value_list=[]
#     for indice, cel_type_k in enumerate(cell_types):
#         n_cel_k=n_cells_per_type[indice]
#         res_bulk = [residuals[i] for i in range(N_cell) if cell_type_all_cells[i] != cel_type_k] 
#         res_specific = [residuals[i] for i in range(N_cell) if cell_type_all_cells[i] == cel_type_k] 
        
#         ks_statistic, p_value = kstest(res_bulk, res_specific,  alternative='greater') 
        
#         ks.append(ks_statistic)
#         p_value_list.append(p_value)
        
        
#     cell_type_ks_test.append(ks)
#     cell_type_p_value.append(p_value_list)

#     #KS TEST (embryo time)
#     res_specific=[]
#     res_bulk=[]
#     ks=[]
#     p_value_list=[]
#     for indice, embryo_time_k in enumerate(time):
#         n_cel_k=n_cells_per_time[indice]
#         res_bulk = [residuals[i] for i in range(N_cell) if time_all_cells[i] != embryo_time_k] 
#         res_specific = [residuals[i] for i in range(N_cell) if time_all_cells[i] == embryo_time_k] 
    
        
#         ks_statistic, p_value = kstest(res_bulk, res_specific,  alternative='greater') 
        
#         # if indice==0:
#         #     print(len(res_specific))
#         #     print(ks_statistic, embryo_time_k, p_value)
#         #     plt.hist(res_bulk, bins=100, log=True)
#         #     plt.show()
#         #     plt.hist(res_specific, bins=100, log=True)
#         #     plt.show()

#         # if indice==1:
#         #     print(ks_statistic, embryo_time_k, p_value)
#         #     plt.hist(res_bulk, bins=100, log=True)
#         #     plt.show()
#         #     plt.hist(res_specific, bins=100, log=True)
#         #     plt.show()
        
#         ks.append(ks_statistic)
#         p_value_list.append(p_value)

#     embryo_times_ks_test.append(ks)
#     embryo_times_p_value.append(p_value_list)


#     print(p)


# embryo_times_ks_test=np.array(embryo_times_ks_test)
# embryo_times_p_value=np.array(embryo_times_p_value)

# cell_type_ks_test=np.array(cell_type_ks_test)
# cell_type_p_value=np.array(cell_type_p_value)


# #We save the matrices without filter them
# np.savetxt(path_save_data+'embryo_times_ks_test.txt', embryo_times_ks_test)
# np.savetxt(path_save_data+'embryo_times_p_value.txt', embryo_times_p_value)

# np.savetxt(path_save_data+'cell_type_ks_test.txt', cell_type_ks_test)
# np.savetxt(path_save_data+'cell_type_p_value.txt', cell_type_p_value)


# #6.) Specific parameters associates with the residual distribution
# std_all_res=np.zeros(len(W[0, :]))
# mean_all_res=np.zeros(len(W[0, :]))
# std_positive_res=np.zeros(len(W[0, :]))
# kurtosis_nmf_phen=np.zeros(len(W[0, :]))
# skew_nmf_phen=np.zeros(len(W[0, :]))

# for p in range(len(W[0, :])):
#     chosen_phen=p

#     #In each cell we search for the expressed genes
#     #Having those index, we add the associted weights to the specific phenotype
    
#     n_genes_specific_per_cel_ind = np.zeros(N_cell)
#     for i in range(N_cell):
#         #We have the list of gene expressed in each singel cell
#         indices = list_index_genes_per_single_cell[i]
#         for x in range(len(indices)):
#             n_genes_specific_per_cel_ind[i]=n_genes_specific_per_cel_ind[i]+W[int(indices[x])][p]
     
#     sum_weights_per_phen=np.sum(W[:, p])
#     #We normalize the weights in each phenotype
#     n_genes_specific_per_cel_ind=n_genes_specific_per_cel_ind/sum_weights_per_phen
        

#     #Linear regression (using all the cells)
#     y=n_genes_specific_per_cel_ind
    
#     model = LinearRegression()
#     model.fit(X, y)
    
#     y_pred = model.predict(X)
    
#     residuals = y - y_pred
    
#     plt.figure(figsize=(4, 3), dpi=600)
#     plt.hist(residuals, bins=50, color='tomato')
#     plt.xlabel('Residuals', fontsize=20)
#     plt.ylabel('# NMF comp', fontsize=20)
#     plt.yticks(fontsize=16)
#     plt.xticks(fontsize=10)
#     plt.title('NMF component %d' %p, fontsize=16)
#     plt.savefig(path_save_data+'\\residuals\\figures\\dist_res_nmf_phen_%d.png' %p, dpi=600,  bbox_inches='tight')
#     plt.show()
    
    
#     std_all_res[p]=np.std(residuals)
#     mean_all_res[p]=np.mean(residuals)
    
#     ind_res_higher_zero=np.where(residuals>0)[0]
#     res_higher_than_zero=residuals[ind_res_higher_zero]
#     std_positive_res[p]=np.std(res_higher_than_zero)
    
#     kurtosis_nmf_phen[p]=kurtosis(residuals, fisher=True)
#     skew_nmf_phen[p]=skew(residuals)

#     print(p)
    

# np.savetxt(path_save_data+'\\residuals\\std_all_res.txt', std_all_res, fmt='%f')
# np.savetxt(path_save_data+'\\residuals\\mean_all_res.txt', mean_all_res, fmt='%f')
# np.savetxt(path_save_data+'\\residuals\\std_positive_res.txt', std_positive_res, fmt='%f')
# np.savetxt(path_save_data+'\\residuals\\kurtosis_nmf_phen.txt', kurtosis_nmf_phen, fmt='%f')
# np.savetxt(path_save_data+'\\residuals\\skew_nmf_phen.txt', skew_nmf_phen, fmt='%f')


# color_scater_skew_std=np.array(['#0000FF'] * 100)
# for i in range(len(skew_nmf_phen)):
#     if (skew_nmf_phen[i]<1) & (skew_nmf_phen[i]>0.8) & (std_all_res[i]<0.01):
#             print(skew_nmf_phen[i], std_all_res[i], i)
#             ind_low_skew_low_std=i
#             color_scater_skew_std[i]='#FF0000'
    
        
#     if (skew_nmf_phen[i]>3):
#             print(skew_nmf_phen[i], std_all_res[i], i)
#             color_scater_skew_std[i]='#008000'
#             ind_high_skew=i
    
#     if (std_all_res[i]>0.06):
#             print(skew_nmf_phen[i], std_all_res[i], i)
#             color_scater_skew_std[i]='#FFA500'
#             ind_high_std=i
    
# ind_nmf_phen_good=ind_low_skew_low_std
# ind_h=np.argsort(H[int(ind_nmf_phen_good), :])[::-1]
# sorted_h_ind_low_skew_low_std=H[int(ind_nmf_phen_good), ind_h]
# old_phen_sorted_ind_low_skew_low_std=phen[ind_h]

# ind_nmf_phen_good=ind_high_skew
# ind_h=np.argsort(H[int(ind_nmf_phen_good), :])[::-1]
# sorted_h_ind_high_skew=H[int(ind_nmf_phen_good), ind_h]
# old_phen_sorted_ind_high_skew=phen[ind_h]

# ind_nmf_phen_good=ind_high_std
# ind_h=np.argsort(H[int(ind_nmf_phen_good), :])[::-1]
# sorted_h_ind_high_std=H[int(ind_nmf_phen_good), ind_h]
# old_phen_sorted_ind_high_std=phen[ind_h]


# df_skew_std=pd.DataFrame()
# df_skew_std['high skew - phen']=old_phen_sorted_ind_high_skew
# df_skew_std['high skew - h']=sorted_h_ind_high_skew
# df_skew_std['low skew low std - phen']=old_phen_sorted_ind_low_skew_low_std
# df_skew_std['low skew low std - h']=sorted_h_ind_low_skew_low_std
# df_skew_std['high std - phen']=old_phen_sorted_ind_high_std
# df_skew_std['high std - h']=sorted_h_ind_high_std



#We are going to study this components

# #6.1.) kurtosis vs skewness
# plt.figure(figsize=(4, 3), dpi=600)
# plt.scatter(kurtosis_nmf_phen, skew_nmf_phen, color='blue', alpha=0.5)
# plt.ylabel('Skewness', fontsize=20)
# plt.xlabel('Kurtosis', fontsize=20)
# plt.yticks(fontsize=14)
# plt.xticks(fontsize=14)
# plt.savefig(path_save_data+'\\residuals\\kurtosis_vs_skew.png', dpi=600,  bbox_inches='tight')
# plt.show()


# color_scater_skew_std=list(color_scater_skew_std)
# plt.figure(figsize=(4, 3), dpi=600)
# plt.scatter(std_all_res, skew_nmf_phen, c=color_scater_skew_std, alpha=0.5)
# plt.ylabel('Skewness', fontsize=20)
# plt.xlabel('Standard deviation', fontsize=20)
# plt.yticks(fontsize=14)
# plt.xticks(fontsize=14)
# plt.savefig(path_save_data+'\\residuals\\skew_vs_std.png', dpi=600,  bbox_inches='tight')
# plt.show()

# #6.2.) We analyze the skew
# plt.figure(figsize=(4, 3), dpi=600)
# plt.hist(skew_nmf_phen, bins=50, color='blue', alpha=0.5)
# plt.xlabel('Skewness', fontsize=20)
# plt.ylabel('# NMF comp', fontsize=20)
# plt.yticks(fontsize=16)
# plt.xticks(fontsize=10)
# plt.savefig(path_save_data+'\\residuals\\skew_distrib_res.png', dpi=600,  bbox_inches='tight')
# plt.show()


# #We sort the skew
# ind_skew_sorted_nmf=np.argsort(skew_nmf_phen)[::-1]
# skew_sorted=skew_nmf_phen[ind_skew_sorted_nmf]
# std_sorted_by_skew=std_all_res[ind_skew_sorted_nmf]

#=================================================================
#=================================================================
#7.) We are going to read the matrices


embryo_times_ks_test=np.loadtxt(path_save_data+'matrices_day_30_october\\embryo_times_ks_test.txt', dtype=float)
embryo_times_p_value=np.loadtxt(path_save_data+'matrices_day_30_october\\embryo_times_p_value.txt', dtype=float)



#8.) TIMES
#8.1.) We filter the matrix by eliminating those with less than 50 cells
ind_good_times=np.where(n_cells_per_time>50)[0]
new_time_type=time[ind_good_times]
new_embryo_times_ks_test=embryo_times_ks_test[:, ind_good_times]
new_embryo_times_p_value=embryo_times_p_value[:, ind_good_times]

del embryo_times_ks_test, embryo_times_p_value


#8.2.) Clustering
dist_phen=pairwise_distances(new_embryo_times_ks_test, metric='euclidean')
n_clust_phen, opt_clust, ordered_indices, Z=n_clust_sorted_tree(dist_phen, 15, 'NMF components')

inner_clust=[]
unique_clusters = np.unique(opt_clust)
for cluster in unique_clusters:
    cluster_points = np.where(opt_clust == cluster)[0]
    inner_clust.append(cluster_points)
    print(f"Cluster {cluster}: {len(cluster_points)} puntos, Índices: {cluster_points}")
    

#8.3.) Sort matrix
new_ks=np.zeros((len(W[0, :]), len(new_time_type)))
new_p_value=np.zeros((len(W[0, :]), len(new_time_type)))
phen_sorted=[]

count=0
for i in range(len(ordered_indices)):
    new_ks[count, :]=new_embryo_times_ks_test[int(ordered_indices[i]), :]
    new_p_value[count, :]=new_embryo_times_p_value[int(ordered_indices[i]), :]
    phen_sorted.append(int(ordered_indices[i]))
    count=count+1

#8.4.) We represent those ks with a p-value<0.0001
new_ks_signif=np.zeros((len(phen_sorted), len(new_time_type)))
new_p_value_signif=np.zeros((len(phen_sorted), len(new_time_type)))

for i in range(len(phen_sorted)):
    for j in range(len(new_time_type)):
        if new_p_value[i][j]<0.0001:
            new_ks_signif[i][j]=new_ks[i][j]
            new_p_value_signif[i][j]=new_p_value[i][j]
        else:
            new_ks_signif[i][j]=np.log(0)
            new_p_value_signif[i][j]=np.log(0)
            
            
new_time_type=new_time_type.astype(float)


# Set up figure with high resolution
fig = plt.figure(figsize=(15, 13), dpi=600)
# Axis for the row dendrogram on the left
ax_in = fig.add_axes([0.1, 0.1, 0.2, 0.6])  # [left, bottom, width, height]
dendro = dendrogram(Z, color_threshold=15, leaf_rotation=90, leaf_font_size=7, 
                    above_threshold_color='gray', orientation='left', ax=ax_in)
ax_in.invert_yaxis()  # To align with the heatmap's orientation
ax_in.set_ylabel('Phenotypic components', fontsize=40)
# Axis for the heatmap
ax_heatmap = fig.add_axes([0.3, 0.1, 0.4, 0.6])
img = ax_heatmap.imshow(new_ks_signif, cmap='viridis_r', aspect='auto')
# Add colorbar with customized label and tick size
cbar = fig.colorbar(img, ax=ax_heatmap, shrink=0.8)
cbar.ax.yaxis.set_tick_params(labelsize=30)
cbar.set_label('ks', fontsize=40, fontweight='bold')
# Heatmap labels and styling
ax_heatmap.set_xlabel('Embryo time (min)', fontsize=40)
ax_heatmap.grid(False)
ax_heatmap.set_yticks([])  # Hide y-axis ticks
ax_heatmap.set_xticks(np.arange(len(new_time_type)))  # Ensure xticks match the data length
ax_heatmap.set_xticklabels(new_time_type, fontsize=5, rotation=90)  # Set text labels
# Hide ticks and labels on the dendrogram
ax_in.set_xticks([])
ax_in.set_yticks([])
plt.savefig(path_save_data+'ks_embryo_time_dendrogram.png', dpi=600, bbox_inches='tight')
plt.show()





# plt.figure(figsize=(10, 13), dpi=600)
# plt.imshow(new_ks_signif, cmap='viridis_r', aspect='auto')
# cbar=plt.colorbar(shrink=0.8)
# cbar.ax.yaxis.set_tick_params(labelsize=30) 
# cbar.set_label('ks', fontsize=40, fontweight='bold')  # Aquí puedes ajustar el tamaño como desees
# plt.grid(False)
# plt.ylabel('Phenotypic components', fontsize=40)
# plt.xlabel('Embryo time (min)', fontsize=40)
# plt.yticks([])
# plt.xticks(np.arange(0, len(new_time_type)), new_time_type, fontsize=7, rotation=90)  # Set text labels.
# plt.savefig(path_save_data+'ks_embryo_time.png', dpi=600, bbox_inches='tight')
# plt.show()   


# plt.figure(figsize=(10, 13), dpi=600)
# plt.imshow(new_p_value_signif, cmap='viridis_r', aspect='auto')
# cbar=plt.colorbar(shrink=0.8)
# cbar.ax.yaxis.set_tick_params(labelsize=30) 
# cbar.set_label('p-value', fontsize=40, fontweight='bold')  # Aquí puedes ajustar el tamaño como desees
# plt.grid(False)
# plt.ylabel('Phenotypic components', fontsize=40)
# plt.xlabel('Embryo time (min)', fontsize=40)
# plt.yticks([])
# plt.xticks(np.arange(0, len(new_time_type)), new_time_type, fontsize=7, rotation=90)  # Set text labels.
# plt.savefig(path_save_data+'p_val_embryo_time.png', dpi=600, bbox_inches='tight')
# plt.show()  


#if this is required retunr to execute

# #8.4.1.) clsuter 3 vs clsuter 4
# #We analyze some specific components vs others 
# #those that appear in the first embryo time and belong to the first cluster
# #When distance in dendrogram = 2.5 
# #We compare the cluster 2 and 3
# new_H_clust2=[]
# for i in range(len(inner_clust[2])):
#     new_H_clust2.append(H[int(inner_clust[2][i]), :])
    
# # av_new_H_clust2=np.mean(new_H_clust2, axis=0)
# av_new_H_clust2=H[int(inner_clust[2][6]), :]

# ind_av_new_H_clust2_sorted=np.argsort(av_new_H_clust2)[::-1]
# phen_sorted_clust2=phen[ind_av_new_H_clust2_sorted]
    
# # new_H_clust3=[]
# # for i in range(len(inner_clust[3])):
# #     new_H_clust3.append(H[int(inner_clust[3][i]), :])
    
# # av_new_H_clust3=np.mean(new_H_clust3, axis=0)
# av_new_H_clust3=H[int(inner_clust[2][5]), :]

# ind_av_new_H_clust3_sorted=np.argsort(av_new_H_clust3)[::-1]
# phen_sorted_clust3=phen[ind_av_new_H_clust3_sorted]

# np.savetxt(path_save_data+'phen_sorted_no_time_null_embryo_time_comp5.txt', phen_sorted_clust3, fmt='%s')
# np.savetxt(path_save_data+'phen_sorted_yes_time_null_embryo_time_comp6.txt', phen_sorted_clust2, fmt='%s')



#8.4.) In each cluster we search in matrix H the most important phenotypes
old_phen_per_clust_sorted=[]
average_H_old_phen_per_clust_sorted=[]
for i in range(len(inner_clust)):
    new_H=[]
    for j in range(len(inner_clust[i])):
        new_H.append(H[int(inner_clust[i][j]), :])
    new_H=np.array(new_H)
    average=np.mean(new_H, axis=0)
    plt.hist(average, bins=100, log=True)
    plt.show()
    
    
    sorted_indices = np.argsort(average)[::-1]  
    sorted_old_phen = phen[sorted_indices]       
    h_sorted=average[sorted_indices]
    
    n_highest = 50
    highest_phen = sorted_old_phen[:n_highest]
    av_h_highest=h_sorted[: n_highest]
    
    old_phen_per_clust_sorted.append(highest_phen)
    # old_phen_per_clust_sorted.append(sorted_old_phen)
    # average_H_old_phen_per_clust_sorted.append(average[sorted_indices])
    
    df=pd.DataFrame()
    df['phen']=highest_phen
    df['h']=av_h_highest
    
    df.to_csv(path_save_data+'\\times\\important_phen_clust%d.csv' %i, sep='\t')
    
    # np.savetxt(path_
    print(len(np.where(average>0)[0]))


# comon01=np.intersect1d(old_phen_per_clust_sorted[0], old_phen_per_clust_sorted[1])
# comon12=np.intersect1d(old_phen_per_clust_sorted[1], old_phen_per_clust_sorted[2])
# comon02=np.intersect1d(old_phen_per_clust_sorted[0], old_phen_per_clust_sorted[2])
# common_total=np.intersect1d(comon02, comon12)

# df_50_important_phen_per_clust = {
#     'common': common_total,
#     'common clust 01': comon01,
#     'common clust 02': comon02, 
#     'common clust 12': comon12, 
#     'cluster 0': old_phen_per_clust_sorted[0], 
#     'cluster 1': old_phen_per_clust_sorted[1], 
#     'cluster 2': old_phen_per_clust_sorted[2], 
# }


# # np.savetxt(path_save_data+'embryo_time_phen_cluster0.txt', old_phen_per_clust_sorted[0], fmt='%s')
# # np.savetxt(path_save_data+'embryo_time_phen_cluster1.txt', old_phen_per_clust_sorted[1], fmt='%s')
# # np.savetxt(path_save_data+'embryo_time_phen_cluster2.txt', old_phen_per_clust_sorted[2], fmt='%s')




# #figure of the most important embryo timess

# all_old_phen_unique=np.unique(np.concatenate([old_phen_per_clust_sorted[0], old_phen_per_clust_sorted[1], old_phen_per_clust_sorted[2]]))
# all_old_phen_unique_sorted=[]
# #we sort them
# for i in range(len(phen)):
#     ind_phen=np.where(all_old_phen_unique==phen[i])[0]
#     if len(ind_phen)>0:
#         all_old_phen_unique_sorted.append(phen[i])
    
# all_old_phen_unique_sorted=np.array(all_old_phen_unique_sorted)
# del all_old_phen_unique
    
# color_all_phen=np.zeros(len(all_old_phen_unique_sorted))
# for i in range(len(all_old_phen_unique_sorted)):
#     if all_old_phen_unique_sorted[i] in np.array(old_phen_per_clust_sorted[0]):
#         color_all_phen[i]=0
#     if all_old_phen_unique_sorted[i] in np.array(old_phen_per_clust_sorted[1]):
#         color_all_phen[i]=1
#     if all_old_phen_unique_sorted[i] in np.array(old_phen_per_clust_sorted[2]):
#         color_all_phen[i]=2
#     if all_old_phen_unique_sorted[i] in comon01:
#         color_all_phen[i]=3
#     if all_old_phen_unique_sorted[i] in comon02:
#         color_all_phen[i]=4
#     if all_old_phen_unique_sorted[i] in comon12:
#         color_all_phen[i]=5
#     # if all_old_phen_unique_sorted[i] in common_total:
#     #     color_all_phen[i]=6 
        
        
# # len(np.where(color_all_phen==3)[0])
# sorted_indices = np.argsort(color_all_phen)
# color_all_phen_sorted = color_all_phen[sorted_indices]
# phen_sorted_color=all_old_phen_unique_sorted[sorted_indices] 

# #figure
# cmap = ListedColormap(['magenta', 'cyan', 'yellow', 'mediumslateblue', 'orange', 'limegreen'])
# fig, ax = plt.subplots(figsize=(3, 22))  # Increased figsize for a larger matrix display
# cax = ax.imshow(color_all_phen_sorted.reshape(-1, 1), cmap=cmap, aspect='auto')
# cbar_ax = fig.add_axes([1, 0.3, 0.08, 0.4])  # [left, bottom, width, height]
# cbar = fig.colorbar(cax, cax=cbar_ax)
# cbar.ax.set_yticklabels(['Cluster 1', 'Cluster 2', 'Cluster 3', '1 & 2', '1 & 3', '2 & 3'], fontsize=40, fontweight='bold')
# ax.set_ylabel("Phenotypes", fontsize=50)
# ax.set_yticks(np.arange(len(phen_sorted_color)))
# ax.set_yticklabels(phen_sorted_color, fontsize=25)
# plt.show() 
            


#=======================================
#==========================================

cell_types_ks_test=np.loadtxt(path_save_data+'matrices_day_30_october\\cell_type_ks_test.txt', dtype=float)
cell_types_p_value=np.loadtxt(path_save_data+'matrices_day_30_october\\cell_type_p_value.txt', dtype=float)


#8.) CELL TYPES
#8.1.) We filter the matrix by eliminating those with less than 50 cells
ind_good_cells=np.where(n_cells_per_type>50)[0]
new_cell_type=cell_types[ind_good_cells]
new_cell_type_ks_test=cell_types_ks_test[:, ind_good_cells]
new_cell_type_p_value=cell_types_p_value[:, ind_good_cells]

del cell_types_ks_test, cell_types_p_value


#8.2.b) We perform a second clustering
dist_cell_types=pairwise_distances(np.transpose(new_cell_type_ks_test), metric='euclidean')
n_clust_cell, opt_clust_cell, ordered_indices_cells, Z_cell_type=n_clust_sorted_tree(dist_cell_types, 30, 'Cell types')

inner_clust=[]
inner_clust_cell_type=[]
unique_clusters = np.unique(opt_clust_cell)
for cluster in unique_clusters:
    cluster_points = np.where(opt_clust_cell == cluster)[0]
    inner_clust.append(cluster_points)
    inner_clust_cell_type.append(new_cell_type[cluster_points])
    print(f"Cluster {cluster}: {len(cluster_points)} puntos, Índices: {cluster_points}")
    
#8.3.b) We sort the matrix
new_ks_past=np.zeros((len(W[0, :]), len(new_cell_type)))
new_p_value_past=np.zeros((len(W[0, :]), len(new_cell_type)))
cell_types_sorted=[]

count=0
for i in range(len(ordered_indices_cells)):
    new_ks_past[:, count]=new_cell_type_ks_test[:, int(ordered_indices_cells[i])]
    new_p_value_past[:, count]=new_cell_type_p_value[:, int(ordered_indices_cells[i]), ]
    cell_types_sorted.append(new_cell_type[int(ordered_indices_cells[i])])
    count=count+1

del new_cell_type
new_cell_type=cell_types_sorted
del cell_types_sorted


#8.2.) Clustering
dist_phen=pairwise_distances(new_ks_past, metric='euclidean')
n_clust_phen, opt_clust, ordered_indices, Z_nmf=n_clust_sorted_tree(dist_phen, 12, 'NMF components')

inner_clust=[]
unique_clusters = np.unique(opt_clust)
for cluster in unique_clusters:
    cluster_points = np.where(opt_clust == cluster)[0]
    inner_clust.append(cluster_points)
    print(f"Cluster {cluster}: {len(cluster_points)} puntos, Índices: {cluster_points}")
    

            
# # 8.2) We sort the phenotypes by skewness
# # Instead of performing a second clustering we sort the phen by skewness
# ordered_indices=ind_skew_sorted_nmf

#8.3.) Sort matrix
new_ks=np.zeros((len(W[0, :]), len(new_cell_type)))
new_p_value=np.zeros((len(W[0, :]), len(new_cell_type)))
phen_sorted=[]

count=0
for i in range(len(ordered_indices)):
    new_ks[count, :]=new_ks_past[int(ordered_indices[i]), :]
    new_p_value[count, :]=new_p_value_past[int(ordered_indices[i]), :]
    phen_sorted.append(int(ordered_indices[i]))
    count=count+1


#8.4.) We represent those ks with a p-value<0.0001
new_ks_signif=np.zeros((len(phen_sorted), len(new_cell_type)))
new_p_value_signif=np.zeros((len(phen_sorted), len(new_cell_type)))
cell_type_pleio=np.zeros(len(new_cell_type))
nmf_comp_pleio=np.zeros(len(phen_sorted))
for i in range(len(phen_sorted)):
    for j in range(len(new_cell_type)):
        if new_p_value[i][j]<0.0001:
            new_ks_signif[i][j]=new_ks[i][j]
            new_p_value_signif[i][j]=new_p_value[i][j]
            cell_type_pleio[j]=cell_type_pleio[j]+new_ks[i][j]
            nmf_comp_pleio[i]=nmf_comp_pleio[i]+new_ks[i][j]
        else:
            new_ks_signif[i][j]=np.log(0)
            new_p_value_signif[i][j]=np.log(0)


#8.4.1.) Pleiotropy per cell type
#histogram
p75=np.percentile(cell_type_pleio, 75)
plt.figure(figsize=(4, 3),dpi=600)
plt.hist(cell_type_pleio, bins=50, color='royalblue')
plt.xlabel('Sum KS \n Cell type pleiotropy', fontsize=14, fontweight='bold')
plt.ylabel('# cell types', fontsize=14, fontweight='bold')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.axvline(x=p75, color='indigo', linestyle='--')
plt.text(p75+0.15, 9, 'P-75', fontweight='bold')
plt.savefig(path_save_data+'cell_type_pleiotropy_ks_matrix.png', dpi=600, bbox_inches='tight')
plt.show()

#searching the sorted cell types
ind_cell_type_more_pleio=np.argsort(cell_type_pleio)[::-1]
cell_type_pleio=cell_type_pleio[ind_cell_type_more_pleio]
new_cell_type=np.array(new_cell_type)
new_cell_types_sorted_pleio=new_cell_type[ind_cell_type_more_pleio]
optimal_clust_sorted_pleio=opt_clust_cell[ind_cell_type_more_pleio]


np.percentile(cell_type_pleio, 50)
np.savetxt(path_save_data+'new_cell_types_sorted_pleio.txt', new_cell_types_sorted_pleio, fmt='%s')

df_cell_types_pleio=pd.DataFrame()
df_cell_types_pleio['Cell type']=new_cell_types_sorted_pleio
df_cell_types_pleio['Pleio']=cell_type_pleio
df_cell_types_pleio['Cluster']=optimal_clust_sorted_pleio

df_cell_types_pleio.to_csv(path_save_data+'df_cell_types_pleio.csv', sep='\t')

sim_pleio_enriched=np.zeros(len(cell_types_enriched_pleio_sec2))
for i in range(len(cell_types_enriched_pleio_sec2)):
    ind_cel=np.where(new_cell_types_sorted_pleio==cell_types_enriched_pleio_sec2[i])[0]
    sim_pleio_enriched[i]=cell_type_pleio[int(ind_cel)]
    if sim_pleio_enriched[i]<p75:
        print(new_cell_types_sorted_pleio[int(ind_cel)], int(ind_cel))



#8.4.1.) Pleiotropy per NMF component
#histogram
p75=np.percentile(nmf_comp_pleio, 75)
plt.figure(figsize=(4, 3),dpi=600)
plt.hist(nmf_comp_pleio, bins=50, color='mediumseagreen')
plt.xlabel('Sum KS \n NMF phenotype pleiotropy', fontsize=14, fontweight='bold')
plt.ylabel('# NMF phenotypes', fontsize=14, fontweight='bold')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.axvline(x=p75, color='darkgreen', linestyle='--')
plt.text(p75+0.25, 6.5, 'P-75', fontweight='bold')
plt.savefig(path_save_data+'nmf_pleiotropy_ks_matrix.png', dpi=600, bbox_inches='tight')
plt.show()

#searching the sorted cell types
ind_nmf_more_pleio=np.argsort(nmf_comp_pleio)[::-1]
nmf_comp_pleio=nmf_comp_pleio[ind_nmf_more_pleio]
phen_sorted=np.array(phen_sorted)
phen_sorted_by_pleio=phen_sorted[ind_nmf_more_pleio]
optimal_clust_sorted_pleio_nmf=opt_clust[ind_nmf_more_pleio]


df_nmf_pleio_dendro=pd.DataFrame()
df_nmf_pleio_dendro['NMF comp']=phen_sorted_by_pleio
df_nmf_pleio_dendro['Pleio']=nmf_comp_pleio
df_nmf_pleio_dendro['Cluster']=optimal_clust_sorted_pleio_nmf

df_nmf_pleio_dendro.to_csv(path_save_data+'df_nmf_pleio_dendro.csv', sep='\t')

for i in range(5):
    H_sorted_ind=np.argsort(H[int(phen_sorted_by_pleio[i]), :])[::-1]
    h_sorted_val=H[int(phen_sorted_by_pleio[i]), H_sorted_ind]
    original_phen_sorted=phen[H_sorted_ind]

    print(phen_sorted_by_pleio[i])
    print(original_phen_sorted[:3], h_sorted_val[:3])
    # print(h_sorted_val[:3])


from scipy.cluster import hierarchy
hierarchy.set_link_color_palette( ['red', 'darkorange', 'gray', 'gray', 'gray', 'blue', 'gray', 'gray', 'gray', 'magenta', 'darkviolet'])

#FIGURE WITH DENDROGRAM AND CELL TYPES
# Set up figure with high resolution
fig = plt.figure(figsize=(35, 13), dpi=600)

# Axis for the row dendrogram on the left
ax_in = fig.add_axes([0.22, 0.3, 0.08, 0.6])  # [left, bottom, width, height]

dendro = dendrogram(Z_nmf, color_threshold=12, leaf_rotation=90, leaf_font_size=7, 
                    above_threshold_color='gray', orientation='left', ax=ax_in)
ax_in.invert_yaxis()  # To align with the heatmap's orientation
ax_in.set_ylabel('NMF phenotypes', fontsize=40)

# Remove borders (spines) for ax_in
for spine in ax_in.spines.values():
    spine.set_visible(False)

# Adjusted position for the cell type dendrogram below the heatmap
ax_low = fig.add_axes([0.3, 0.1, 0.32, 0.2])  # [left, bottom, width, height]
dendro = dendrogram(Z_cell_type, color_threshold=0, leaf_rotation=90, leaf_font_size=7, 
                    above_threshold_color='gray', orientation='top', ax=ax_low)
ax_low.invert_yaxis()  # Ensure consistent orientation
ax_low.set_xlabel('Cell types', fontsize=40)

# Remove borders (spines) for ax_in
for spine in ax_low.spines.values():
    spine.set_visible(False)

# Axis for the heatmap
ax_heatmap = fig.add_axes([0.3, 0.3, 0.4, 0.6])  # Adjusted to fit above ax_low
img = ax_heatmap.imshow(new_ks_signif, cmap='viridis_r', aspect='auto')

# Add colorbar with customized label and tick size
cbar = fig.colorbar(img, ax=ax_heatmap, shrink=0.8)
cbar.ax.yaxis.set_tick_params(labelsize=30)
cbar.set_label('ks', fontsize=40, fontweight='bold')

# Heatmap labels and styling
ax_heatmap.grid(False)
ax_heatmap.set_yticks([])  # Hide y-axis ticks
ax_heatmap.set_xticks([])  # Ensure xticks match the data length

# Hide ticks and labels on the dendrogram
ax_in.set_xticks([])
ax_in.set_yticks([])
ax_low.set_xticks([])
ax_low.set_yticks([])

# Save the figure
plt.savefig(path_save_data+'fig4B.png', dpi=600, bbox_inches='tight')
plt.show()


#==========================================================================================================
#==========================================================================================================
#===========================================================================================

# #We are going to sort the cell types and clasify them


# #We read cell_types and times
# f=open(path_dev+'cell_types_classification_by_hand.txt', 'r')
# txt = f.read()
# cell_types_by_hand = txt.split('\n')
# del txt, f
# cell_types_by_hand=np.delete(cell_types_by_hand, len(cell_types_by_hand)-1)
# cell_types_by_hand=np.array(cell_types_by_hand)


# cell_types_related_by_hand = []
# big_class_by_hand = []

# for item in cell_types_by_hand:
#     parts = item.split(' = ')
    
#     cell_types_related_by_hand.append(parts[0])  
#     big_class_by_hand.append(parts[1]) 

# del cell_types_by_hand

# cell_types_related_by_hand=np.array(cell_types_related_by_hand)
# big_class_by_hand=np.array(big_class_by_hand)

# cell_non_class = np.setdiff1d(cell_types, cell_types_related_by_hand)
# np.savetxt('cell_non_class.txt', cell_non_class, fmt='%s')

# big_classification_of_our_cells=[]
# # classified_cells=[]
# # non_classified_cells=[]
# for i in range(len(cell_types)):
#     ind_cell=np.where(cell_types_related_by_hand==cell_types[i])[0]
#     if len(ind_cell)>1:
#         if big_class_by_hand[int(ind_cell[0])]==big_class_by_hand[int(ind_cell[1])]:
#             big_classification_of_our_cells.append(big_class_by_hand[int(ind_cell[1])])
#     else:
#         big_classification_of_our_cells.append(big_class_by_hand[int(ind_cell)])

    
# big_classification_of_our_cells=np.array(big_classification_of_our_cells)

# np.savetxt(path_dev+'final_broad_class_cell_type.txt', big_classification_of_our_cells, fmt='%s')

# big_class_unique=np.unique(big_classification_of_our_cells)
# big_classification_of_our_cells=np.loadtxt(path_dev+'final_broad_class_cell_type.txt')

colors_dendro=['orange', 'mediumseagreen', 'red', 'mediumpurple', 'sienna']
label_clust=np.array(['1', '2', '3', '4', '5'])


#Separate the terms in clusters:
big_cell_types_per_clust=[]
n_times_big_cell_types_per_clust=[]
frac_precursor_cells=np.zeros(len(inner_clust_cell_type))
for i in range(len(inner_clust_cell_type)):
    big_cells=[]
    for j in range(len(inner_clust_cell_type[i])):
        ind_cell=np.where(cell_types==inner_clust_cell_type[i][j])[0]
        big_cells.append(big_classification_of_our_cells[int(ind_cell)])
    unique_big_cell_type=np.unique(big_cells, return_counts=True)
    big_cell_types_per_clust.append(unique_big_cell_type)
    # n_times_big_cell_types_per_clust.append(times_big_cells)
    
    x_fig=np.linspace(1, len(unique_big_cell_type[0]), len(unique_big_cell_type[0]))
    plt.figure(figsize=(4, 3), dpi=600)
    plt.bar(x_fig, unique_big_cell_type[1]/len(inner_clust_cell_type[i]), color=colors_dendro[i])
    plt.xlabel('Cell type', fontsize=18)
    plt.ylabel('Fraction of cells', fontsize=18)
    plt.xticks(np.arange(1, len(unique_big_cell_type[0])+1), unique_big_cell_type[0], fontsize=11, rotation=90)
    plt.yticks(fontsize=16)
    plt.title('Cluster %s' %label_clust[i], fontsize=16, fontweight='bold')
    plt.savefig(path_save_data+'cell_types_dendro\\big_cell_types_cluster_%s.png' %label_clust[i], dpi=600, bbox_inches='tight')
    plt.show()
    #Paint the histogram of cells
    
    #We find the number of precursor cells
    count_prec=0
    for k in range(len(unique_big_cell_type[0])):
        if 'Precursor Cells' in unique_big_cell_type[0][k]:
            count_prec=count_prec+unique_big_cell_type[1][k]
            print(unique_big_cell_type[0][k], i)

    frac_precursor_cells[i]=count_prec/len(inner_clust_cell_type[i])*100

x_fig=np.linspace(1, len(inner_clust_cell_type), len(inner_clust_cell_type), dtype=int)
plt.figure(figsize=(4, 3), dpi=600)
plt.bar(x_fig, frac_precursor_cells, color='deepskyblue')
plt.xlabel('Cluster', fontsize=18)
plt.ylabel('% precursor cells', fontsize=17, fontweight='bold')
plt.xticks(np.arange(1, len(inner_clust_cell_type)+1), x_fig, fontsize=16)
plt.yticks(fontsize=16)
plt.savefig(path_save_data+'cell_types_dendro\\frac_precursor_cells.png', dpi=600, bbox_inches='tight')
plt.show()

colors_dendro=['orange', 'green', 'red', 'purple', 'brown']

#Create DataFrame with cluster number, color, cell_type, big cell type
df_n_clust=[]
df_color_clust=[]
df_cell_type_clust=[]
df_big_cell_type=[]
for i in range(len(inner_clust_cell_type)):
    for j in range(len(inner_clust_cell_type[i])):
        df_n_clust.append(int(i+1))
        df_color_clust.append(colors_dendro[i])
        df_cell_type_clust.append(inner_clust_cell_type[i][j])
        ind_cell=np.where(cell_types==inner_clust_cell_type[i][j])[0]
        df_big_cell_type.append(big_classification_of_our_cells[int(ind_cell)])

df_cell_types=pd.DataFrame()
df_cell_types['Cluster']=df_n_clust
df_cell_types['Color']=df_color_clust
df_cell_types['Cell type']=df_cell_type_clust
df_cell_types['Broad classification cell type']=df_big_cell_type

df_cell_types.to_csv(path_save_data+'df_cell_types_per_cluster.csv', sep='\t')

#DataFrame of broad class of cell types
df_cell_types_new_classification=pd.DataFrame()
df_cell_types_new_classification['Cell type']=cell_types
df_cell_types_new_classification['Broad classification cell type']=big_classification_of_our_cells

df_cell_types_new_classification.to_csv(path_dev+'df_cell_types_broad_classification.csv', sep='\t')
#==========================================================================================================
#==========================================================================================================
#===========================================================================================================



# #FIGURE WITH DENDROGRAM AND CELL TYPES
# # Set up figure with high resolution
# fig = plt.figure(figsize=(35, 13), dpi=600)
# # Axis for the row dendrogram on the left
# ax_in = fig.add_axes([0.1, 0.1, 0.2, 0.6])  # [left, bottom, width, height]
# dendro = dendrogram(Z_nmf, color_threshold=12, leaf_rotation=90, leaf_font_size=7, 
#                     above_threshold_color='gray', orientation='left', ax=ax_in)
# ax_in.invert_yaxis()  # To align with the heatmap's orientation
# ax_in.set_ylabel('Phenotypic components', fontsize=40)
# # Axis for the heatmap
# ax_heatmap = fig.add_axes([0.3, 0.1, 0.4, 0.6])
# img = ax_heatmap.imshow(new_ks_signif, cmap='viridis_r', aspect='auto')
# # Add colorbar with customized label and tick size
# cbar = fig.colorbar(img, ax=ax_heatmap, shrink=0.8)
# cbar.ax.yaxis.set_tick_params(labelsize=30)
# cbar.set_label('ks', fontsize=40, fontweight='bold')
# # Heatmap labels and styling
# ax_heatmap.set_xlabel('Cell type', fontsize=40, rotation=180)
# ax_heatmap.grid(False)
# ax_heatmap.set_yticks([])  # Hide y-axis ticks
# ax_heatmap.set_xticks(np.arange(len(new_cell_type)))  # Ensure xticks match the data length
# ax_heatmap.set_xticklabels(new_cell_type, fontsize=7, rotation=90)  # Set text labels
# # Hide ticks and labels on the dendrogram
# ax_in.set_xticks([])
# ax_in.set_yticks([])
# plt.savefig(path_save_data+'ks_cell_type_phen_dendrogram.png', dpi=600, bbox_inches='tight')
# plt.show()



# plt.figure(figsize=(18, 10), dpi=600)
# plt.imshow(new_ks_signif, cmap='viridis_r', aspect='auto')
# cbar=plt.colorbar(shrink=0.8)
# cbar.ax.yaxis.set_tick_params(labelsize=30) 
# cbar.set_label('ks', fontsize=40, fontweight='bold')  # Aquí puedes ajustar el tamaño como desees
# plt.grid(False)
# plt.ylabel('Phenotypic components', fontsize=40)
# plt.xlabel('Cell type', fontsize=40, rotation=180)
# # plt.yticks([])
# plt.xticks(np.arange(0, len(new_cell_type)), new_cell_type, fontsize=7, rotation=90)  # Set text labels.
# plt.savefig(path_save_data+'ks_cell_type_phen_sorted_by_skew.png', dpi=600, bbox_inches='tight')
# plt.show()   

# plt.figure(figsize=(18, 10), dpi=600)
# plt.imshow(new_p_value_signif, cmap='viridis_r', aspect='auto')
# cbar=plt.colorbar(shrink=0.8)
# cbar.ax.yaxis.set_tick_params(labelsize=30) 
# cbar.set_label('p-value', fontsize=40, fontweight='bold')  # Aquí puedes ajustar el tamaño como desees
# plt.grid(False)
# plt.ylabel('Phenotypic components', fontsize=40)
# plt.xlabel('Cell type', fontsize=40, rotation=180)
# plt.yticks([])
# plt.xticks(np.arange(0, len(new_cell_type)), new_cell_type, fontsize=7, rotation=90)  # Set text labels.
# plt.savefig(path_save_data+'p_val_cell_type.png', dpi=600, bbox_inches='tight')
# plt.show()  


#8.4.) In each cluster we search in matrix H the most important phenotypes
old_phen_per_clust_sorted=[]
average_H_old_phen_per_clust_sorted=[]
for i in range(len(inner_clust)):
    new_H=[]
    for j in range(len(inner_clust[i])):
        new_H.append(H[int(inner_clust[i][j]), :])
    new_H=np.array(new_H)
    average=np.mean(new_H, axis=0)
    plt.hist(average, bins=100, log=True)
    plt.show()
    
    
    sorted_indices = np.argsort(average)[::-1]  
    sorted_old_phen = phen[sorted_indices]       
    h_sorted=average[sorted_indices]
    
    n_highest = 50
    highest_phen = sorted_old_phen[:n_highest]
    av_h_highest=h_sorted[: n_highest]
    
    old_phen_per_clust_sorted.append(highest_phen)
    # old_phen_per_clust_sorted.append(sorted_old_phen)
    # average_H_old_phen_per_clust_sorted.append(average[sorted_indices])
    
    df=pd.DataFrame()
    df['phen']=highest_phen
    df['h']=av_h_highest
    
    df.to_csv(path_save_data+'\\cell_types\\new_clusters\\important_phen_clust%d.csv' %i, sep='\t')
    
    # np.savetxt(path_save_data+'\\cell_types\\new_clusters\\important_phen_clust%d.txt' %i, np.column_stack((highest_phen, np.array(average)[sorted_indices])), fmt='%s \t %f')
    # # np.savetxt(path_save_data+'\\cell_types\\new_clusters\\h_average_phen_clust%d.txt' %i, , fmt='%f')

    print(len(np.where(average>0)[0]))


comon01=np.intersect1d(old_phen_per_clust_sorted[0], old_phen_per_clust_sorted[1])
comon12=np.intersect1d(old_phen_per_clust_sorted[1], old_phen_per_clust_sorted[2])
comon02=np.intersect1d(old_phen_per_clust_sorted[0], old_phen_per_clust_sorted[2])
common_total=np.intersect1d(comon02, comon12)

df_50_important_phen_per_clust_cell_type = {
    'common': common_total,
    'common clust 01': comon01,
    'common clust 02': comon02, 
    'common clust 12': comon12, 
    'cluster 0': old_phen_per_clust_sorted[0], 
    'cluster 1': old_phen_per_clust_sorted[1], 
    'cluster 2': old_phen_per_clust_sorted[2], 
}


np.savetxt(path_save_data+'cell_type_phen_cluster0.txt', old_phen_per_clust_sorted[0], fmt='%s')
np.savetxt(path_save_data+'cell_type_phen_cluster1.txt', old_phen_per_clust_sorted[1], fmt='%s')
np.savetxt(path_save_data+'cell_type_phen_cluster2.txt', old_phen_per_clust_sorted[2], fmt='%s')




#figure of the most important cell types

all_old_phen_unique=np.unique(np.concatenate([old_phen_per_clust_sorted[0], old_phen_per_clust_sorted[1], old_phen_per_clust_sorted[2]]))
all_old_phen_unique_sorted=[]
#we sort them
for i in range(len(phen)):
    ind_phen=np.where(all_old_phen_unique==phen[i])[0]
    if len(ind_phen)>0:
        all_old_phen_unique_sorted.append(phen[i])
    
all_old_phen_unique_sorted=np.array(all_old_phen_unique_sorted)
del all_old_phen_unique
    
color_all_phen=np.zeros(len(all_old_phen_unique_sorted))
for i in range(len(all_old_phen_unique_sorted)):
    if all_old_phen_unique_sorted[i] in np.array(old_phen_per_clust_sorted[0]):
        color_all_phen[i]=0
    if all_old_phen_unique_sorted[i] in np.array(old_phen_per_clust_sorted[1]):
        color_all_phen[i]=1
    if all_old_phen_unique_sorted[i] in np.array(old_phen_per_clust_sorted[2]):
        color_all_phen[i]=2
    if all_old_phen_unique_sorted[i] in comon02:
        color_all_phen[i]=3
    # if all_old_phen_unique_sorted[i] in comon02:
    #     color_all_phen[i]=4
    # if all_old_phen_unique_sorted[i] in comon12:
    #     color_all_phen[i]=5
    # if all_old_phen_unique_sorted[i] in common_total:
    #     color_all_phen[i]=6 
        
        
# len(np.where(color_all_phen==3)[0])
sorted_indices = np.argsort(color_all_phen)
color_all_phen_sorted = color_all_phen[sorted_indices]
phen_sorted_color=all_old_phen_unique_sorted[sorted_indices] 

#figure
cmap = ListedColormap(['magenta', 'cyan', 'yellow', 'orange'])
fig, ax = plt.subplots(figsize=(3, 22))  # Increased figsize for a larger matrix display
cax = ax.imshow(color_all_phen_sorted.reshape(-1, 1), cmap=cmap, aspect='auto')
cbar_ax = fig.add_axes([1, 0.3, 0.08, 0.4])  # [left, bottom, width, height]
cbar = fig.colorbar(cax, cax=cbar_ax)
cbar.ax.set_yticklabels(['Cluster 1', 'Cluster 2', 'Cluster 3', '1 & 3'], fontsize=40, fontweight='bold')
ax.set_ylabel("Phenotypes", fontsize=50)
ax.set_yticks(np.arange(len(phen_sorted_color)))
ax.set_yticklabels(phen_sorted_color, fontsize=25)
plt.show() 
            



#9.) CELL TYPES SPECIFIC ANALYSIS

del new_cell_type_ks_test, new_ks_past

new_cell_type=np.array(new_cell_type, dtype=str)

#We sort the matrix -> ks value (high ks values)
#max_ks (per nmf phen and cell type)
#average ks per nmf phen and signif ks per phen

comp_nnmf_with_high_ks=[]
cell_type_associated=[]
max_ks=[]
for i in range(len(new_ks_signif[:, 0])):
    max_ks.append(np.max(new_ks_signif[i, :]))
    ind_cell=np.where(new_ks_signif[i, :]>0.8)[0]
    if len(ind_cell)>0:
        comp_nnmf_with_high_ks.append(i)
        cell_type_associated.append(new_cell_type[ind_cell])
        
new_cell_type=np.array(new_cell_type, dtype=str)

average_ks=np.zeros(len(new_ks_signif[:, 0]))
ks_signif_cell_types_per_phen=np.zeros(len(new_ks_signif[:, 0]))
for i in range(len(new_ks_signif[:, 0])):
    ind_not_null=np.where(new_ks_signif[i, :]>0)[0]
    ks_signif_cell_types_per_phen[i]=len(ind_not_null)
    # print(np.sum(new_ks_signif[i, ind_not_null]))
    average_ks[i]=np.sum(new_ks_signif[i, ind_not_null])/len(ind_not_null)
    if (average_ks[i]<0.2) & (ks_signif_cell_types_per_phen[i]<15):
        print(i, phen_sorted[i])
        print(max_ks[i])
        # ind_cell=np.where(new_ks_signif[int(i), :]>0)[0]
        # for j in range(len(ind_cell)):
        #     print(new_cell_type[int(ind_cell)])
    
ind_h=np.argsort(H[49, :])[::-1]
sorted_h=H[int(49), ind_h]
old_phen_sorted=phen[ind_h]


np.median(ks_signif_cell_types_per_phen)
np.mean(ks_signif_cell_types_per_phen)
pearsonr(average_ks, ks_signif_cell_types_per_phen)

plt.figure(figsize=(4, 3), dpi=600)
plt.hist(ks_signif_cell_types_per_phen, bins=50, color='chocolate')
plt.xlabel('# significant KS \n (cell types)', fontsize=20, fontweight='bold')
plt.ylabel('# NMF phen', fontsize=20, fontweight='bold')
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.savefig(path_save_data+'ks_signif_cell_types_per_phen.png',dpi=600, bbox_inches='tight')

#COLOR WITH SKEWNESS
colors=[]
for i in range(len(phen_sorted)):
    ind_phen_final=np.where(ind_skew_sorted_nmf==phen_sorted[i])[0]
    colors.append(skew_sorted[int(ind_phen_final)])

# #average vs ks
plt.figure(figsize=(4, 3), dpi=600)
scatter=plt.scatter(ks_signif_cell_types_per_phen, average_ks, c=colors, cmap='rainbow', s=50)
plt.xlabel('signif ks (cell types)', fontsize=20)
plt.ylabel('<ks>', fontsize=20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
# Add color bar with customization
cbar = plt.colorbar(scatter, label='Intensity Level')  # Add color bar
cbar.set_label('Skewness', rotation=270, labelpad=15, fontsize=15, fontweight='bold')  # Label with rotation
# Optional: set color bar tick values if needed (customize range and format)
# cbar.set_ticks()  # Example of setting specific ticks
cbar.ax.tick_params(labelsize=13)  # Adjust tick font size
plt.savefig(path_save_data+'signif_ks_(cell_types)_vs_average_ks.png',dpi=600,  bbox_inches='tight')

#COLOR WITH STD
colors=[]
for i in range(len(phen_sorted)):
    ind_phen_final=np.where(ind_skew_sorted_nmf==phen_sorted[i])[0]
    colors.append(std_sorted_by_skew[int(ind_phen_final)])

# #average vs ks
plt.figure(figsize=(4, 3), dpi=600)
scatter=plt.scatter(ks_signif_cell_types_per_phen, average_ks, c=colors, cmap='gnuplot', s=50)
plt.xlabel('signif ks (cell types)', fontsize=20)
plt.ylabel('<ks>', fontsize=20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
# Add color bar with customization
cbar = plt.colorbar(scatter, label='Intensity Level')  # Add color bar
cbar.set_label('Standard deviation', rotation=270, labelpad=15, fontsize=15, fontweight='bold')  # Label with rotation
# Optional: set color bar tick values if needed (customize range and format)
# cbar.set_ticks()  # Example of setting specific ticks
cbar.ax.tick_params(labelsize=13)  # Adjust tick font size
plt.savefig(path_save_data+'sup3_signif_ks_(cell_types)_vs_average_ks_colorbar_std.png',dpi=600,  bbox_inches='tight')


#===================================================
phen_sorted=np.array(phen_sorted)
ind_phen_data_sorted_num_ks=np.argsort(ks_signif_cell_types_per_phen)
ks_signif_cell_types_per_phen_sorted=ks_signif_cell_types_per_phen[ind_phen_data_sorted_num_ks]
phen_comp_sorted_by_num_ks=phen_sorted[ind_phen_data_sorted_num_ks]


ind_h=np.argsort(H[int(phen_comp_sorted_by_num_ks[1]), :])[::-1]
sorted_h=H[int(phen_comp_sorted_by_num_ks[1]), ind_h]
old_phen_sorted=phen[ind_h]

cel_types_ind=np.where(new_ks_signif[ind_phen_data_sorted_num_ks[1], :]>0)[0]
cel_types_menos_ks=new_cell_type[cel_types_ind]


#==================================================


# plt.figure(figsize=(4, 3), dpi=600)
# plt.hist(average_ks, bins=50, color='darkorange')
# plt.xlabel('<ks>', fontsize=20)
# plt.ylabel('# nnmf phen', fontsize=20)
# plt.xticks(fontsize=16)
# plt.yticks(fontsize=16)
# plt.savefig(path_save_data+'average_ks_per_phen.png',dpi=600,  bbox_inches='tight')



plt.figure(figsize=(4, 3), dpi=600)
plt.hist(max_ks, bins=50, color='tomato')
plt.xlabel('Maximum KS', fontsize=20, fontweight='bold')
plt.ylabel('# NMF phen', fontsize=20, fontweight='bold')
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.savefig(path_save_data+'max_ks_per_phen.png',dpi=600,  bbox_inches='tight')



# #average vs ks
plt.figure(figsize=(4, 3), dpi=600)
scatter = plt.scatter(max_ks, average_ks, c=colors, cmap='rainbow', s=50)
plt.xlabel('max ks', fontsize=20)
plt.ylabel('<ks>', fontsize=20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

# Add color bar with customization
cbar = plt.colorbar(scatter, label='Intensity Level')  # Add color bar
cbar.set_label('Skewness', rotation=270, labelpad=15, fontsize=15, fontweight='bold')  # Label with rotation

# Optional: set color bar tick values if needed (customize range and format)
# cbar.set_ticks()  # Example of setting specific ticks
cbar.ax.tick_params(labelsize=13)  # Adjust tick font size


plt.savefig(path_save_data+'max_ks_vs_average_ks.png',dpi=600,  bbox_inches='tight')


#9.1) Specific analysis of cell types (maximun ks associated)
max_ks_cell_type=[]
for i in range(len(new_ks_signif[0, :])):
    max_ks_cell_type.append(np.max(new_ks_signif[:, i]))
    
max_ks_cell_type=np.array(max_ks_cell_type)
    
plt.figure(figsize=(4, 3), dpi=600)
plt.hist(max_ks_cell_type, bins=50, color='tomato')
plt.xlabel('Maximum KS', fontsize=20, fontweight='bold')
plt.ylabel('# cell types', fontsize=20, fontweight='bold')
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.savefig(path_save_data+'max_ks_per_cell_type.png',dpi=600,  bbox_inches='tight')


ind_sorted_values_max_ks_cell_type=np.argsort(max_ks_cell_type)[::-1]
new_cell_types_sorted_ks=new_cell_type[ind_sorted_values_max_ks_cell_type]
max_ks_sorted=max_ks_cell_type[ind_sorted_values_max_ks_cell_type]

df_ks_max_cell_type_associated=pd.DataFrame()
df_ks_max_cell_type_associated['cell type']=new_cell_types_sorted_ks
df_ks_max_cell_type_associated['max ks']=max_ks_sorted

del new_cell_types_sorted_ks, max_ks_sorted, ind_sorted_values_max_ks_cell_type


#EXAMPLE!!!
##9.1.1.) We chose the specific cell type to analyse
# We find two nice examples:
    #a) intestine
cell_type_chosen=df_ks_max_cell_type_associated['cell type'][10]
ind_cell_type=np.where(new_cell_type==cell_type_chosen)[0]

ind_nmf_phen_sorted=np.where(new_ks_signif[:, int(ind_cell_type)]==np.max(new_ks_signif[:, int(ind_cell_type)]))[0]
ind_nmf_phen_good=phen_sorted[int(ind_nmf_phen_sorted)]

#We search for the old phenotypes associated with the components that have a higher value 
ind_h=np.argsort(H[int(ind_nmf_phen_good), :])[::-1]
sorted_h_intestine=H[int(ind_nmf_phen_good), ind_h]
old_phen_sorted_intestine=phen[ind_h]

    #b) ciliated amphid neuron
cell_type_chosen=df_ks_max_cell_type_associated['cell type'][0]
ind_cell_type=np.where(new_cell_type==cell_type_chosen)[0]

#We search for the old phenotypes associated with the components that have a higher value 
ind_nmf_phen_sorted=np.where(new_ks_signif[:, int(ind_cell_type)]==np.max(new_ks_signif[:, int(ind_cell_type)]))[0]
ind_nmf_phen_good=phen_sorted[int(ind_nmf_phen_sorted)]

ind_h=np.argsort(H[int(ind_nmf_phen_good), :])[::-1]
sorted_h_neuron=H[int(ind_nmf_phen_good), ind_h]
old_phen_sorted_neuron=phen[ind_h]

df_important_old_phen_max_ks=pd.DataFrame()
df_important_old_phen_max_ks['intestine phen']=old_phen_sorted_intestine
df_important_old_phen_max_ks['h intestine phen']=sorted_h_intestine
df_important_old_phen_max_ks['neuron phen']=old_phen_sorted_neuron
df_important_old_phen_max_ks['h neuron phen']=sorted_h_neuron


#9.2.) NMF phen components (with the higest average & max ks) (specific analysis) -> cell types and old phen comp associated!!
#We have 17 comp_NNMF_with_high_ks (with at least one ks >0.8 significant in a cell type)
#In cell_type_associated we have the cell_types
#For each component nnmf we are going to sort the h values and its old phen
df_important_old_phen_max_ks_comp_nmf=[]
for i in range(len(comp_nnmf_with_high_ks)):
    
    ind_nmf_phen_good=phen_sorted[int(comp_nnmf_with_high_ks[i])]
    
    #We search for the old phenotypes associated with the components that have a higher value 
    ind_h=np.argsort(H[int(ind_nmf_phen_good), :])[::-1]
    sorted_h=H[int(ind_nmf_phen_good), ind_h]
    old_phen_sorted=phen[ind_h]

    df=pd.DataFrame()
    df['phen']=old_phen_sorted
    df['h phen']=sorted_h
    
    df_important_old_phen_max_ks_comp_nmf.append(df)





#======================================================================================================================
#======================================================================================================================#
#Specific example of the method for computing the highest KS 
np.max(new_ks_signif)
for i in range(len(new_cell_type)):
    ind_comp=np.where(new_ks_signif[:, i]==np.max(new_ks_signif))[0]
    if len(ind_comp)>0:
        ind_comp_fig=ind_comp
        real_comp=phen_sorted[int(ind_comp)]
        print('nmf comp=%d' %real_comp)
    
len(np.where(new_ks_signif[int(ind_comp_fig), :]>0)[0])

ind_cel=np.where(new_ks_signif[int(ind_comp_fig), :]==np.max(new_ks_signif))[0]
new_cell_type[int(ind_cel)]

#We are going to plot the ks method with comp 49 and cell type = 'Ciliated amphid neuron - ASK'
p=49

#In each cell we search for the expressed genes
#Having those index, we add the associted weights to the specific phenotype

n_genes_specific_per_cel_ind = np.zeros(N_cell)
for i in range(N_cell):
    #We have the list of gene expressed in each singel cell
    indices = list_index_genes_per_single_cell[i]
    for x in range(len(indices)):
        n_genes_specific_per_cel_ind[i]=n_genes_specific_per_cel_ind[i]+W[int(indices[x])][p]
 
sum_weights_per_phen=np.sum(W[:, p])
#We normalize the weights in each phenotype
n_genes_specific_per_cel_ind=n_genes_specific_per_cel_ind/sum_weights_per_phen
    

#Linear regression (using all the cells)
y=n_genes_specific_per_cel_ind

model = LinearRegression()
model.fit(X, y)

y_pred = model.predict(X)

residuals = y - y_pred
 
colors=[]
indice=np.where(cell_types=='Ciliated amphid neuron - ASK')[0]
cel_type_k='Ciliated amphid neuron - ASK'
n_cel_k=n_cells_per_type[indice]
res_bulk = [residuals[i] for i in range(N_cell) if cell_type_all_cells[i] != cel_type_k] 
res_specific = [residuals[i] for i in range(N_cell) if cell_type_all_cells[i] == cel_type_k] 
    
ks_statistic, p_value = kstest(res_bulk, res_specific,  alternative='greater') 

print(ks_statistic, p_value)

  
#figure
n_total_good_genes=[]
n_specific_good_genes=[]

for i in range(N_cell):
    if cel_type_k==cell_type_all_cells[i]:
        n_specific_good_genes.append(n_genes_specific_per_cel_ind[i])
        n_total_good_genes.append(n_genes_total_per_cel_ind[i])


n_total_good_genes=np.array(n_total_good_genes)
n_specific_good_genes=np.array(n_specific_good_genes)
x=n_genes_total_per_cel_ind

s_resid = np.sqrt(np.sum(residuals**2) / (len(x) - 2))
y_err = 2* s_resid * np.sqrt(1/len(x))



plt.figure(figsize=(5.5, 4), dpi=600)
plt.scatter(n_genes_total_per_cel_ind, n_genes_specific_per_cel_ind, s=10, color='silver', alpha=0.7, label='Rest of the cells')
plt.scatter(n_total_good_genes, n_specific_good_genes, s=10, color='darkviolet', alpha=0.7, label=cel_type_k)
plt.plot(n_genes_total_per_cel_ind, y_pred, lw=0.2)
# plt.fill_between(n_genes_total_per_cel_ind, lower_bound, upper_bound, color='pink', alpha=0.5)
plt.fill_between(x, y_pred - y_err, y_pred + y_err, color='deepskyblue', alpha=0.5)
plt.title('NMF phenotype component %d' %p, fontweight='bold', fontsize=16)
plt.xlabel('Fraction of total genes', fontsize=18)
plt.ylabel('Gene relative weights', fontsize=18)
plt.yticks(fontsize=14)
plt.xticks(fontsize=14)
plt.grid([])
plt.legend(markerscale=2)
plt.savefig(path_save_data+'fig4A.png', dpi=600, bbox_inches='tight')
plt.show()



#========================================================================================================
#========================================================================================================
#10.) We find the associted lineages to each cluster of cell types 

searched_lin=['AB', 'MS', 'E', 'C', 'D', 'P4', 'not \n label']


#10.1.) We read the cell_type-lineage association
n_cell_cel_type_lin=np.loadtxt(path_dev+'n_cell_cel_type_lin.txt')

#10.2.) We see what lineages are in each subcluster
unique_lineages_per_clust=[]
for i in range(len(inner_clust_cell_type)):
    asso_lin=[]
    for j in range(len(inner_clust_cell_type[i])):
        ind_cell_type=np.where(cell_types==inner_clust_cell_type[i][j])[0]
        ind_lin=np.where(n_cell_cel_type_lin[int(ind_cell_type), :]>0)[0]
        for k in range(len(ind_lin)):
            asso_lin.append(searched_lin[int(ind_lin[k])])
    
    unique_label=np.unique(asso_lin)
    final_label="\n".join(unique_label)
    
    unique_lineages_per_clust.append(final_label)
    
    
#10.3.) We are going to plot the dendrogram
def plot_dendrogram_with_cluster_labels(Z, cluster_labels, distance_threshold):

    cluster_assignments = fcluster(Z, t=distance_threshold, criterion='distance')

    plt.figure(figsize=(12, 8))
    dendro = dendrogram(
        Z,
        color_threshold=distance_threshold,  
        no_labels=True,                      
        leaf_rotation=90,                    
        leaf_font_size=10                    
    )

    ax = plt.gca()
    xlbls = ax.get_xmajorticklabels()  # Get leaf positions

    clusters_with_labels = {}
    for leaf_index, cluster_id in enumerate(cluster_assignments):
        clusters_with_labels.setdefault(cluster_id, []).append(leaf_index)

    new_labels = []
    for cluster_id in sorted(clusters_with_labels.keys()):
        label = cluster_labels[cluster_id - 1]  # Use cluster label
        new_labels.append(label)

    ax.set_xticks([225, 610, 850, 1200, 1340])
    ax.set_xticklabels(new_labels, rotation=0, ha='center', fontsize=18)

    plt.title('Cell type dendrogram - lineage relationship', fontweight='bold', fontsize=22)
    plt.savefig(path_save_data+'cell_type_dendrogram_lineage_relationship.png', dpi=600, bbox_inches='tight')

# Llamar a la función
# Z: tu matriz de enlace
# cluster_labels: un diccionario que asigna etiquetas a los clusters
plot_dendrogram_with_cluster_labels(
    Z_cell_type,
    cluster_labels=unique_lineages_per_clust,  # Etiquetas personalizadas
    distance_threshold=30  # Umbral de truncamiento
)


#10.4.) We reorganize the matrix of lineages and cell types
#We count the mean number of cell associted with each linage in each cluster
freq_unique_lin_per_clust=[]
for i in range(len(inner_clust_cell_type)):
    lin_per_clust=[]
    for j in range(len(inner_clust_cell_type[i])):
        ind_cell_type=np.where(cell_types==inner_clust_cell_type[i][j])[0]
        ind_lin=np.where(n_cell_cel_type_lin[int(ind_cell_type), :]>0)[0]
        asso_lin=np.zeros(len(searched_lin))
        for k in range(len(ind_lin)):
            asso_lin[int(ind_lin[k])]=asso_lin[int(ind_lin[k])]+n_cell_cel_type_lin[int(ind_cell_type)][int(ind_lin[k])]
        asso_lin=asso_lin/np.sum(n_cell_cel_type_lin[int(ind_cell_type), :])
        lin_per_clust.append(asso_lin)
    freq_unique_lin_per_clust.append(np.mean(lin_per_clust, axis=0))
    
freq_unique_lin_per_clust=np.array(freq_unique_lin_per_clust)
freq_unique_lin_per_clust=freq_unique_lin_per_clust*100    

freq_unique_lin_per_clust_represent=np.where(freq_unique_lin_per_clust==0, np.log(0), freq_unique_lin_per_clust)

n_clusters=np.linspace(1, 5, 5, dtype=int)
plt.figure(figsize=(7, 5), dpi=600)
plt.imshow(freq_unique_lin_per_clust_represent, cmap='coolwarm', aspect='auto')
cbar=plt.colorbar(shrink=0.5, aspect=15)
cbar.set_label('% cells (per cluster)', size=20)  # Aquí puedes ajustar el tamaño como desees
plt.grid(False)
for i in range(freq_unique_lin_per_clust.shape[0]):  
    for j in range(freq_unique_lin_per_clust.shape[1]):  
        plt.text(j, i, "{:.1f}".format(freq_unique_lin_per_clust[i, j]), ha='center', va='center', color='black', fontsize=14)
plt.ylabel('Cluster', fontsize=20)
plt.xlabel('Lineage', fontsize=20)
plt.xticks(np.arange(len(searched_lin)), searched_lin, fontsize=16, fontweight='bold')  # Set text labels.
plt.yticks(np.arange(len(n_clusters)), n_clusters, fontsize=16)
plt.savefig(path_save_data+"frec_cells_cluster_cell_type_lineage.png", dpi=600, bbox_inches='tight')
plt.show()  
    
    
    
    
