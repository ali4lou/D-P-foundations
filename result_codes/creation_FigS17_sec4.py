# -*- coding: utf-8 -*-
"""
File to get Fig. S17 (section 4)
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




    
    
"""
path_save_data, path_dev, path_phen, path_sim and path_pleio
are the path that you chose after download the needed files
"""

path_save_data='C:\\Users\\logslab\\Desktop\\comprobacion_dp_rule\\all_files\\'

path_dev='C:\\Users\\logslab\\Desktop\\comprobacion_dp_rule\\all_files\\'
path_phen='C:\\Users\\logslab\\Desktop\\comprobacion_dp_rule\\all_files\\'
path_sim='C:\\Users\\logslab\\Desktop\\comprobacion_dp_rule\\all_files\\'
path_pleio='C:\\Users\\logslab\\Desktop\\comprobacion_dp_rule\\all_files\\'
path_sec2='C:\\Users\\logslab\\Desktop\\comprobacion_dp_rule\\all_files\\'

#1.) Developmental data: cells x genes matrix

adata = anndata.read(path_dev+'packer2019.h5ad')
adata #(cell x genes)

adata=quality_control(adata)
adata   

N_cell=adata.obs.shape
N_cell=N_cell[0]#numero de celulas
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


#2.) We charge the data

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



cell_types_ks_test=np.loadtxt(path_save_data+'cell_type_ks_test.txt', dtype=float)
cell_types_p_value=np.loadtxt(path_save_data+'cell_type_p_value.txt', dtype=float)


#6.) CELL TYPES
#6.1.) We filter the matrix by eliminating those with less than 50 cells
ind_good_cells=np.where(n_cells_per_type>50)[0]
new_cell_type=cell_types[ind_good_cells]
new_cell_type_ks_test=cell_types_ks_test[:, ind_good_cells]
new_cell_type_p_value=cell_types_p_value[:, ind_good_cells]

del cell_types_ks_test, cell_types_p_value


#6.2.) We perform a second clustering
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
    
#6.3.) We sort the matrix
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
    
    
    
                






