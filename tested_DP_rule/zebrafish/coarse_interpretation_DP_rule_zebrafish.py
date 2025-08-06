# -*- coding: utf-8 -*-
"""
Created on Mon Mar 10 10:41:04 2025

Cell type - phenotype association
 in zebrafish with anatomical phenotype ontology

@author: logslab
"""

import scanpy as sc
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import warnings; warnings.simplefilter('ignore')
import anndata
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster, leaves_list, optimal_leaf_ordering
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.decomposition import NMF
from scipy.stats import ks_2samp
from sklearn.linear_model import LinearRegression
 

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


path_save_data='PATH_TO_SAVE_YOUR_DATA'


#4.) SECTION 3 -> association between cell types and phenotypes
def quality_control(adata):
    #we calculate QC metrics
    sc.pp.calculate_qc_metrics(adata, percent_top=None, log1p=False, inplace=True)
    #we eliminate genes that are not expressed in at least 1 cell
    sc.pp.filter_genes(adata, min_cells=1)
    #we eliminate cells that do not achieve 1000 UMIs 
    sc.pp.filter_cells(adata, min_counts=1000)
    
    return adata
    



#We read common data

#1.1.) We read genes from phen
f=open(path_save_data+'genes_associated_phen.txt', 'r')
txt = f.read()
genes_associated_phen = txt.split('\n')
del txt, f
genes_associated_phen=np.delete(genes_associated_phen, len(genes_associated_phen)-1)
genes_associated_phen=np.array(genes_associated_phen)

#1.2.) We read commmon phenotypes
f=open(path_save_data+'phen_anatomy.txt', 'r')
txt = f.read()
phen = txt.split('\n')
del txt, f
phen=np.delete(phen, len(phen)-1)
phen=np.array(phen)

#1.3.) We read phenotype matrices
phen_matrix=np.loadtxt(path_save_data+'gene_phen_association_matrix.txt')


# #1.4.) We read developmental matrices
# dev_matrix_pseudo_bulk=np.loadtxt(path_dev+'pseudo_bulk_matrix.txt')
m_types=np.loadtxt(path_save_data+'m_types.txt')

f=open(path_save_data+'cell_type_unique.txt', 'r')
txt = f.read()
cell_type_unique = txt.split('\n')
del txt, f
cell_type_unique=np.delete(cell_type_unique, len(cell_type_unique)-1)
cell_type_unique=np.array(cell_type_unique)

n_cell_per_type=np.sum(m_types, axis=1)


#1.5.) We read the genes of development
f=open(path_save_data+'genes_frac_cell_matrix.txt', 'r')
txt = f.read()
genes_pseudo_bulk = txt.split('\n')
del txt, f
genes_pseudo_bulk=np.delete(genes_pseudo_bulk, len(genes_pseudo_bulk)-1)
genes_pseudo_bulk=np.array(genes_pseudo_bulk)



#1.7.) We search the common genes and rebuild the matrices
genes_analyzed, ind_phen, ind_dev =np.intersect1d(genes_associated_phen, genes_pseudo_bulk, return_indices=True)
# matrix_dev=dev_matrix_pseudo_bulk[ind_dev, :]
submatrix_phen=phen_matrix[ind_phen, :]


# Initialize the NMF model with deterministic initialization using 'nndsvd'
n_components = 100  # The number of latent components (reduced dimensions)
model = NMF(n_components=n_components, init='nndsvd', random_state=42, max_iter=500)

# Fit the model to the data and perform the transformation
W = model.fit_transform(submatrix_phen)  # The reduced representation of the data
H = model.components_  # The latent components (patterns)


del submatrix_phen, phen_matrix, genes_associated_phen, genes_pseudo_bulk


#2.) We read the raw data
timepoints = ['10hpf', '12hpf', '14hpf', '16hpf', '19hpf', '24hpf', '2dpf', '3dpf', '5dpf', '10dpf']


all_genes=[]
all_times=[]
all_cell_types=[]
all_data=[]

for tp in timepoints:
    file_name = f"zf_atlas_{tp}_v4_release.h5ad"
    file_path = path_save_data + file_name
    print(f"Leyendo archivo: {file_path}")  # Para verificar qué archivo se está cargando
    adata = anndata.read(file_path)

    # adata=quality_control(adata)
    # adata 
    embryo=adata.obs['fish'].to_numpy()
    print(np.unique(embryo))
    
    genes = adata.var['gene_ids'].to_numpy()  # Convertir a numpy array
    all_genes.append(genes)  # Agregar a la lista
    
    data_past = adata.X.toarray()
    all_data.append(data_past)
    
    dev_stage=adata.obs['timepoint'].to_numpy()
    all_times.append(dev_stage)
    
    cell_type=adata.obs['zebrafish_anatomy_ontology_class'].to_numpy()
    all_cell_types.append(cell_type)

    
del adata   
del data_past 

all_genes = np.array(all_genes[0])
all_cell_types = np.concatenate(all_cell_types)
all_times = np.concatenate(all_times)
all_data = np.vstack(all_data)



#Preprocesing and quality control
#Each cell have at least 1000 UMIS
sum_cell=np.sum(all_data, axis=1)
sum_n_cell_per_gene=np.zeros(len(genes))
for i in range(len(genes)):
    sum_n_cell_per_gene[i]=len(np.where(all_data[:, i]>0)[0])
    
ind_selected_cell=np.where(sum_cell>=1000)[0]
ind_selected_genes=np.where(sum_n_cell_per_gene>=1)[0]

filtered_data = all_data[np.ix_(ind_selected_cell, ind_selected_genes)]
del all_data

filtered_cell_types=all_cell_types[ind_selected_cell]
filtered_times=all_times[ind_selected_cell]
filtered_genes=all_genes[ind_selected_genes]

time_unique=timepoints

N_cell=len(filtered_cell_types)

del all_genes, all_cell_types, all_times
#important variables: filtered_data, filtered_cell_types, filteres_times, filtered_genes



#3.) We create a cellxgene matrix with the matching genes with phenotypic space
#3.1.) We search the final genes analyzed and transform the matrix
genes_final, i, ind_gene=np.intersect1d(genes_analyzed, filtered_genes, return_indices=True)
cell_genes_data=filtered_data[:, ind_gene]

del filtered_data, genes, filtered_genes
#ojo!!!
#chequear que los genes están todos dentro de filtered_genes y el orden!!


#4.) We compute the total of genes expressed in each of the cells of the scRNAseq matrix
n_genes_total_per_cel_ind=np.zeros(N_cell)
for i in range(N_cell):
    n_genes_total_per_cel_ind[i]=len(np.where(cell_genes_data[i, :]>0)[0])
    
#n_genes_total_por_cel_ind=n_genes_total_por_cel_ind/np.max(n_genes_total_por_cel_ind)
n_genes_total_per_cel_ind=n_genes_total_per_cel_ind/len(genes_final)

X=n_genes_total_per_cel_ind.reshape(-1, 1)


#5.) We create the list of gene index expressed in each cell
list_index_genes_per_single_cell=[]
for i in range(N_cell):
    list_index_genes_per_single_cell.append(np.where(cell_genes_data[i, :]>0)[0])



#6.)Link phenotype-cell types and phenotypes-times

cell_type_ks_test=[]
cell_type_p_value=[]


for p in range(len(W[0, :])):
    chosen_phen=p

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
     

    #KS TEST (cell type)
    res_specific=[]
    res_bulk=[]
    ks=[]
    p_value_list=[]
    for indice, cel_type_k in enumerate(cell_type_unique):
        n_cel_k=n_cell_per_type[indice]
        res_bulk = [residuals[i] for i in range(N_cell) if filtered_cell_types[i] != cel_type_k] 
        res_specific = [residuals[i] for i in range(N_cell) if filtered_cell_types[i] == cel_type_k] 
        
        ks_statistic, p_value = ks_2samp(res_bulk, res_specific,  alternative='greater') 
        
        ks.append(ks_statistic)
        p_value_list.append(p_value)
        
        
    cell_type_ks_test.append(ks)
    cell_type_p_value.append(p_value_list)
    
    print(p)



cell_type_ks_test=np.array(cell_type_ks_test)
cell_type_p_value=np.array(cell_type_p_value)


#We save the matrices without filter them

np.savetxt(path_save_data+'cell_type_ks_test.txt', cell_type_ks_test)
np.savetxt(path_save_data+'cell_type_p_value.txt', cell_type_p_value)



# cell_type_unique, n_cells_per_type=np.unique(filtered_cell_types, return_counts=True)


cell_type_ks_test=np.loadtxt(path_save_data+'cell_type_ks_test.txt', dtype=float)
cell_type_p_value=np.loadtxt(path_save_data+'cell_type_p_value.txt', dtype=float)

cell_type_ks_test=np.array(cell_type_ks_test)
cell_type_p_value=np.array(cell_type_p_value)

#8.) CELL TYPES
#8.1.) We filter the matrix by eliminating those with less than 50 cells
ind_good_cells=np.where(n_cell_per_type>50)[0]
new_cell_type=cell_type_unique[ind_good_cells]
new_cell_type_ks_test=cell_type_ks_test[:, ind_good_cells]
new_cell_type_p_value=cell_type_p_value[:, ind_good_cells]

del cell_type_ks_test, cell_type_p_value



#8.2.b) We perform a second clustering
dist_cell_types=pairwise_distances(np.transpose(new_cell_type_ks_test), metric='euclidean')
n_clust_cell, opt_clust_cell, ordered_indices_cells, Z_cell_type=n_clust_sorted_tree(dist_cell_types, 20, 'Cell types')

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
sorted_opt_clust_cell=[]

count=0
for i in range(len(ordered_indices_cells)):
    new_ks_past[:, count]=new_cell_type_ks_test[:, int(ordered_indices_cells[i])]
    new_p_value_past[:, count]=new_cell_type_p_value[:, int(ordered_indices_cells[i]), ]
    cell_types_sorted.append(new_cell_type[int(ordered_indices_cells[i])])
    sorted_opt_clust_cell.append(opt_clust_cell[int(ordered_indices_cells[i])])
    count=count+1

del new_cell_type
new_cell_type=cell_types_sorted
del cell_types_sorted


#8.2.) Clustering
dist_phen=pairwise_distances(new_ks_past, metric='euclidean')
n_clust_phen, opt_clust, ordered_indices, Z_nmf=n_clust_sorted_tree(dist_phen, 20, 'NMF components')

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
sorted_opt_clust=[]
phen_sorted=[]

count=0
for i in range(len(ordered_indices)):
    new_ks[count, :]=new_ks_past[int(ordered_indices[i]), :]
    new_p_value[count, :]=new_p_value_past[int(ordered_indices[i]), :]
    phen_sorted.append(int(ordered_indices[i]))
    sorted_opt_clust.append(opt_clust[int(ordered_indices[i])])
    count=count+1


#8.4.) We represent those ks with a p-value<0.0001
new_ks_signif=np.zeros((len(phen_sorted), len(new_cell_type)))
new_p_value_signif=np.zeros((len(phen_sorted), len(new_cell_type)))
ks_histogram_values=[]
cell_type_pleio=np.zeros(len(new_cell_type))
nmf_comp_pleio=np.zeros(len(phen_sorted))
for i in range(len(phen_sorted)):
    for j in range(len(new_cell_type)):
        if new_p_value[i][j]<0.0001:
            new_ks_signif[i][j]=new_ks[i][j]
            ks_histogram_values.append(new_ks[i][j])
            new_p_value_signif[i][j]=new_p_value[i][j]
            cell_type_pleio[j]=cell_type_pleio[j]+new_ks[i][j]
            nmf_comp_pleio[i]=nmf_comp_pleio[i]+1
        else:
            new_ks_signif[i][j]=np.log(0)
            new_p_value_signif[i][j]=np.log(0)


p75_ks=np.percentile(np.array(ks_histogram_values), 75)

plt.figure(figsize=(4, 3),dpi=600)
plt.hist(np.array(ks_histogram_values), bins=50, color='chocolate')
plt.xlabel('KS (p-value<0.0001)', fontsize=20, fontweight='bold')
plt.ylabel('# associations', fontsize=20, fontweight='bold')
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.axvline(x=p75_ks, color='darkred', linestyle='--')
plt.text(p75_ks+0.01, 210, 'P-75', fontweight='bold')
plt.savefig(path_save_data+'ks_values_signif.png', dpi=600, bbox_inches='tight')
plt.show()

#8.4.0.) We perform a second thereshold
new_new_ks_signif=np.zeros((len(phen_sorted), len(new_cell_type)))
cell_type_pleio_high_ks=np.zeros(len(new_cell_type))
nmf_comp_pleio_high_ks=np.zeros(len(phen_sorted))
for i in range(len(phen_sorted)):
    for j in range(len(new_cell_type)):
        if new_ks[i][j]>p75_ks:
            new_new_ks_signif[i][j]=new_ks[i][j]
            cell_type_pleio_high_ks[j]=cell_type_pleio_high_ks[j]+1
            nmf_comp_pleio_high_ks[i]=nmf_comp_pleio_high_ks[i]+1
        else:
            new_new_ks_signif[i][j]=np.log(0)

#pleiotropy associated
#cell type
ind_cell_type_more_pleio=np.argsort(cell_type_pleio_high_ks)[::-1]
cell_type_pleio_high_ks=cell_type_pleio_high_ks[ind_cell_type_more_pleio]
new_cell_type=np.array(new_cell_type)
sorted_opt_clust_cell=np.array(sorted_opt_clust_cell)
new_cell_types_sorted_pleio_high_ks=new_cell_type[ind_cell_type_more_pleio]
optimal_clust_sorted_pleio_high_ks=sorted_opt_clust_cell[ind_cell_type_more_pleio]

df_cell_types_pleio_high_ks=pd.DataFrame()
df_cell_types_pleio_high_ks['Cell type']=new_cell_types_sorted_pleio_high_ks
df_cell_types_pleio_high_ks['Pleio']=cell_type_pleio_high_ks
df_cell_types_pleio_high_ks['Cluster']=optimal_clust_sorted_pleio_high_ks

df_cell_types_pleio_high_ks.to_csv(path_save_data+'df_cell_types_pleio_high_ks.csv', sep='\t')


#nmf component
ind_nmf_more_pleio=np.argsort(nmf_comp_pleio_high_ks)[::-1]
nmf_comp_pleio_high_ks=nmf_comp_pleio_high_ks[ind_nmf_more_pleio]
phen_sorted=np.array(phen_sorted)
sorted_opt_clust=np.array(sorted_opt_clust)
phen_sorted_by_pleio_high_ks=phen_sorted[ind_nmf_more_pleio]
optimal_clust_sorted_pleio_nmf_high_ks=sorted_opt_clust[ind_nmf_more_pleio]

df_nmf_pleio_high_ks=pd.DataFrame()
df_nmf_pleio_high_ks['NMF phen']=phen_sorted_by_pleio_high_ks
df_nmf_pleio_high_ks['Pleio']=nmf_comp_pleio_high_ks
df_nmf_pleio_high_ks['Cluster']=optimal_clust_sorted_pleio_nmf_high_ks

df_nmf_pleio_high_ks.to_csv(path_save_data+'df_nmf_pleio_high_ks.csv', sep='\t')



#figure
from scipy.cluster import hierarchy
hierarchy.set_link_color_palette( ['red', 'darkorange', 'green', 'blue', 'magenta'])

#FIGURE WITH DENDROGRAM AND CELL TYPES
# Set up figure with high resolution
fig = plt.figure(figsize=(35, 13), dpi=600)

# Axis for the row dendrogram on the left
ax_in = fig.add_axes([0.22, 0.3, 0.08, 0.6])  # [left, bottom, width, height]

dendro = dendrogram(Z_nmf, color_threshold=20, leaf_rotation=90, leaf_font_size=7, 
                    above_threshold_color='gray', orientation='left', ax=ax_in)
ax_in.invert_yaxis()  # To align with the heatmap's orientation
ax_in.set_ylabel('NMF phenotypes', fontsize=40)

# Remove borders (spines) for ax_in
for spine in ax_in.spines.values():
    spine.set_visible(False)

# Adjusted position for the cell type dendrogram below the heatmap
ax_low = fig.add_axes([0.3, 0.1, 0.32, 0.2])  # [left, bottom, width, height]
dendro = dendrogram(Z_cell_type, color_threshold=20, leaf_rotation=90, leaf_font_size=7, 
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
plt.savefig(path_save_data+'fig_ks_zebra.png', dpi=600, bbox_inches='tight')
plt.show()




#8.4.) In each cluster we search in matrix H the most important phenotypes
old_phen_per_clust_sorted=[]
average_H_old_phen_per_clust_sorted=[]
for i in range(len(inner_clust)):
    new_H=[]
    for j in range(len(inner_clust[i])):
        new_H.append(H[int(inner_clust[i][j]), :])
    new_H=np.array(new_H)
    average=np.mean(new_H, axis=0)
    # plt.hist(average, bins=100, log=True)
    # plt.show()
    
    
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
    
    df.to_csv(path_save_data+'important_phen_clust%d.csv' %i, sep='\t')
    
    # np.savetxt(path_save_data+'\\cell_types\\new_clusters\\important_phen_clust%d.txt' %i, np.column_stack((highest_phen, np.array(average)[sorted_indices])), fmt='%s \t %f')
    # # np.savetxt(path_save_data+'\\cell_types\\new_clusters\\h_average_phen_clust%d.txt' %i, , fmt='%f')

    print(len(np.where(average>0)[0]))




            

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
    ind_cell=np.where(new_ks_signif[i, :]>0.8)[0]
    if len(ind_cell)>0:
        comp_nnmf_with_high_ks.append(phen_sorted[i])
        cell_type_associated.append(new_cell_type[ind_cell])
        max_ks.append(np.max(new_ks_signif[i, :]))


max_ks=np.array(max_ks)
comp_nnmf_with_high_ks=np.array(comp_nnmf_with_high_ks)
ind_phen_max=np.where(max_ks==np.max(max_ks))[0]
print(comp_nnmf_with_high_ks[int(ind_phen_max)])
print(ind_phen_max)
    

#specific figures

p=16
chosen_phen=p

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
 
count=0
#KS TEST (cell type)
np.random.seed(42)

res_specific=[]
res_bulk=[]
ks=[]
p_value_list=[]

for indice, cel_type_k in enumerate(cell_type_unique):
    n_cel_k=n_cell_per_type[indice]
    res_bulk = [residuals[i] for i in range(N_cell) if filtered_cell_types[i] != cel_type_k] 
    res_specific = [residuals[i] for i in range(N_cell) if filtered_cell_types[i] == cel_type_k] 
    
    ks_statistic, p_value = ks_2samp(res_bulk, res_specific,  alternative='greater') 
    
    ks.append(ks_statistic)
    p_value_list.append(p_value)
          
    if (p_value<0.0001) & (n_cel_k>50) & (ks_statistic>0.6):      
                
        #figure
        n_total_good_genes=[]
        n_specific_good_genes=[]
        print(cel_type_k)
        
        for i in range(N_cell):
            if cel_type_k==filtered_cell_types[i]:
                n_specific_good_genes.append(n_genes_specific_per_cel_ind[i])
                n_total_good_genes.append(n_genes_total_per_cel_ind[i])
        
        
        n_total_good_genes=np.array(n_total_good_genes)
        n_specific_good_genes=np.array(n_specific_good_genes)
        x=n_genes_total_per_cel_ind
        
        s_resid = np.sqrt(np.sum(residuals**2) / (len(x) - 2))
        y_err = 2* s_resid * np.sqrt(1/len(x))
        
        
        # Muestra aleatoria: la mitad de los puntos
        all_x = np.array(n_genes_total_per_cel_ind)
        all_y = np.array(n_genes_specific_per_cel_ind)
        
        n_points = len(all_x)
        if n_points > 2:  # Solo si hay suficientes puntos
            sample_idx = np.random.choice(n_points, size=n_points // 2, replace=False)
            x_sample = all_x[sample_idx]
            y_sample = all_y[sample_idx]
        else:
            x_sample = all_x
            y_sample = all_y
        
                
        plt.figure(figsize=(5.5, 4), dpi=300)
        plt.scatter(x_sample, y_sample, s=10, color='silver', alpha=0.7, label='Rest of the cells')
        plt.scatter(n_total_good_genes, n_specific_good_genes, s=10, color='tomato', alpha=0.7, label=cel_type_k)
        plt.plot(n_genes_total_per_cel_ind, y_pred, lw=0.2)
        # plt.fill_between(n_genes_total_per_cel_ind, lower_bound, upper_bound, color='pink', alpha=0.5)
        plt.fill_between(x, y_pred - y_err, y_pred + y_err, color='deepskyblue', alpha=0.5)
        plt.title('KS=%f' %ks_statistic, fontweight='bold', fontsize=16)
        plt.suptitle('NMF phenotype component %d' %p, fontsize=10)
        plt.xlabel('Fraction of total genes', fontsize=18)
        plt.ylabel('Gene relative weights', fontsize=18)
        plt.yticks(fontsize=14)
        plt.xticks(fontsize=14)
        plt.grid([])
        plt.legend(markerscale=2)
        plt.savefig(path_save_data+'comp_16/%s.png' %cel_type_k, dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()        
        
        count=count+1
            



ind_h=np.argsort(H[16, :])[::-1]
sorted_h=H[int(16), ind_h]
old_phen_sorted=phen[ind_h]

print(old_phen_sorted)







