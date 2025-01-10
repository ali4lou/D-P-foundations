# -*- coding: utf-8 -*-
"""
Developmental space -> matrix construction
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
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.metrics.pairwise import pairwise_distances


def n_clust(dist, cut_height):
        
    linkage_matrix = linkage(dist, method='ward')
    
    plt.figure(figsize=(10, 6), dpi=600)
    
    dn=dendrogram(linkage_matrix,
               color_threshold=cut_height,
               leaf_rotation=90,
               leaf_font_size=10,
               above_threshold_color='gray')
    

    # Plotting dendrogram
    plt.xlabel('Gene indexes')
    plt.ylabel('Distance')
    plt.xticks([])  # Set text labels.
    plt.show()
    
    optimal_clusters = fcluster(linkage_matrix, cut_height, criterion='distance')
    
    #Number of clusters
    num_clusters = len(np.unique(optimal_clusters))

    return num_clusters, optimal_clusters


def quality_control(adata):
    #we calculate QC metrics
    sc.pp.calculate_qc_metrics(adata, percent_top=None, log1p=False, inplace=True)
    #we eliminate genes that are not expressed in at least 3 cells
    sc.pp.filter_genes(adata, min_cells=3)
    #we eliminate cells that do not achieve 1000 UMIs 
    sc.pp.filter_cells(adata, min_counts=1000)
    
    return adata
    




path_save_data='YOUR_PATH_TO_SAVE_DATA'

path_dev='PATH_WHERE_IS_DOWNLOADED_THE_DATA'

#1.) Developmental data: cells x genes matrix
adata = anndata.read(path_dev+'packer2019.h5ad')
adata #(cell x genes)


#1. EXPLORATORY ANALYSIS TO FILTER DATASET 
datos_past=adata.X.toarray()

#1.1.) Total UMIs per cell
umis_per_cell=np.sum(datos_past, axis=1)

plt.figure(figsize=(4, 3),dpi=600)
plt.hist(umis_per_cell, bins=100, color='lightseagreen', log=True)
plt.xlabel('UMI counts', fontsize=14, fontweight='bold')
plt.ylabel('# cells', fontsize=14, fontweight='bold')
plt.xticks(fontsize=8)
plt.yticks(fontsize=12)
plt.grid(False)
plt.axvline(x=np.median(umis_per_cell), color='darkslategrey', linestyle='--', lw=1)
plt.savefig(path_save_data+'distrib_umis_per_cell.png', dpi=600, bbox_inches='tight')
plt.show()

#1.2.) Total cells expressing a specific gene 
genes_n_cell=np.zeros(len(datos_past[0, :]))
for i in range(len(datos_past[0, :])):
    genes_n_cell[i]=len(np.where(datos_past[:, i]>0)[0])

plt.figure(figsize=(4, 3),dpi=600)
plt.hist(genes_n_cell, bins=100, color='lightseagreen', log=True)
plt.xlabel('# cells', fontsize=14, fontweight='bold')
plt.ylabel('# genes', fontsize=14, fontweight='bold')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(False)
plt.axvline(x=np.median(genes_n_cell), color='darkslategrey', linestyle='--', lw=1)
plt.savefig(path_save_data+'distrib_cells_expressing_gene.png', dpi=600, bbox_inches='tight')
plt.show()

del umis_per_cell, genes_n_cell


# # 1) PREPROCESING
adata=quality_control(adata)
adata  

N_cell=adata.obs.shape
N_cell=N_cell[0]#numero de celulas
genes=adata.var['gene_name'].to_numpy()
genes_id=adata.var['gene_id'].to_numpy()
N_genes=len(genes) 
print("Number of cells:", N_cell)
print('Number of genes:', N_genes)


## 2) GLOBAL ANALYSIS AFTER PREPROCESSING
#2.1.) Data to numpy
data_past=adata.X.toarray()
time= adata.obs['embryo_time'].to_numpy() 
lineage=adata.obs['lineage'].to_numpy()
cell_type=adata.obs['cell_type'].to_numpy() 
cell_subtype=adata.obs['cell_subtype'].to_numpy()
# description=adata.var['gene_description'].to_numpy()

unique_lin=np.unique(lineage)
cell_type_unique=np.unique(cell_type)

#2.2.) We unify the labels cell type ad cell subtype to tag our cells
union_cell_type=[]
for i in range(len(cell_type)):
    inner_list=cell_type[i]+ ' ' + cell_subtype[i] 
    union_cell_type.append(inner_list)
    
del inner_list
union_cell_type=np.array(union_cell_type)
unique_union_cell_type=np.unique(union_cell_type)

#2.3.) We compute the number of cells without label
n_nan_cells=0
n_cell_union_lin=0
n_cell_union_without_lin=0
n_cell_lin_only=0

distrib_times_n_nan_cells=[]
distrib_times_n_cell_union_lin=[]
distrib_times_n_cell_union_without_lin=[]
distrib_times_lin_only=[]

lin_cell_only_lin=[]

for i in range(N_cell):
    count=0
    if (lineage[i]=='nan') & (cell_subtype[i]=='nan') & (cell_type[i]=='nan'):
        n_nan_cells=n_nan_cells+1
        distrib_times_n_nan_cells.append(time[i])
        count=count+1
        
        
    if (lineage[i]=='nan') & (cell_subtype[i]!='nan') & (cell_type[i]=='nan'):
        n_cell_union_without_lin=n_cell_union_without_lin+1
        distrib_times_n_cell_union_without_lin.append(time[i])
        count=count+1

    if (lineage[i]=='nan') & (cell_type[i]!='nan') & (cell_subtype[i]=='nan'):
        n_cell_union_without_lin=n_cell_union_without_lin+1
        distrib_times_n_cell_union_without_lin.append(time[i])
        count=count+1
    
    if (lineage[i]=='nan') & (cell_type[i]!='nan') & (cell_subtype[i]!='nan'):
        n_cell_union_without_lin=n_cell_union_without_lin+1
        distrib_times_n_cell_union_without_lin.append(time[i])
        count=count+1

        
    if (lineage[i]!='nan') & (cell_subtype[i]!='nan') & (cell_type[i]=='nan'):
        n_cell_union_lin=n_cell_union_lin+1
        distrib_times_n_cell_union_lin.append(time[i])
        count=count+1

    if (lineage[i]!='nan') & (cell_type[i]!='nan') & (cell_subtype[i]=='nan'):
        n_cell_union_lin=n_cell_union_lin+1
        distrib_times_n_cell_union_lin.append(time[i]) 
        count=count+1

    if (lineage[i]!='nan') & (cell_type[i]!='nan') & (cell_subtype[i]!='nan'):
        n_cell_union_lin=n_cell_union_lin+1
        distrib_times_n_cell_union_lin.append(time[i])
        count=count+1

        
    if (lineage[i]!='nan') & (cell_type[i]=='nan') & (cell_subtype[i]=='nan'):
        n_cell_lin_only=n_cell_lin_only+1
        distrib_times_lin_only.append(time[i])
        lin_cell_only_lin.append(lineage[i])
        count=count+1

    if count==0:
        print(i, lineage[i], cell_type[i], cell_subtype[i])
        
        
#2.3.1.) Plotting figures
colors = ['grey', 'olive', "coral", 'goldenrod']

pos=[0, 1, 2, 3]
label=['Not labelled', 'Lineage only', 'Cell type-subtype \n & Lineage', 'Cell type-subtype \n Not Lineage']
plt.figure(figsize=(5, 5), dpi=600)
sns.violinplot([distrib_times_n_nan_cells, distrib_times_lin_only, distrib_times_n_cell_union_lin, distrib_times_n_cell_union_without_lin],  palette=colors, inner='quartile', edgecolor='white', linewidth=1.2, cut=0)
plt.xticks(pos, label, fontsize=16, fontweight='bold', rotation=90)
plt.yticks(fontsize=18)
plt.ylabel('Embryo time', fontsize=22)
plt.grid(False)
plt.savefig(path_save_data+'embryo_time_labelled_cells.png', dpi=600, bbox_inches='tight')
plt.show()

plt.figure(figsize=(5, 5), dpi=600)
bars=(n_nan_cells, n_cell_lin_only, n_cell_union_lin, n_cell_union_without_lin)
plt.bar(pos, bars, color=colors)
plt.xticks(pos, label, fontsize=16, fontweight='bold', rotation=90)
plt.yticks(fontsize=18)
# # plt.xlabel('Labelled  cells', fontsize=18)
plt.ylabel("# cells", fontsize=22)
plt.grid(False)
plt.savefig(path_save_data+'n_cells_labelled.png', dpi=600, bbox_inches='tight')
plt.show()

# #2.3.2.) Analyze the specific lineages
# unique_only_lin, n_times_lin=np.unique(np.array(lin_cell_only_lin), return_counts=True)

# lin_cell_only_lin=np.array(lin_cell_only_lin)
# distrib_times_lin_only=np.array(distrib_times_lin_only)
# unique_times_associated=[]
# for i in range(len(unique_only_lin)):
#     ind_lin=np.where(lin_cell_only_lin==unique_only_lin[i])[0]
#     unique_times_associated.append(distrib_times_lin_only[ind_lin])

# unique_times_only_lin=np.unique(distrib_times_lin_only)
# n_cell_coord=np.zeros((len(unique_only_lin), len(unique_times_only_lin)))
# for i in range(len(unique_only_lin)):
#     ind_lin=np.where(lin_cell_only_lin==unique_only_lin[i])[0]
#     times=distrib_times_lin_only[ind_lin]
#     for j in range(len(times)):
#         ind_t=np.where(unique_times_only_lin==times[j])[0]
#         n_cell_coord[i][int(ind_t)]=n_cell_coord[i][int(ind_t)]+1


# #We sort the linages by their initial sampling
# new_n_cell_coord=np.zeros((len(unique_only_lin), len(unique_times_only_lin)))
# ini_t=np.zeros(len(unique_only_lin))
# for i in range(len(unique_only_lin)):
#     t=np.where(n_cell_coord[i, :]>0)[0]
#     ini_t[i]=t[0]

# ind_lin=np.argsort(ini_t)
# new_lin_unique=unique_only_lin[ind_lin]
# for i in range(len(new_lin_unique)):
#     new_n_cell_coord[i, :]=n_cell_coord[int(ind_lin[i]), :]

# # n_cell_coord=(np.where(n_cell_coord==0), np.log(0), n_cell_coord)
# #Figure of m_types
# plt.figure(figsize=(6, 12), dpi=600)
# plt.imshow(np.log(new_n_cell_coord) , cmap='copper_r', aspect='auto')
# cbar=plt.colorbar(shrink=0.5, aspect=15)
# cbar.set_label('# cells (log scale)', size=20)  # Aquí puedes ajustar el tamaño como desees
# plt.grid(False)
# plt.ylabel('Lineage', fontsize=20)
# plt.xlabel('Embryo time (min)', fontsize=20)
# plt.xticks(np.arange(len(unique_times_only_lin)), unique_times_only_lin, fontsize=5, rotation=90)  # Set text labels.
# plt.yticks(np.arange(len(new_lin_unique)), new_lin_unique, fontsize=3)
# plt.savefig(path_save_data+'n_cell_only_lin.png', dpi=600, bbox_inches='tight')  
# plt.show()      



del n_nan_cells, n_cell_union_lin, n_cell_union_without_lin, n_cell_lin_only
del distrib_times_n_nan_cells, distrib_times_n_cell_union_lin, distrib_times_n_cell_union_without_lin, distrib_times_lin_only


#2.4.) We compute the number of null genes that each cell has
zeros=np.zeros(N_cell)
for i in range(N_cell):
    zeros[i]=len(np.where(data_past[i, :]==0)[0])/N_genes

# plt.figure(figsize=(4, 3),dpi=600)
# plt.hist(zeros, bins=100, color='darkorange', log=True)
# plt.xlabel('Fraction of null genes', fontsize=14, fontweight='bold')
# plt.ylabel('# cells', fontsize=14, fontweight='bold')
# plt.xticks(fontsize=12)
# plt.yticks(fontsize=12)
# plt.grid(False)
# plt.axvline(x=np.median(zeros), color='darkred', linestyle='--', lw=1)
# plt.savefig(path_save_data+'null_genes_per_cell.png', dpi=600, bbox_inches='tight')
# plt.show()


#2.5.) Binarizeed gene expression and UMI count gene expression
pearson=[]
p_val=[]
for i in range(N_genes):
    non_null=np.where(data_past[:, i]!=0)[0]
    exp=np.zeros(N_cell)
    for j in range(len(non_null)):
        exp[int(non_null[j])]=1
    pear=pearsonr(exp, data_past[:, i])
    pearson.append(pear[0])
    p_val.append(pear[1])

# plt.figure(figsize=(4, 3),dpi=600)
# plt.hist(pearson, bins=100, color='darkturquoise', log=True)
# plt.xlabel('Pearson coefficient', fontsize=14, fontweight='bold')
# plt.ylabel('# genes', fontsize=14, fontweight='bold')
# plt.xticks(fontsize=12)
# plt.yticks(fontsize=12)
# plt.grid(False)
# plt.axvline(x=np.median(zeros), color='darkslategrey', linestyle='--', lw=1)
# plt.savefig(path_save_data+'pearson_binary_umi_dev_profile.png', dpi=600, bbox_inches='tight')
# plt.show()


#3) We eliminate not labelled cells -> 'nan' cells
total_index=np.linspace(0, N_cell-1, N_cell, dtype=int)
index_nan_cells=np.where((cell_subtype == 'nan') & (cell_type == 'nan'))[0]
time_without_nan=np.delete(time, index_nan_cells)
union_cell_type_without_nan=np.delete(union_cell_type, index_nan_cells)
lineage_without_nan=np.delete(lineage, index_nan_cells)
index_without_nan_cells=np.delete(total_index, index_nan_cells)

del adata, time, union_cell_type, lineage, cell_type, cell_subtype, total_index
time=time_without_nan
lineage=lineage_without_nan
union_cell_type=union_cell_type_without_nan
del time_without_nan, lineage_without_nan, union_cell_type_without_nan

N_cell=len(union_cell_type)
print("Number of cells:", N_cell)
print('Number of genes:', N_genes)


data=[]
for i in range(len(index_without_nan_cells)):
    data.append(data_past[int(index_without_nan_cells[i]), :])
    
data=np.array(data)
del data_past, index_without_nan_cells, index_nan_cells



#3.1.) We build a DataFrame to sort the values by time and cell type
new_index=np.linspace(0, N_cell-1, N_cell, dtype=int)
df=pd.DataFrame()
df['index']=new_index
df['time'] = time
df['lineage']=lineage
df['cell_type']=union_cell_type
df_sort_t_cell_type=df.sort_values(by=["time", "cell_type"])
a=df_sort_t_cell_type.to_numpy()

del df, df_sort_t_cell_type

#We sort the data by df[index]
sort_index=a[:, 0].astype(int)
sorted_data=[]
for i in range(N_cell):
    sorted_data.append(data[int(sort_index[i]), :])

del data
data=np.array(sorted_data)
del sorted_data, sort_index, new_index

a=np.delete(a, 0, axis=1)
#3.2.) We search unique union of cell types, lineages and times
time=a[:, 0].astype(float)
lineage=a[:, 1].astype(str)
union_cell_type=a[:, 2].astype(str)
unique_time=np.unique(time)
unique_union_cell_type=np.unique(union_cell_type)
unique_lin=np.unique(lineage)


#3.3.) We create the dev space and count the number of cells in each coordinate
m_types=np.zeros((len(unique_union_cell_type), len(unique_time)))

for i in range(N_cell): 
    tag=np.where(unique_union_cell_type==union_cell_type[i])[0]
    if len(tag)>0:
        tag=int(tag)
        tag_t=np.where(unique_time==time[i])[0]
        tag_t=int(tag_t)
        m_types[tag][tag_t]=m_types[tag][tag_t]+1
    
del tag, tag_t
m_types=np.array(m_types, dtype=int)

#We find the order of sorted cell types
cell_types_sorted=[]
remove_cell_type=unique_union_cell_type
for i in range(N_cell):
    ind_cell=np.where(remove_cell_type==union_cell_type[i])[0]
    if len(ind_cell)>0:
        cell_types_sorted.append(remove_cell_type[int(ind_cell)])
        remove_cell_type=np.delete(remove_cell_type, int(ind_cell))

#We sort the m_types_matrix
new_m_types=np.zeros((len(unique_union_cell_type), len(unique_time)))

cell_types_sorted=np.array(cell_types_sorted)
for i in range(len(cell_types_sorted)):
    ind_cell=np.where(unique_union_cell_type==cell_types_sorted[i])[0]
    new_m_types[i, :]=m_types[int(ind_cell), :]

#Cell type label
cell_type_without_slash=[]
for i in range(len(cell_types_sorted)):
    cell_type_without_slash.append(cell_types_sorted[i].replace(' ', ' - '))
    
unique_union_cell_type=[]
for i in range(len(cell_types_sorted)):
    unique_union_cell_type.append(cell_type_without_slash[i].replace('_', ' '))

del cell_type_without_slash    
unique_union_cell_type=np.array(unique_union_cell_type)
unique_union_cell_type_conserved=cell_types_sorted
del cell_types_sorted

#Time label
label_time=[]
label_time2=[]
for i in range(len(unique_time)):
    if i%4==0:
        label_time.append(int(unique_time[i]))
    else:
        label_time.append('')
    if i%2==0:
        label_time2.append(int(unique_time[i]))
    else:
        label_time2.append('')

#Figure of m_types
plt.figure(figsize=(6, 12), dpi=600)
plt.imshow(np.log(new_m_types) , cmap='copper_r', aspect='auto')
cbar=plt.colorbar(shrink=0.5, aspect=15)
cbar.set_label('# cells (log scale)', size=20)  # Aquí puedes ajustar el tamaño como desees
plt.grid(False)
plt.ylabel('Cell type', fontsize=20)
plt.xlabel('Embryo time (min)', fontsize=20)
plt.xticks(np.arange(len(unique_time)), label_time2, fontsize=5, rotation=90)  # Set text labels.
plt.yticks(np.arange(len(unique_union_cell_type)), unique_union_cell_type, fontsize=3)
plt.savefig(path_save_data+'m_types.png', dpi=600, bbox_inches='tight')  
plt.show()       

del m_types
m_types=new_m_types
del new_m_types


# #We save m_types, cell_types and times
# np.savetxt(path_save_data+'m_types.txt', m_types, fmt='%d')
# np.savetxt(path_save_data+'cell_types.txt', unique_union_cell_type, fmt='%s')
#np.savetxt(path_save_data+'cell_types_conserved.txt', unique_union_cell_type, fmt='%s')
np.savetxt(path_save_data+'time.txt', unique_time, fmt='%f')


#3.3.1.)We compute the number of cells per embryo time -> Collapse cell types
n_cell_per_time=np.zeros(len(unique_time))
for i in range(len(unique_time)):            
    n_cell_per_time[i]=len(np.where(time==unique_time[i])[0])

# fig=plt.figure(figsize=(12, 3), dpi=600)
# plt.bar(np.arange(len(n_cell_per_time)), n_cell_per_time, width=0.7, color='peru')
# plt.xlabel("Embryo time (min)", fontsize=20, rotation=180)
# plt.yscale("log")
# plt.xticks(np.arange(len(unique_time)), unique_time.astype(int), rotation=90, fontsize=5)
# plt.yticks(rotation=90)
# plt.ylabel("# cells", fontsize=20, fontweight='bold')
# plt.savefig(path_save_data+"n_cell_per_embryo_time.png", dpi=600, bbox_inches='tight')
# plt.grid(False)
# plt.show()

#3.3.2.) We compute the number of cells per cell type -> Collapse embryo times
n_cell_per_type=np.sum(m_types, axis=1)

# fig=plt.figure(figsize=(12,3), dpi=600)
# plt.bar(np.arange(len(n_cell_per_type)), n_cell_per_type, width=0.7, color='peru')
# plt.xlabel("Cell type", fontsize=20, rotation=180)
# plt.yscale("log")
# plt.yticks(rotation=90)
# plt.xticks(np.arange(len(unique_union_cell_type)), unique_union_cell_type, rotation=90, fontsize=3)
# plt.ylabel("# cells", fontsize=20, fontweight='bold')
# plt.savefig(path_save_data+"n_cell_per_type.png", dpi=600, bbox_inches='tight')
# plt.grid(False)
# plt.show()


#4) RELATIONSHIP BETWEEN OUR CELLS AND PARENT LINEAGES
#for each cell type we searched the associated parent lineages
searched_lin=['AB', 'MS', 'E', 'C', 'D', 'P4', 'not \n labelled']

n_cell_cel_type_lin=np.zeros((len(unique_union_cell_type), len(searched_lin)))
count_out=0
for i in range(N_cell):
    ind_cel_type=np.where(unique_union_cell_type_conserved==union_cell_type[i])[0]
    count=0
    if 'AB' in lineage[i]:
        n_cell_cel_type_lin[int(ind_cel_type)][0]=n_cell_cel_type_lin[int(ind_cel_type)][0]+1
        count=count+1
    if 'MS' in lineage[i]:
        n_cell_cel_type_lin[int(ind_cel_type)][1]=n_cell_cel_type_lin[int(ind_cel_type)][1]+1
        count=count+1
    if 'E' in lineage[i]:
        n_cell_cel_type_lin[int(ind_cel_type)][2]=n_cell_cel_type_lin[int(ind_cel_type)][2]+1
        count=count+1
    if 'C' in lineage[i]:
        n_cell_cel_type_lin[int(ind_cel_type)][3]=n_cell_cel_type_lin[int(ind_cel_type)][3]+1
        count=count+1
    if 'D' in lineage[i]:
        n_cell_cel_type_lin[int(ind_cel_type)][4]=n_cell_cel_type_lin[int(ind_cel_type)][4]+1
        count=count+1
    if 'Z2/Z3' in lineage[i]:
        n_cell_cel_type_lin[int(ind_cel_type)][5]=n_cell_cel_type_lin[int(ind_cel_type)][5]+1
        count=count+1
    if 'nan' in lineage[i]:
        n_cell_cel_type_lin[int(ind_cel_type)][6]=n_cell_cel_type_lin[int(ind_cel_type)][6]+1
        count=count+1
    
    if count>1:
        count_out=count_out+1
        print(i,lineage[i], union_cell_type[i])

#save the data
np.savetxt(path_save_data+'n_cell_cel_type_lin.txt', n_cell_cel_type_lin, fmt='%d')

np.sum(n_cell_cel_type_lin)-count_out
print(N_cell)

# plt.figure(figsize=(6, 12), dpi=600)
# plt.imshow(np.log(n_cell_cel_type_lin), cmap='copper_r', aspect='auto')
# cbar=plt.colorbar(shrink=0.5, aspect=15)
# cbar.set_label('# cells (log scale)', size=20)  # Aquí puedes ajustar el tamaño como desees
# plt.grid(False)
# plt.ylabel('Cell type', fontsize=20)
# plt.xlabel('Lineage', fontsize=20)
# plt.xticks(np.arange(len(searched_lin)), searched_lin, fontsize=14, fontweight='bold')  # Set text labels.
# plt.yticks(np.arange(len(unique_union_cell_type)), unique_union_cell_type, fontsize=3)
# plt.savefig(path_save_data+"relation_cell_type_lin.png", dpi=600, bbox_inches='tight')
# plt.show()       


#5) We search for the matching gene in the WPO associatoins data sheet
#5.1.) We read the txt generated in the folder phen_space
path_read='C:\\Users\\logslab\\Desktop\\D-P_rule_paper\\matrix_construction\\phen_space\\'  
f=open(path_read+'\\gene_id_WPO_associations.txt', 'r')
txt = f.read()
gene_id_WPO = txt.split('\n')
del txt, f
gene_id_WPO=np.array(gene_id_WPO)

#5.2.) Intersection of genes from paper and genes from WPO
genes_intersection=np.intersect1d(genes_id, gene_id_WPO)
N_genes=len(genes_intersection)



#6) DEVEPLOPMENTAL MATRIX CREATION (sorting the data)
magic_coordinates=np.multiply(len(unique_union_cell_type_conserved), len(unique_time))
magic_sum_cells_matrix_new=[]
final_genes=[]
new_data=[]
for i in range(N_genes):
    ind_gene=np.where(genes_id==genes_intersection[i])[0]
    index_sc=np.where(data[:, int(ind_gene)]>0)[0]
    if len(index_sc)>0:
        final_genes.append(genes_intersection[i])
        new_data.append(data[:, int(ind_gene)])
        exp_gene=np.zeros(magic_coordinates)
        for j in range(len(index_sc)):
            ind_cell_type=np.where(unique_union_cell_type_conserved==union_cell_type[int(index_sc[j])])[0]
            ind_time=np.where(unique_time==time[int(index_sc[j])])[0]
            real_index=ind_time*len(unique_union_cell_type_conserved)+ind_cell_type
            exp_gene[int(real_index)]=exp_gene[int(real_index)]+1    
        magic_sum_cells_matrix_new.append(exp_gene)

magic_sum_cells_matrix_new=np.array(magic_sum_cells_matrix_new)
N_genes=len(final_genes)

#6.1.) We copy genes, n_cells_per_coord_matrix
#We save the gene name 
gene_name_final=[]
for i in range(len(final_genes)):
    ind_gene=np.where(genes_id==final_genes[i])[0]
    gene_name_final.append(genes[int(ind_gene)])

np.savetxt(path_save_data+'genes_name.txt', gene_name_final, fmt='%s')
np.savetxt(path_save_data+'genes_id.txt', final_genes, fmt='%s')

np.savetxt(path_save_data+'n_cells_per_coord_matrix.txt', magic_sum_cells_matrix_new, fmt='%d')


# #6.1.1.) We compute the mean_cells_per_coord_matrix
# magic_mean_cells_matrix_new=np.zeros((N_genes, magic_coordinates))
# for i in range(N_genes):
#     index=np.where(magic_sum_cells_matrix_new[i, :]>0)[0]
#     for j in range(len(index)):
#         ind_cel_type=index[j]%len(unique_union_cell_type_conserved)
#         ind_time=np.trunc(index[j]/len(unique_union_cell_type_conserved))
#         magic_mean_cells_matrix_new[i][int(index[j])]=magic_sum_cells_matrix_new[i][int(index[j])]/m_types[int(ind_cel_type)][int(ind_time)]
        
# np.savetxt(path_save_data+'mean_cells_per_coord_matrix.txt', magic_mean_cells_matrix_new, fmt='%.6f')

#6.2.) We create the binarized_matrix
magic_on_off_matrix=np.zeros((N_genes, magic_coordinates))
for i in range(N_genes):
    for j in range(magic_coordinates):
        if magic_sum_cells_matrix_new[i][j]>0:
            magic_on_off_matrix[i][j]=1

np.savetxt(path_save_data+'binarized_matrix.txt', magic_on_off_matrix, fmt='%d')

        
#6.3.) We check the new matrix 
del data
new_data=np.array(new_data)
data=np.transpose(new_data)
del new_data

for i in range(N_genes):
    sum1=len(np.where(data[:, i]>0)[0])
    sum2=np.sum(magic_sum_cells_matrix_new[i, :])
    if sum1!=sum2:
        print(i)



##ANOTHER WAY TO CREATE THE MATRIX!!!!!!

# #6) Other way!!!!!!! -> DEVEPLOPMENTAL MATRIX CREATION (sorting the data)

# #We create a matrix with the index of each cell type 
# #For each time we have annotated the index of the cell types in list_index_cell_type
# #For each time, in big_list_index_cell  we have annotated the index of single cells that we have to check in each coordinate
# #big_list_index_cell is a list where we have each time, then each cell type and then, each index of the single cells

# # #data is already sorted following the order of array a


# #6.1.) We build the list of expresion
# big_list_index_cell=[]
# for i in range(len(unique_time)):
#     index_per_cell_type=[]
#     for j in range(len(unique_union_cell_type)):
#         ind_cell=[]
#         if m_types[j][i]!=0:
#             ind_type=np.where(union_cell_type==unique_union_cell_type_conserved[j])[0]
#             ind_t=np.where(time==unique_time[i])[0]
#             for k in range(len(ind_type)):
#                 for t in range(len(ind_t)):
#                     if ind_t[t]==ind_type[k]:
#                         ind_cell.append(int(ind_type[k]))
#             index_per_cell_type.append(ind_cell)
#     big_list_index_cell.append(index_per_cell_type)
   
# list_index_cell_type=[]
# for i in range(len(unique_time)):
#     ind_per_time=[]
#     for j in range(len(unique_union_cell_type)):
#         if m_types[j][i]!=0:
#             ind_per_time.append(int(j))
#     list_index_cell_type.append(ind_per_time)


# #6.2.) For each time, for each cell type we searched the expression of all the genes
# big_expression_time_cell_types_all_genes=[]
# # sum_genes_per_time_type=[]

# for i in range(len(unique_time)):
#     distrib_per_cell_type=[]
#     sum_distrib_per_cell_type=[]
#     #solo tipos celulares que existen en embryo time
#     for j in range(len(list_index_cell_type[i])):
#         count=0
#         exp_gene=np.zeros((len(big_list_index_cell[i][j]), N_genes))
#         for r in big_list_index_cell[i][j]:
#             exp_gene[count, :]=(data[int(r), :]) 
#             count=count+1
#         distrib_per_cell_type.append(exp_gene)
#         # sum_distrib_per_cell_type.append(np.mean(exp_gene, axis=0))
#     big_expression_time_cell_types_all_genes.append(distrib_per_cell_type)
#     # sum_genes_per_time_type.append(sum_distrib_per_cell_type)
    
# #6.3.) We get (for each time, then cell type) a submatrix of the data is in big_expression_time_cell_types_all_genes
# #In genes_ON we get the number of cells expressing each gene. We divided this number by the total number of cells in the coordinate
# #Thus, gene expression is the mean number of cells expressing a gene in each coordinate. 

# expresion_ON_per_time_per_type=[]
# for i in range(len(unique_time)):
#     exp_per_type=[]
#     count=0
#     for j in (list_index_cell_type[i]):
#         #sumando células del mismo tipo celular 
#         searched_matrix=np.array(big_expression_time_cell_types_all_genes[i][count])
#         genes_ON=[]
#         for k in range(N_genes):
#             genes_ON.append(len(np.where(searched_matrix[:, k]!=0)[0]))#/m_types[j][i])
#         exp_per_type.append(genes_ON)
#         count=count+1
#     expresion_ON_per_time_per_type.append(exp_per_type)
    

# # #7.) DEVEPLOPMENTAL MATRIX CREATION -> FINAL MATRIZ CREATION

# magic_coordinates=np.multiply(len(unique_union_cell_type_conserved), len(unique_time))
# magic_mean_cells_matrix=np.zeros((N_genes, magic_coordinates))
# for k in range(N_genes):
#     this_ind=0
#     for i in range(len(unique_time)):
#         count=0
#         for j in range(len(unique_union_cell_type_conserved)):
#             if list_index_cell_type[i][count]==j:
#                 magic_mean_cells_matrix[k][int(this_ind)]=expresion_ON_per_time_per_type[i][count][k]
#                 if count<int(len(list_index_cell_type[i])-1):
#                     count=count+1
#             else:
#                 magic_mean_cells_matrix[k][int(this_ind)]=0
#             this_ind=this_ind+1
         
# #7.1.) We check if it is the same developmental matrix!!!
# for i in range(N_genes):
#     for j in range(magic_coordinates):
#         if magic_sum_cells_matrix_new[i][j]!=magic_mean_cells_matrix[i][j]:
#             print(i, j)






















