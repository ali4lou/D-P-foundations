# -*- coding: utf-8 -*-
"""
Gene developmental expression matrix creation for the zebrafish analysis. 
Lange et al. data (2024)
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
from scipy.stats import linregress
from scipy.sparse import vstack


path_save_data='YOUR_PATH_TO_SAVE_DATA'


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

cell_type_unique=np.unique(all_cell_types)
time_unique=timepoints

N_cell=len(filtered_cell_types)

#3. Developemtal space -> cell type x embryo time space
#3.1.) We build a DataFrame to sort the values by time and cell type
new_index=np.linspace(0, N_cell-1, N_cell, dtype=int)
df=pd.DataFrame()
df['index']=new_index
df['time']=filtered_times
df['Category'] = pd.Categorical(df['time'], categories=time_unique, ordered=True)
df['cell_type']=filtered_cell_types
df_sort_t_cell_type=df.sort_values(by=["Category", "cell_type"])
a=df_sort_t_cell_type.to_numpy()

del df, df_sort_t_cell_type


cell_type=a[:, 3]
time=a[:, 2]
time_unique=np.array(time_unique)


cell_type_unique = []
seen = set()

for value in cell_type:
    if value not in seen:
        cell_type_unique.append(value)
        seen.add(value)

cell_type_unique = np.array(cell_type_unique)


#Save cell_type_unique and time_unique
np.savetxt(path_save_data+'cell_type_unique.txt', cell_type_unique, fmt='%s')
np.savetxt(path_save_data+'time_unique.txt', time_unique, fmt='%s')


#3.2.) We create the dev space and count the number of cells in each coordinate
m_types=np.zeros((len(cell_type_unique), len(time_unique)))

for i in range(N_cell): 
    tag=np.where(cell_type_unique==cell_type[i])[0]
    if len(tag)>0:
        tag=int(tag)
        tag_t=np.where(time_unique==time[i])[0]
        tag_t=int(tag_t)
        m_types[tag][tag_t]=m_types[tag][tag_t]+1
    
del tag, tag_t
m_types=np.array(m_types, dtype=int)


m_types_represent=np.where(m_types == 0, np.log(0), m_types)
#Figure of m_types
plt.figure(figsize=(5, 25), dpi=600)
plt.imshow(m_types_represent , cmap='plasma_r', aspect='auto')
cbar=plt.colorbar(shrink=0.5, aspect=15)
cbar.set_label('# cells', size=20)  # Aquí puedes ajustar el tamaño como desees
plt.grid(False)
plt.ylabel('Cell type', fontsize=20)
plt.xlabel('Stage', fontsize=20)
plt.xticks(np.arange(len(time_unique)), time_unique, fontsize=15, rotation=90)  # Set text labels.
plt.yticks(np.arange(len(cell_type_unique)), cell_type_unique, fontsize=10)
plt.savefig(path_save_data+'m_types.png', dpi=600, bbox_inches='tight')  
plt.show()       

magic_coordinates=np.multiply(len(cell_type_unique), len(time_unique))

m_types_array=np.array(m_types).flatten(order='F')

#4.) PSEUDO-BULK 
# total_UMIS_matrix=[]
n_cell_per_coord=[]
for i in range(len(filtered_genes)):
    index_sc=np.where(filtered_data[:, i]>0)[0]
    if len(index_sc)>0:
        exp_gene=np.zeros(magic_coordinates)
        for j in range(len(index_sc)):
            ind_cell_type=np.where(cell_type_unique==cell_type[int(index_sc[j])])[0]
            ind_time=np.where(time_unique==time[int(index_sc[j])])[0]
            real_index=ind_time*len(cell_type_unique)+ind_cell_type
            # exp_gene[int(real_index)]=exp_gene[int(real_index)]+filtered_data[int(index_sc[j])][i]    
            exp_gene[int(real_index)]=exp_gene[int(real_index)]+1 
        n_cell_per_coord.append(exp_gene)
        # total_UMIS_matrix.append(exp_gene)
    print(i)

n_cell_per_coord=np.array(n_cell_per_coord)
frac_cell_per_coord=np.zeros((len(filtered_genes), magic_coordinates))
for k in range(len(filtered_genes)):
    for i in range(magic_coordinates):
        if m_types_array[i]>0:
            frac_cell_per_coord[k][i]=n_cell_per_coord[k][i]/m_types_array[i]
        
np.savetxt(path_save_data+'frac_cell_per_coord_dev_matrix.txt', frac_cell_per_coord, fmt='%f')
np.savetxt(path_save_data+'genes_frac_cell_matrix.txt', filtered_genes, fmt='%s')



