# -*- coding: utf-8 -*-
"""
D-P rule - Xiao data
New developmental space
New phen space

@author: Alicia
"""

import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
from scipy.cluster.hierarchy import dendrogram
from matplotlib import cm 
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
from sklearn.decomposition import NMF
from sklearn.metrics import mean_squared_error
from scipy.stats import fisher_exact
from sklearn.metrics.pairwise import pairwise_distances

path_save_data='PATH_TO_SAVE_YOUR_DATA'

#0.) Data from the paper (pleiotropy per gene)
df = pd.read_csv(path_save_data+'phen_defect_buffering_cellular.txt',  delimiter='\t')
# Eliminar las filas donde hay NaN en la columna 'phenotypes'
df_clean = df.dropna(subset=['Phenotypes '])

del df

genes_paper=np.array(df_clean['Wormbase ID '], dtype=str)
genes_name_paper=np.array(df_clean['Gene name '], dtype=str)
pleio_paper=np.array(df_clean['Phenotypes '])

genes_paper_without_spaces = [elemento.rstrip() for elemento in genes_paper]
genes_name_paper_without_spaces = [elemento.rstrip() for elemento in genes_name_paper]
genes_name_paper_without_spaces=np.array(genes_name_paper_without_spaces, dtype=str)

genes_name_paper=np.array(genes_name_paper_without_spaces)
del genes_name_paper_without_spaces
genes_paper=np.array(genes_paper_without_spaces)
del genes_paper_without_spaces


#1.) Genes - embryo association
#We read the excel
import pandas as pd
arch='_download_Supplemental tables_Table_S3.xlsx'
sheet = pd.ExcelFile(path_save_data+arch).sheet_names
df = pd.read_excel(path_save_data+arch, sheet_name=sheet[1])  # Cambia hojas[0] por el nombre deseado


genes_emb = df.iloc[:, 0].to_numpy(dtype=str)
embryos = df.iloc[:, 2].to_numpy(dtype=str)

all_unique_genes=np.unique(genes_emb)



#1.1.) We find the number of linages and phenotypes in the matrix
lin=[]
xiao_phen=[]
xiao_phen_unique=[]
#2.) We read the sheet 4
arch='_download_Supplemental tables_Table_S4.xlsx'
sheet = pd.ExcelFile(path_save_data+arch).sheet_names
for i in range(1, 7):
    df = pd.read_excel(path_save_data+arch, sheet_name=sheet[i])  # Cambia hojas[0] por el nombre deseado
    lin_inner=(df.iloc[:, 0].to_numpy(dtype=str))
    lin.append(lin_inner)
    embryo_per_fen_inner=np.array(df.columns.tolist(), dtype=str)
    phen_inner=embryo_per_fen_inner[0]
    xiao_phen_unique.append(phen_inner)
    xiao_phen.append([phen_inner]*len(lin_inner))

lin=np.concatenate(lin)
xiao_phen=np.concatenate(xiao_phen)
xiao_phen_unique=np.array(xiao_phen_unique)

#1.2.) We build the matrix genes x phen/linages

def is_valid_number(x):
    return isinstance(x, (int, float)) and not np.isnan(x) 


xiao_phen_matrix=[]
cuenta_gene=np.zeros(len(all_unique_genes))
xiao_matrix_6_phen=np.zeros((len(all_unique_genes), len(xiao_phen_unique)))
for i in range(0, 6):
    df = pd.read_excel(path_save_data+arch, sheet_name=sheet[i+1])  # Cambia hojas[0] por el nombre deseado
    embryo_per_fen_inner=np.array(df.columns.tolist(), dtype=str)
    phen_inner=embryo_per_fen_inner[0]
    embryo_per_fen_inner=np.delete(embryo_per_fen_inner, 0)
    inner_lin=df.iloc[:, 0].to_numpy(dtype=str)
    
    df_modified = df.iloc[:, 1:]
    
    
    df_modified.replace(['N.A.'], np.nan, inplace=True)
    df_modified = df_modified.apply(pd.to_numeric, errors='coerce')
    
   
    
    matriz = df_modified.to_numpy()


    if i==3:
        binary_mat = np.isin(matriz, [1.0, -1.0, 0.0]).astype(int)
    else:
        binary_mat = (matriz < 0.01).astype(int)
    

    g, ind_emb, ind_matrix=np.intersect1d(embryos, embryo_per_fen_inner, return_indices=True)
    gene_arr=genes_emb[ind_emb]
    binary_mat=binary_mat[:, ind_matrix]
   
    xiao_phen_matrix_innner=np.zeros((len(all_unique_genes), len(inner_lin)))
    for j in range(len(all_unique_genes)):
        ind_gene=np.where(gene_arr==all_unique_genes[j])[0]
        for k in range(len(ind_gene)):
            lin_not_null=np.where(binary_mat[:, int(ind_gene[k])]>0)[0]
            if len(lin_not_null)>0:
                xiao_phen_matrix_innner[j][lin_not_null]=1
                cuenta_gene[j]=1
    
    xiao_phen_matrix.append(xiao_phen_matrix_innner)
    
    ind_genes_not_null_phen=np.where(np.sum(xiao_phen_matrix_innner, axis=1)>0)[0]    
    print(len(ind_genes_not_null_phen), phen_inner)
    xiao_matrix_6_phen[ind_genes_not_null_phen, i]=1

    print(i+1)
    
    
xiao_phen_matrix = np.hstack(xiao_phen_matrix)

    
pleio=np.sum(xiao_matrix_6_phen, axis=1)
len(np.where(pleio>0)[0])
    
np.sum(cuenta_gene)
cuenta_gene[95]

#2.) Pleio comparison
#2.1.) We compare the pleiotropy from their web with the q-values computed pleio

c_g, id_use, i_paper_use=np.intersect1d(all_unique_genes, genes_paper, return_indices=True)
len(c_g)
pleio_excel_comp=pleio[id_use]
pleio_paper_comp=pleio_paper[i_paper_use]

pearsonr(pleio_excel_comp, pleio_paper_comp)
spearmanr(pleio_excel_comp, pleio_paper_comp)
plt.scatter(pleio_excel_comp, pleio_paper_comp)

#2.2.) We compare the q-values pleio with our NMF pleio
#We read commmon genes
f=open(path_save_data+'genes_id.txt', 'r')
txt = f.read()
genes_nmf_pleio = txt.split('\n')
del txt, f
genes_nmf_pleio=np.delete(genes_nmf_pleio, len(genes_nmf_pleio)-1)
genes_nmf_pleio=np.array(genes_nmf_pleio)

#We read NMF pleio score
f=open(path_save_data+'nmf_pleiotropy.txt', 'r')
txt = f.read()
pleio_score_nnmf = txt.split('\n')
del txt, f
pleio_score_nnmf=np.delete(pleio_score_nnmf, len(pleio_score_nnmf)-1)
pleio_score_nnmf=np.array(pleio_score_nnmf, dtype='f')

c_g, id_use, i_nmf=inters_paper_excel=np.intersect1d(all_unique_genes, genes_nmf_pleio, return_indices=True)
pleio_excel_comp=pleio[id_use]
pleio_nmf_comp=pleio_score_nnmf[i_nmf]

pearsonr(pleio_excel_comp, pleio_nmf_comp)
spearmanr(pleio_excel_comp, pleio_nmf_comp)

#2.3.) We compare the paper/web pleio with our NMF pleio
#This comparison was already performed
c_g, id_use, i_nmf=inters_paper_excel=np.intersect1d(genes_paper, genes_nmf_pleio, return_indices=True)
pleio_paper_comp=pleio_paper[id_use]
pleio_nmf_comp=pleio_score_nnmf[i_nmf]
len(c_g)

pearsonr(pleio_paper_comp, pleio_nmf_comp)
spearmanr(pleio_paper_comp, pleio_nmf_comp)




#4.) We create the developmental space
dev_matrix_sum_cells=np.loadtxt(path_save_data+'n_cells_per_coord_matrix.txt')

#4.1.) The 350 cell stage ~ 7h ~ 420 mins
f=open(path_save_data+'time.txt', 'r')
txt = f.read()
time = txt.split('\n')
del txt, f
time=np.delete(time, len(time)-1)
time=np.array(time, dtype=float)

ind_t=np.where(time<=420.0)[0]
short_t=time[ind_t]

m_types=np.loadtxt(path_save_data+'m_types.txt')
m_types_short=m_types[:, ind_t]
m_types_array=m_types_short.flatten('F')

del time, m_types

m_types=m_types_short
time=short_t
del m_types_short, short_t




#4.2.) We compute the fraction of cells developmental matrix
pleio=np.sum(xiao_phen_matrix, axis=1)
ind_not_null=np.where(pleio>0)[0]

genes_not_null=all_unique_genes[ind_not_null]

np.savetxt(path_save_data+'genes_xiao_dev.txt', genes_not_null, fmt='%s')

common_genes, ind_use_dev, i=np.intersect1d(genes_nmf_pleio, genes_not_null, return_indices=True)


c, ind_use_phen, i=np.intersect1d(all_unique_genes, common_genes, return_indices=True)
final_xiao_phen_matrix=xiao_phen_matrix[ind_use_phen, :]
np.savetxt(path_save_data+'xiao_phen_matrix.txt', final_xiao_phen_matrix, fmt='%f')


dev_matrix_fraction_cells=np.zeros((len(common_genes), len(dev_matrix_sum_cells[0, :])))
for i in range(len(ind_use_dev)):
    for j in range(len(m_types_array)):
        if m_types_array[j]>0:
            dev_matrix_fraction_cells[i][j]=dev_matrix_sum_cells[int(ind_use_dev[i])][j]/m_types_array[j]

np.savetxt(path_save_data+'dev_matrix_fraction_cells.txt', dev_matrix_fraction_cells, fmt='%f')


#5.) We compute the similarities
from scipy.spatial.distance import pdist, squareform

sim_dev=1-pdist(dev_matrix_fraction_cells, metric='cosine')
sim_phen=1-pdist(final_xiao_phen_matrix, metric='cosine')

pearsonr(sim_dev, sim_phen)
spearmanr(sim_dev, sim_phen)


dist_dev=pdist(dev_matrix_fraction_cells, metric='cosine')
sim_dev_matrix=1-squareform(dist_dev)

dist_phen=pdist(final_xiao_phen_matrix, metric='cosine')
sim_phen_matrix=1-squareform(dist_phen)



average_sim_dev_per_gen=np.median(sim_dev_matrix, axis=1)
average_sim_phen_per_gen=np.median(sim_phen_matrix, axis=1)

pearsonr(average_sim_dev_per_gen, average_sim_phen_per_gen)
spearmanr(average_sim_dev_per_gen, average_sim_phen_per_gen)


average_sim_dev_per_gen_sorted=np.sort(average_sim_dev_per_gen)
index_Dev_sort=np.argsort(average_sim_dev_per_gen)
average_sim_phen_per_gen_sorted=np.zeros(len(average_sim_dev_per_gen))
for i in range(len(index_Dev_sort)):
    average_sim_phen_per_gen_sorted[i]=average_sim_phen_per_gen[int(index_Dev_sort[i])]

#↓esto con funcion
serie1 = pd.Series(average_sim_dev_per_gen_sorted)
serie2 = pd.Series(average_sim_phen_per_gen_sorted)

sw_dev = serie1.rolling(window=100, center=False).mean()
sw_phen = serie2.rolling(window=100, center=False).mean()

# # figura
plt.figure(figsize=(5, 5), dpi=600)
plt.scatter(sw_dev, sw_phen, s=0.5, color='slateblue')
plt.xlabel('<sim-D>', fontsize=22, fontweight='bold')
plt.ylabel('<sim-P>', fontsize=22, fontweight='bold')
plt.xticks(fontsize=18, rotation=90)
plt.yticks(fontsize=18)
plt.savefig(path_save_data+'DP_rule_frac_cells_xiao.png', dpi=600, bbox_inches='tight')
plt.show()







