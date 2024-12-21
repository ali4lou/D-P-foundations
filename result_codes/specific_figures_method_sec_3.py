# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 18:44:46 2024

Section 3 -> specific figures of components and cell types method relation
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



def quality_control(adata):
    #we calculate QC metrics
    sc.pp.calculate_qc_metrics(adata, percent_top=None, log1p=False, inplace=True)
    #we eliminate genes that are not expressed in at least 3 cells
    sc.pp.filter_genes(adata, min_cells=3)
    #we eliminate cells that do not achieve 1000 UMIs 
    sc.pp.filter_cells(adata, min_counts=1000)
    
    return adata
    
# path_save_data='D:\\DPrule\\D-P_rule_paper\\section3\\'
path_save_data='C:\\Users\\logslab\\Desktop\\D-P_rule_paper\\section3\\'

#1.) Developmental data: cells x genes matrix

# adata = anndata.read('D:\\atlas_gusanos_bien\\packer2019.h5ad')
adata = anndata.read('C:\\Users\\logslab\\Desktop\\papeles alicia\\packer2019.h5ad')
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
# path_dev='D:\\DPrule\\D-P_rule_paper\\matrix_construction\\dev_space\\'
# path_phen='D:\\DPrule\\D-P_rule_paper\\matrix_construction\\phen_space\\'
# path_pleio='D:\\DPrule\\D-P_rule_paper\\NNMF_justification\\'
path_dev='C:\\Users\\logslab\\Desktop\\D-P_rule_paper\\matrix_construction\\dev_space\\'
path_phen='C:\\Users\\logslab\\Desktop\\D-P_rule_paper\\matrix_construction\\phen_space\\'
path_pleio='C:\\Users\\logslab\\Desktop\\D-P_rule_paper\\NNMF_justification\\'

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

#1.5) We read cell_types and times
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



#6.)Link phenotype-cell types and phenotypes-times

p=82
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
res_specific=[]
res_bulk=[]
ks=[]
p_value_list=[]
for indice, cel_type_k in enumerate(cell_types):
    n_cel_k=n_cells_per_type[indice]
    res_bulk = [residuals[i] for i in range(N_cell) if cell_type_all_cells[i] != cel_type_k] 
    res_specific = [residuals[i] for i in range(N_cell) if cell_type_all_cells[i] == cel_type_k] 
    
    ks_statistic, p_value = kstest(res_bulk, res_specific,  alternative='greater') 
    
    ks.append(ks_statistic)
    p_value_list.append(p_value)
          
    if (p_value<0.0001) & (n_cel_k>50):      
                
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
        plt.savefig(path_save_data+'\\comp_82\\comp_82_%s.png' %cel_type_k, dpi=600, bbox_inches='tight')
        plt.show()
                
        count=count+1
            