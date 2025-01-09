# -*- coding: utf-8 -*-
"""
Section 2 -> PLEIOTROPY (pleio vs non pleio genes and relationship development and pleiotropy)
"""
import scanpy as sc
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import warnings; warnings.simplefilter('ignore')
import csv
import math
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
from scipy.stats import binom
import seaborn as sns
from matplotlib.colors import ListedColormap
from scipy.stats import mstats, kstest, ttest_ind, fisher_exact


def enrichement_phen(submatrix_gene_phen, gene_array):

    #numberof times that a phenotype is associated with a gene
    phen_n_times_all_genes=np.zeros(len(phen))
    for i in range(len(phen)):
        phen_n_times_all_genes[i]=np.sum(phen_matrix[:, i])

    odd_ratio_enrich=np.zeros(len(phen))
    p_value_enrich=np.zeros(len(phen))
    n_genes_subset=len(gene_array)
    phen_enrich_fisher_genes_subset=[]
    n_genes_subset_associated_phen=[]
    p_val_subset=[]
    #For each phenotype we compute a score that indicates if the phenotypes is enriched
    for fen in range(len(phen)):
        phen_n_times_subset=np.sum(submatrix_gene_phen[:, fen])
        tabla=[[phen_n_times_subset, n_genes_subset-phen_n_times_subset],[phen_n_times_all_genes[fen], len(genes)-phen_n_times_all_genes[fen]]]
        odd_ratio_enrich[fen], p_value_enrich[fen] = fisher_exact(tabla, alternative="greater") 
        if p_value_enrich[fen]<0.001:
            phen_enrich_fisher_genes_subset.append(phen[fen])
            n_genes_subset_associated_phen.append(phen_n_times_subset)
            p_val_subset.append(p_value_enrich[fen])

    return np.array(phen_enrich_fisher_genes_subset), np.array(n_genes_subset_associated_phen, dtype=int), np.array(p_val_subset, dtype=float)



"""
path_save_data, path_dev, path_phen, path_sim and path_pleio
are the path that you chose after download the needed files
"""

path_save_data='YOUR_PATH_TO_SAVE_DATA'

#1.) We are going to read all the data
path_dev='PATH_WHERE_IS_DOWNLOADED_THE_DATA'
path_phen='PATH_WHERE_IS_DOWNLOADED_THE_DATA'
path_sim='PATH_WHERE_IS_DOWNLOADED_THE_DATA'
path_pleio='PATH_WHERE_IS_DOWNLOADED_THE_DATA'


#1.1.) We read commmon genes
f=open(path_dev+'genes_id.txt', 'r')
txt = f.read()
genes = txt.split('\n')
del txt, f
genes=np.delete(genes, len(genes)-1)
genes=np.array(genes)

#1.2.) We read commmon phenotypes
f=open(path_phen+'phenotypes_description.txt', 'r')
txt = f.read()
phen = txt.split('\n')
del txt, f
phen=np.delete(phen, len(phen)-1)
phen=np.array(phen)

#1.3.) We read phenotype matrices
phen_matrix=np.loadtxt(path_phen+'gene_phen_matrix.txt')
W=np.loadtxt(path_phen+'W.txt')
H=np.loadtxt(path_phen+'H.txt')

#1.4.) We read NMF pleio score
f=open(path_pleio+'nmf_pleiotropy.txt', 'r')
txt = f.read()
pleio_score_nnmf = txt.split('\n')
del txt, f
pleio_score_nnmf=np.delete(pleio_score_nnmf, len(pleio_score_nnmf)-1)
pleio_score_nnmf=np.array(pleio_score_nnmf, dtype='f')


#1.5.) We read developmental matrices
dev_matrix_sum_cells=np.loadtxt(path_dev+'n_cells_per_coord_matrix.txt')
dev_matrix_frac_cells=np.loadtxt(path_dev+'dev_matrix_fraction_cells.txt')

m_types=np.loadtxt(path_dev+'m_types.txt')
n_coord_with_cells=len(np.where(m_types>0)[0])
N_cell=np.sum(m_types)

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
time=np.array(time)


#2) Most pleiotropic and less pleiotropic genes
nmf_pleio_p95=np.percentile(pleio_score_nnmf, 95)
nmf_pleio_p5=np.percentile(pleio_score_nnmf, 5)

high_pleio_genes=[]
high_pleio_genes_score=[]
high_pleio_genes_n_total_cells=[]
for i in range(len(pleio_score_nnmf)):
    if pleio_score_nnmf[i]>=nmf_pleio_p95:
        high_pleio_genes_score.append(pleio_score_nnmf[i])
        high_pleio_genes.append(genes[i])
        high_pleio_genes_n_total_cells.append(np.sum(dev_matrix_sum_cells[i, :]))
high_pleio_genes=np.array(high_pleio_genes)
high_pleio_genes_score=np.array(high_pleio_genes_score)
high_pleio_genes_n_total_cells=np.array(high_pleio_genes_n_total_cells)


less_pleio_genes=[]
less_pleio_genes_score=[]
less_pleio_genes_n_total_cells=[]
for i in range(len(pleio_score_nnmf)):
    if pleio_score_nnmf[i]<=nmf_pleio_p5:
        less_pleio_genes_score.append(pleio_score_nnmf[i])
        less_pleio_genes.append(genes[i])
        less_pleio_genes_n_total_cells.append(np.sum(dev_matrix_sum_cells[i, :]))
less_pleio_genes=np.array(less_pleio_genes)
less_pleio_genes_score=np.array(less_pleio_genes_score)
less_pleio_genes_n_total_cells=np.array(less_pleio_genes_n_total_cells)

np.savetxt(path_save_data+'pleio_genes.txt', high_pleio_genes, fmt='%s')
np.savetxt(path_save_data+'non_pleio_genes.txt', less_pleio_genes, fmt='%s')


mean_pleio=np.mean(high_pleio_genes_score)
mean_non_pleio=np.mean(less_pleio_genes_score)
median_pleio=np.median(high_pleio_genes_score)
median_non_pleio=np.median(less_pleio_genes_score)


#3) PLEIO VS NON PLEIO

#3.1.) Gene expression probability (number of total cells that are expressing the gene)
n_cell_all_genes=np.sum(dev_matrix_sum_cells, axis=1)

prob_all_genes=n_cell_all_genes/N_cell
prob_high_pleio_genes=high_pleio_genes_n_total_cells/N_cell
prob_less_pleio_genes=less_pleio_genes_n_total_cells/N_cell

mean_prob_high_pleio_genes=np.mean(prob_high_pleio_genes)
mean_prob_less_pleio_genes=np.mean(prob_less_pleio_genes)


#We compare the <gene expression probability> of 415 random genes vs the one of the most pleio and less pleio
n_random=int(len(high_pleio_genes)+len(less_pleio_genes))/2
n_run=10000

#high pleio
mean_comparison=[]
for n in range(n_run):
    selected = np.random.choice(prob_all_genes, size=int(n_random), replace=False)
    mean_comparison.append(np.mean(selected))


#boxplot
plt.figure(figsize=(3, 4), dpi=600)
plt.boxplot(mean_comparison, vert=True, patch_artist=True, 
            boxprops=dict(facecolor='steelblue', color='black', lw=0.7), 
            medianprops=dict(color='navy', linewidth=1.1),
            whiskerprops=dict(linewidth=0.5),  # Grosor de los bigotes
            capprops=dict(linewidth=0.5),      # Grosor de las tapitas
            flierprops=dict(marker='o', markersize=1.5, color='black', alpha=0.5))  # Tamaño de los outliers
plt.ylim(min(mean_prob_less_pleio_genes, mean_prob_high_pleio_genes) - 0.02,
         max(mean_prob_high_pleio_genes, mean_prob_less_pleio_genes) + 0.02)
plt.legend(fontsize=14)
plt.ylabel('<Gene expression prob>', fontsize=18)
plt.xticks([])
plt.yticks(fontsize=14)
plt.text(0.965, mean_prob_high_pleio_genes, 'X', color='mediumseagreen', fontweight='bold', fontsize=20)
plt.text(0.965, mean_prob_less_pleio_genes, 'X', color='darkorange', fontweight='bold', fontsize=20)
plt.text(1.05, mean_prob_high_pleio_genes-0.012, 'Pleio', fontsize=13, color='mediumseagreen')
plt.text(1.05, mean_prob_less_pleio_genes-0.012, 'Non Pleio', fontsize=13, color='darkorange')
plt.grid(False)
plt.savefig(path_save_data + 'gene_exp_prob_high_pleio_boxplot.png', dpi=600, bbox_inches='tight')
plt.show()


# #We are going to check if the average of expression probability in each subset is high or low with respect random genes from the data sheet
# n_run=10000

# #high pleio
# mean_comparison=[]
# for n in range(n_run):
#     selected = np.random.choice(prob_all_genes, size=len(high_pleio_genes), replace=False)
#     mean_comparison.append(np.mean(selected))

plt.figure(figsize=(4, 3), dpi=600)
plt.hist(mean_comparison, bins=100, color='steelblue', range=(0.06, float(mean_prob_high_pleio_genes+0.02)))
plt.legend( fontsize=14)
plt.xlabel('<Gene expression prob>', fontsize=18)
plt.ylabel('Freq', fontsize=20)
plt.xticks([0.05, 0.1, 0.15, 0.20], fontsize=18)
plt.yticks(fontsize=18)
plt.text(mean_prob_high_pleio_genes, 0, '|', color='mediumseagreen', fontweight='bold', fontsize=20)
plt.text(0.188, 130, 'Pleio', fontsize=15, color='mediumseagreen')
plt.text(mean_prob_less_pleio_genes, 0, '|', color='darkorange', fontweight='bold', fontsize=20)
plt.text(mean_prob_less_pleio_genes-0.013, 130, 'Non\npleio', fontsize=15, color='darkorange')
plt.xlim(min(mean_prob_less_pleio_genes, mean_prob_high_pleio_genes) - 0.02,
         max(mean_prob_high_pleio_genes, mean_prob_less_pleio_genes) + 0.02)
plt.grid(False)
plt.savefig(path_save_data+'gene_exp_prob_high_pleio.png', dpi=600, bbox_inches='tight')
plt.show()

# #less pleio
# mean_comparison=[]
# for n in range(n_run):
#     selected = np.random.choice(prob_all_genes, size=len(less_pleio_genes), replace=False)
#     mean_comparison.append(np.mean(selected))

# plt.figure(figsize=(4, 4), dpi=600)
# plt.hist(mean_comparison, bins=100, color='mediumslateblue', range=(float(mean_prob_less_pleio_genes-0.02), 0.115))
# plt.legend( fontsize=14)
# plt.xlabel('<Gene expression prob>', fontsize=18)
# plt.ylabel('Freq', fontsize=20)
# plt.xticks([0.05, 0.1], fontsize=18)
# plt.yticks([0, 200, 400, 600], fontsize=18)
# plt.text(mean_prob_less_pleio_genes, 0, 'X', color='red', fontweight='bold', fontsize=20)
# plt.text(0.0225, 74, 'Less pleio', fontsize=15, fontweight='bold')
# plt.grid(False)
# plt.savefig(path_save_data+'gene_exp_prob_less_pleio.png', dpi=600, bbox_inches='tight')
# plt.show()


#3.2.) Number of activated embryo times vs intial embryo time

n_times_all_genes=np.zeros(len(genes))
n_times_high_pleio_genes=[]
n_times_less_pleio_genes=[]

for i in range(len(genes)):
    ind_not_null=np.where(dev_matrix_sum_cells[i, :]>0)[0]
    not_null_times=np.zeros(len(time))
    for j in range(len(ind_not_null)):
        t=np.trunc(int(ind_not_null[j])/len(cell_types))
        not_null_times[int(t)]=1
    if len(np.where(high_pleio_genes==genes[i])[0])>0:
        n_times_high_pleio_genes.append(np.mean(not_null_times))
    if len(np.where(less_pleio_genes==genes[i])[0])>0:
        n_times_less_pleio_genes.append(np.mean(not_null_times))
    n_times_all_genes[i]=np.mean(not_null_times)

import random
#initial embryo time
t_ini_high_pleio=[]
t_ini_less_pleio=[]
t_ini_all_genes=np.linspace(-3, -3, len(genes))
for i in range(len(genes)):
    gen=dev_matrix_sum_cells[int(i), :]
    index=np.where(gen>0)[0]
    ind_time=np.trunc(index[0]/len(cell_types))
    if len(np.where(high_pleio_genes==genes[i])[0])>0:
        if (float(time[int(ind_time)]))==0:
            t_ini_high_pleio.append(float(time[int(ind_time)])-random.randrange(0,10))
        else:
            t_ini_high_pleio.append(float(time[int(ind_time)]))
    if len(np.where(less_pleio_genes==genes[i])[0])>0:
        if (float(time[int(ind_time)]))==0:
                 t_ini_less_pleio.append(float(time[int(ind_time)])+random.randrange(0, 10))
        else:
            t_ini_less_pleio.append(float(time[int(ind_time)]))
    t_ini_all_genes[i]=time[int(ind_time)]


#figure
plt.figure(figsize=(4,4), dpi=600)
plt.scatter(t_ini_less_pleio, n_times_less_pleio_genes, color='darkorange', s=25, alpha=0.5, label='Non Pleio')  # cmap es el mapa de colores
plt.scatter(t_ini_high_pleio, n_times_high_pleio_genes, color='mediumseagreen', s=25, alpha=0.5, label='Pleio') # cmap es el mapa de colores
plt.xlabel('Initial embryo time (min)', fontsize=18)
plt.ylabel('# embryo times', fontsize=18)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.legend(fontsize=16.5, markerscale=2.2)
plt.savefig(path_save_data+'n_different_times_vs_embryo_time_scatter_plot.png', dpi=600, bbox_inches='tight')
plt.show()


#3.3.) z-score related to pleiotropy
m_types_array=m_types.flatten(order='F')
#3.3.1.) We compute the z-score following a binomial distribution
z_score_high_pleio=[]
p_val_high_pleio=[]
for i in range(len(high_pleio_genes)):
    ind_gene=np.where(genes==high_pleio_genes[i])[0]
    prob=prob_high_pleio_genes[i]
    z=[]
    p_value=[]
    for j in range(len(m_types_array)):
        if (m_types_array[j]>0):
            z.append((dev_matrix_sum_cells[int(ind_gene)][j]-m_types_array[j]*prob)/(np.sqrt(m_types_array[j]*prob*(1-prob))))
            p_value.append(1-float(binom.cdf(dev_matrix_sum_cells[int(ind_gene)][j]- 1, m_types_array[j], prob)))
        else:  
            z.append(0)
            p_value.append(1)
        
    p_val_high_pleio.append(p_value)
    z_score_high_pleio.append(z)


z_score_less_pleio=[]
p_val_less_pleio=[]
for i in range(len(less_pleio_genes)):
    ind_gene=np.where(genes==less_pleio_genes[i])[0]
    prob=prob_less_pleio_genes[i]
    z=[]
    p_value=[]
    for j in range(len(m_types_array)):
        if (m_types_array[j]>0):
            z.append((dev_matrix_sum_cells[int(ind_gene)][j]-m_types_array[j]*prob)/(np.sqrt(m_types_array[j]*prob*(1-prob))))
            p_value.append(1-float(binom.cdf(dev_matrix_sum_cells[int(ind_gene)][j]- 1, m_types_array[j], prob)))
        else:  
            z.append(0)
            p_value.append(1)
        
    p_val_less_pleio.append(p_value)
    z_score_less_pleio.append(z)

z_score_less_pleio=np.array(z_score_less_pleio)
z_score_high_pleio=np.array(z_score_high_pleio)



# #3.4.) Chosing the thresholds: ANALYSIS IN DETAIL: overexpression (z>0)
# #3.4.1.) LESS PLEIO
# #z-score > 0
# median_less_pleio=[]
# weird_gene=[]
# n_cell_weird_gene=[]
# prob_weird_gene=[]
# weird_gene_name=[]
# m_stage_weird_gene=[]
# expected_gen_raro=[]
# median_weird_gene_df=[]
# var_weird_gene=[]
# x_weird_gene=[]
# z_weird_gene=[]
# p_value_weird_gene=[]

# for i in range(len(less_pleio_genes)):
#     ind_gene=np.where(genes==less_pleio_genes[i])[0]
#     array_gen_z=z_score_less_pleio[i, :]
#     ind_not_null=np.where(array_gen_z>0)[0]
#     new_array_z=array_gen_z[ind_not_null]
#     median_less_pleio.append(np.median(new_array_z))

#     if np.median(new_array_z)>30:
    
#         for k in range(len(dev_matrix_sum_cells[int(ind_gene), :])):
            
#             if z_score_less_pleio[i][k]>0:
#                 n=np.sum(dev_matrix_sum_cells[int(ind_gene), :])
#                 p=n/N_cell
#                 m=m_types_array[k]
#                 x=dev_matrix_sum_cells[int(ind_gene)][k]
#                 weird_gene.append(i)
#                 weird_gene_name.append(less_pleio_genes[i])
#                 median_weird_gene_df.append(np.median(new_array_z))
#                 n_cell_weird_gene.append(n)
#                 prob_weird_gene.append(p)
#                 m_stage_weird_gene.append(m)
#                 expected_gen_raro.append(m*p)
#                 var_weird_gene.append(np.sqrt(m*p*(1-p)))
#                 x_weird_gene.append(x)
#                 z_weird_gene.append((x-m*p)/(np.sqrt(m*p*(1-p))))
#                 p_value = 1 - binom.cdf(x - 1, m, p)
#                 p_value_weird_gene.append(p_value)
                

# df_less_pleio_weird_genes = pd.DataFrame({
#     'gene name': weird_gene_name,
#     'z-score median': median_weird_gene_df,
#     'n cells': n_cell_weird_gene, 
#     'gene exp prob': prob_weird_gene,
#     'm_ij': m_stage_weird_gene, 
#     'E(X)': expected_gen_raro, 
#     'Var(X)': var_weird_gene, 
#     'x_ij': x_weird_gene, 
#     'z-score ij': z_weird_gene, 
#     'p-value ij': p_value_weird_gene
# })


# plt.figure(figsize=(4,4), dpi=600)
# plt.scatter(prob_less_pleio_genes, median_less_pleio, s=50, color='deepskyblue', alpha=0.3)
# plt.title('Less pleiotropic genes', fontsize=17, fontweight='bold')
# plt.xlabel('<Gene expression prob>', fontsize=18)
# plt.ylabel('z-score>0 median ', fontsize=18, fontweight='bold')
# plt.xticks(fontsize=16)
# plt.yticks(fontsize=16)
# plt.grid(False)
# plt.savefig(path_save_data+'median_z_score_vs_prob_exp_less_pleio.png', dpi=600, bbox_inches='tight')
# plt.show()


# #Figure of individual distributions of the z-score per gene
# columns = 10
# rows = 4
# count = 0

# fig, ax_array = plt.subplots(rows, columns, dpi=600, sharey=True)
# plt.subplots_adjust(wspace=0.2, hspace=1.6)
# for i, ax_row in enumerate(ax_array):
#     for j, axes in enumerate(ax_row):
        
#         g = weird_gene[count]
#         ind_gene=np.where(genes==less_pleio_genes[g])[0]
#         array_gen_z = z_score_less_pleio[g, :]
#         ind_not_null = np.where(array_gen_z > 0)[0]
#         new_array_z = array_gen_z[ind_not_null]
        
#         array_cell = dev_matrix_sum_cells[int(ind_gene), :]
#         new_array_cell = array_cell[ind_not_null]

#         print(np.sum(dev_matrix_sum_cells[int(ind_gene), :]))
#         print(count)
        
#         axes.hist(new_array_z, color='mediumslateblue')
#         axes.set_title('n={}'.format(int(n_cell_weird_gene[count])), fontsize=9, fontweight='bold')
#         axes.tick_params(axis='x', labelsize=7, rotation=90)
#         axes.grid(False)
#         count += 1
    
# fig.text(0.55, 0.001, 'z-score', ha='center', va='center', fontweight='bold')
# fig.text(0.06, 0.5, 'Frequency', ha='center', va='center', rotation='vertical', fontweight='bold')
# plt.savefig(path_save_data+'distrib_z_score_example_less_pleio.png', dpi=600, bbox_inches='tight')
# plt.show()



# #3.4.2.) HIGH PLEIO
# #z-score > 0
# median_high_pleio=[]

# for i in range(len(high_pleio_genes)):
#     ind_gene=np.where(genes==high_pleio_genes[i])[0]
#     array_gen_z=z_score_high_pleio[i, :]
#     ind_not_null=np.where(array_gen_z>0)[0]
#     new_array_z=array_gen_z[ind_not_null]
#     median_high_pleio.append(np.median(new_array_z))



# plt.figure(figsize=(4,4), dpi=600)
# plt.scatter(prob_high_pleio_genes, median_high_pleio, s=50, color='hotpink', alpha=0.3)
# plt.title('High pleiotropic genes', fontsize=17, fontweight='bold')
# plt.xlabel('<Gene expression prob>', fontsize=18)
# plt.ylabel('z-score>0 median ', fontsize=18, fontweight='bold')
# plt.xticks(fontsize=16)
# plt.yticks(fontsize=16)
# plt.grid(False)
# plt.savefig(path_save_data+'median_z_score_vs_prob_exp_high_pleio.png', dpi=600, bbox_inches='tight')
# plt.show()




#3.5.) ENRICHED DEV COORD ANALYSIS

#3.5.1.) We filter the z-score values to find the enriched developmental coordinates
#high pleio
threshold_z=2
threshold_pval=0.001
new_z_score_high_pleio=np.zeros((len(high_pleio_genes), len(m_types_array)))
for i in range(len(high_pleio_genes)):
    for j in range(len(z_score_high_pleio[i])):
        if z_score_high_pleio[i][j]>threshold_z:
            if (p_val_high_pleio[i][j])<threshold_pval:
                new_z_score_high_pleio[i][j]=z_score_high_pleio[i][j]
     

#less pleio
new_z_score_less_pleio=np.zeros((len(less_pleio_genes), len(m_types_array)))
for i in range(len(less_pleio_genes)):
    for j in range(len(z_score_less_pleio[i])):
        if z_score_less_pleio[i][j]>threshold_z:
            if (p_val_less_pleio[i][j])<threshold_pval:
                new_z_score_less_pleio[i][j]=z_score_less_pleio[i][j]
     


#3.5.2. <z-score> map with the enriched coordinates

# #less pleio
# new_z_score_per_gen_robust = np.copy(new_z_score_less_pleio)
# title_label='Non Pleio genes'

#high pleio
new_z_score_per_gen_robust = np.copy(new_z_score_high_pleio)
title_label='Pleio genes'

umb=0 #(over exp)
new_z_score_per_gen_robust[new_z_score_per_gen_robust < umb] = 0

mean_new_z_score_per_gen_robust=np.mean(new_z_score_per_gen_robust, axis=0)

len(np.where(mean_new_z_score_per_gen_robust>0)[0])/n_coord_with_cells
represent_mean_new_z_score_per_gen_robust = np.copy(mean_new_z_score_per_gen_robust)

represent_mean_new_z_score_per_gen_robust_matrix=np.zeros((len(cell_types), len(time)))
for i in range(len(m_types_array)):
    ind_t=np.trunc(i/len(cell_types))
    ind_cel=i%len(cell_types)
    represent_mean_new_z_score_per_gen_robust_matrix[int(ind_cel)][int(ind_t)]=represent_mean_new_z_score_per_gen_robust[i]
    
len(np.where(represent_mean_new_z_score_per_gen_robust_matrix>0)[0])
len(np.where(m_types_array>0)[0])

grey_matrix=np.zeros((len(cell_types), len(time)))
for i in range(len(grey_matrix[:,0])):
    for j in range(len(grey_matrix[0, :])):
        if m_types[i][j]==0:
            grey_matrix[i][j]=np.log(0)
        else:
            if represent_mean_new_z_score_per_gen_robust_matrix[i][j]>0:
                grey_matrix[i][j]=np.log(0)
            else:
                grey_matrix[i][j]=1      
represent_mean_new_z_score_per_gen_robust_matrix[represent_mean_new_z_score_per_gen_robust_matrix == 0] = np.log(0)
    
#figure
fig, ax = plt.subplots(figsize=(10, 20), dpi=600)

ax.imshow(represent_mean_new_z_score_per_gen_robust_matrix, cmap='Reds_r', aspect='auto')
ax.set_title(title_label, fontweight='bold', fontsize=20)
ax.set_ylabel('Cell type', fontsize=20)
ax.set_xlabel('Embryo time (min)', fontsize=20)
ax.set_xticks(np.arange(len(time)))
ax.set_xticklabels(time, fontsize=3, rotation=90)
ax.set_yticks(np.arange(len(cell_types)))
ax.set_yticklabels(cell_types, fontsize=6)
ax.grid(False)

ax.imshow(grey_matrix, cmap='Greys_r', alpha=0.2, aspect='auto')

cbar = fig.colorbar(ax.imshow(represent_mean_new_z_score_per_gen_robust_matrix, cmap='Reds_r', aspect='auto'), ax=ax, shrink=0.7)
cbar.set_label('Mean (z-score > 0)', size=20)
cbar.ax.set_position([0.85, 0.1, 0.03, 0.8])

plt.tight_layout()
plt.savefig(path_save_data+'sup2_z_score_map_%s.png' %title_label, dpi=600, bbox_inches='tight')
plt.show()


#3.5.3.) Fraction of genes expressed in each enriched coord

frac_genes_high_pleio_signif_per_coord=np.zeros(len(m_types_array))
frac_genes_less_pleio_signif_per_coord=np.zeros(len(m_types_array))
for i in range(len(m_types_array)):
    not_null_high_pleio=np.where(new_z_score_high_pleio[:, i]>0)[0]
    not_null_less_pleio=np.where(new_z_score_less_pleio[:, i]>0)[0]

    frac_genes_high_pleio_signif_per_coord[i]=len(not_null_high_pleio)/len(high_pleio_genes)
    frac_genes_less_pleio_signif_per_coord[i]=len(not_null_less_pleio)/len(less_pleio_genes)

len(np.where(frac_genes_high_pleio_signif_per_coord>0)[0])/n_coord_with_cells
len(np.where(frac_genes_less_pleio_signif_per_coord>0)[0])/n_coord_with_cells


#3.5.3.A.) We search for the outstanding coordinates for pleio genes

n_important=100
ind_sorted_frac_genes_high_pleio = np.argsort(frac_genes_high_pleio_signif_per_coord)[::-1]
ind_sorted_frac_genes_high_pleio_most_important=ind_sorted_frac_genes_high_pleio[:n_important]
fraction_100_most_expressed=frac_genes_high_pleio_signif_per_coord[ind_sorted_frac_genes_high_pleio_most_important]
min_frac=np.min(fraction_100_most_expressed)

#We plot the histogram of the fraction of enriched coordinates for pleio genes
ind_enriched_coord_pleio_genes=np.where(frac_genes_high_pleio_signif_per_coord>0)[0]
frac_enriched_coord_pleio_genes=frac_genes_high_pleio_signif_per_coord[ind_enriched_coord_pleio_genes]


plt.figure(figsize=(4, 3), dpi=600)
plt.hist(frac_enriched_coord_pleio_genes, bins=50, color='dodgerblue', log=True)
plt.legend( fontsize=14)
plt.xlabel('# pleio genes expressed', fontsize=18)
plt.ylabel('# dev coord', fontsize=20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.axvline(x=min_frac, color='navy', linestyle='--', label='Threshold')
plt.grid(False)
plt.savefig(path_save_data+'distribution_frac_enriched_coord_pleio_genes.png', dpi=600, bbox_inches='tight')
plt.show()



ind_sorted_frac_genes_less_pleio = np.argsort(frac_genes_less_pleio_signif_per_coord)[::-1]
ind_sorted_frac_genes_less_pleio_most_important=ind_sorted_frac_genes_less_pleio[:n_important]



#figure
good_index_high_pleio=np.array(ind_sorted_frac_genes_high_pleio_most_important)
good_index_less_pleio=np.array(ind_sorted_frac_genes_less_pleio_most_important)
colors=[]
for i in range(len(m_types_array)):
    ind_pleio=np.where(good_index_high_pleio==i)[0]
    ind_no_pleio=np.where(good_index_less_pleio==i)[0]
    if (len(ind_pleio)>0) & (len(ind_no_pleio)==0):
        colors.append('mediumseagreen')
    if (len(ind_no_pleio)>0) & (len(ind_pleio)==0):
        colors.append('gray')
    if (len(ind_no_pleio)>0) & (len(ind_pleio)>0):
        colors.append('mediumseagreen')
    if (len(ind_no_pleio)==0) & (len(ind_pleio)==0):
        colors.append('gray')


plt.figure(figsize=(4, 4), dpi=600)
plt.plot(frac_genes_high_pleio_signif_per_coord, frac_genes_high_pleio_signif_per_coord, lw=1, color='steelblue')
plt.scatter(frac_genes_high_pleio_signif_per_coord, frac_genes_less_pleio_signif_per_coord, s=20, alpha=0.6, color=colors)
plt.ylabel('# non pleio genes', fontsize=20)
plt.xlabel('# pleio genes', fontsize=20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.scatter(0.005, 0.71, s=60, color='mediumseagreen', alpha=0.6)  # s determina el tamaño del punto
plt.text(0.05, 0.69, 'High # pleio genes', fontsize=16)
plt.grid(False)
plt.savefig(path_save_data+'sup2_frac_genes_expressed_per_coord.png', dpi=600, bbox_inches='tight')
plt.show()



#3.5.3.B) We keep the enriched coordinates with the higher fraction of genes expressed
n_important=100

ind_sorted_frac_genes_high_pleio = np.argsort(frac_genes_high_pleio_signif_per_coord)[::-1]
ind_sorted_frac_genes_high_pleio_most_important=ind_sorted_frac_genes_high_pleio[:n_important]
frac_genes_high_pleio_signif_per_coord_most_important=frac_genes_high_pleio_signif_per_coord[ind_sorted_frac_genes_high_pleio_most_important]
n_cell_high_pleio_most_important=m_types_array[ind_sorted_frac_genes_high_pleio_most_important]

ind_sorted_frac_genes_less_pleio = np.argsort(frac_genes_less_pleio_signif_per_coord)[::-1]
ind_sorted_frac_genes_less_pleio_most_important=ind_sorted_frac_genes_less_pleio[:n_important]
frac_genes_less_pleio_signif_per_coord_most_important=frac_genes_less_pleio_signif_per_coord[ind_sorted_frac_genes_less_pleio_most_important]
n_cell_less_pleio_most_important=m_types_array[ind_sorted_frac_genes_less_pleio_most_important]


coord_high_pleio=[]
for i in range(n_important):
    ind_t=np.trunc(ind_sorted_frac_genes_high_pleio_most_important[i]/len(cell_types))
    ind_cel=ind_sorted_frac_genes_high_pleio_most_important[i]%len(cell_types)
    list_coor=[time[int(ind_t)], cell_types[int(ind_cel)]]
    coord_high_pleio.append(list_coor)
    
coord_less_pleio=[]
for i in range(n_important):
    ind_t=np.trunc(ind_sorted_frac_genes_less_pleio_most_important[i]/len(cell_types))
    ind_cel=ind_sorted_frac_genes_less_pleio_most_important[i]%len(cell_types)
    list_coor=[time[int(ind_t)], cell_types[int(ind_cel)]]
    coord_less_pleio.append(list_coor)
    

#We are going to analize the enriched coordinates (common, high pleio, less pleio)
#there are 89 common enriched coordinates in the top
common_over=np.intersect1d(ind_sorted_frac_genes_high_pleio_most_important, ind_sorted_frac_genes_less_pleio_most_important)

n_cel_common=[]
n_cel_only_pleio=[]
n_cel_only_no_pleio=[]
t_common=[]
t_only_pleio=[]
t_only_no_pleio=[]
cel_common=[]
cel_only_pleio=[]
cel_only_no_pleio=[]
for i in range(n_important):
    if len(np.where(common_over==ind_sorted_frac_genes_high_pleio_most_important[i])[0])>0:
        n_cel_common.append(n_cell_high_pleio_most_important[i])
        ind_t=np.trunc(ind_sorted_frac_genes_high_pleio_most_important[i]/len(cell_types))
        ind_cel=ind_sorted_frac_genes_high_pleio_most_important[i]%len(cell_types)
        t_common.append(time[int(ind_t)])
        cel_common.append(cell_types[int(ind_cel)])
    else:
        n_cel_only_pleio.append(n_cell_high_pleio_most_important[i])
        ind_t=np.trunc(ind_sorted_frac_genes_high_pleio_most_important[i]/len(cell_types))
        ind_cel=ind_sorted_frac_genes_high_pleio_most_important[i]%len(cell_types)
        t_only_pleio.append(time[int(ind_t)])
        cel_only_pleio.append(cell_types[int(ind_cel)])
    if len(np.where(common_over==ind_sorted_frac_genes_less_pleio_most_important[i])[0])>0:
        print(n_cell_less_pleio_most_important[i])
    else:
        n_cel_only_no_pleio.append(n_cell_less_pleio_most_important[i])
        ind_t=np.trunc(ind_sorted_frac_genes_less_pleio_most_important[i]/len(cell_types))
        ind_cel=ind_sorted_frac_genes_less_pleio_most_important[i]%len(cell_types)
        t_only_no_pleio.append(time[int(ind_t)])
        cel_only_no_pleio.append(cell_types[int(ind_cel)])


t_common=np.array(t_common, dtype=float)
t_only_pleio=np.array(t_only_pleio, dtype=float)
n_cel_common=np.array(n_cel_common)
n_cel_only_pleio=np.array(n_cel_only_pleio)
n_cel_only_no_pleio=np.array(n_cel_only_no_pleio)
cel_common=np.array(cel_common)
cel_only_pleio=np.array(cel_only_pleio)
cel_only_no_pleio=np.array(cel_only_no_pleio)

unique_common, unique_common_n=np.unique(cel_common, return_counts=True)
unique_only_pleio, unique_only_pleio_n=np.unique(cel_only_pleio, return_counts=True)



#PLOT OF THE 100 MOST IMPORTANT ENRICHED COORDINATES (SMALL SPACE)
#Now we plot different variables: figures
#We are going to make a different plot(just showing the most important z-scores) -> the 100 most important
small_cell_types=[]
small_times=[]
time=np.array(time, dtype=float)

ind_common=np.intersect1d(ind_sorted_frac_genes_high_pleio_most_important, ind_sorted_frac_genes_less_pleio_most_important)
ind_pleio = np.setdiff1d(ind_sorted_frac_genes_high_pleio_most_important, ind_sorted_frac_genes_less_pleio_most_important)
ind_non_pleio = np.setdiff1d(ind_sorted_frac_genes_less_pleio_most_important, ind_sorted_frac_genes_high_pleio_most_important)

for i in range(len(cell_types)):
    if (len(np.where(unique_common==cell_types[i])[0])>0) or (len(np.where(unique_only_pleio==cell_types[i])[0])>0):    
        small_cell_types.append(cell_types[i])
for i in range(len(time)):
    if (len(np.where(t_common==time[i])[0])>0) or (len(np.where(t_only_pleio==time[i])[0])>0):
        small_times.append(time[i])

small_cell_types=np.array(small_cell_types)
small_times=np.array(small_times)

np.savetxt(path_save_data+'cell_types_enriched_pleio_sec2.txt', small_cell_types, fmt='%s')

new_space=np.zeros((len(small_cell_types), len(small_times)))
for i in range(len(m_types_array)):
    check=0
    if m_types_array[i]>0:
        if len(np.where(ind_common==i)[0])>0:
            
            ind_all_cell=i%len(cell_types)
            ind_small_cell=np.where(small_cell_types==cell_types[int(ind_all_cell)])[0]
            ind_all_t=np.trunc(i/len(cell_types))
            ind_small_t=np.where(small_times==time[int(ind_all_t)])[0]
            
            print(ind_small_cell, ind_small_t)
            
            new_space[int(ind_small_cell)][int(ind_small_t)]=3
            check=1
            
            
        if len(np.where(ind_pleio==i)[0])>0:
            
            ind_all_cell=i%len(cell_types)
            ind_small_cell=np.where(small_cell_types==cell_types[int(ind_all_cell)])[0]
            ind_all_t=np.trunc(i/len(cell_types))
            ind_small_t=np.where(small_times==time[int(ind_all_t)])[0]
            
            new_space[int(ind_small_cell)][int(ind_small_t)]=3
            check=1
            
        if check==0:
            ind_all_cell=i%len(cell_types)
            ind_small_cell=np.where(small_cell_types==cell_types[int(ind_all_cell)])[0]
            ind_all_t=np.trunc(i/len(cell_types))
            ind_small_t=np.where(small_times==time[int(ind_all_t)])[0]
            
            if (len(ind_small_cell)>0) & (len(ind_small_t)>0):
                new_space[int(ind_small_cell)][int(ind_small_t)]=2



cmap = ListedColormap(['white', 'gainsboro', 'mediumseagreen'])


fig, ax = plt.subplots(figsize=(8,8))
cax = ax.imshow(new_space, cmap=cmap, vmin=0, vmax=4)
cbar_ax = fig.add_axes([0.95, 0.344, 0.02, 0.3])  # [left, bottom, width, height]
cbar = fig.colorbar(cax, cax=cbar_ax, ticks=[0.7, 2, 3.3])
cbar.ax.set_yticklabels(['No sampled cells', 'Sampled cells', 'Pleio enriched'], fontsize=15)
ax.set_xlabel("Embryo time (min)", fontsize=30)
ax.set_ylabel("Cell type", fontsize=30)
ax.set_xticks(np.arange(len(small_times)), small_times, rotation=90)
ax.set_yticks(np.arange(len(small_cell_types)), small_cell_types)
ax.tick_params(axis='both', which='major', labelsize=8)
plt.savefig(path_save_data+'small_space_enriched_coord.png', dpi=600, bbox_inches='tight')
plt.show()



#4.) Enrichement of pleio and non-pleio genes
#We search in the binary phen matrix the enriched phen
phen_n_times_matching=np.zeros(len(phen))

for i in range(len(phen)):
    phen_n_times_matching[i]=np.sum(phen_matrix[:, i])
N_genes=len(genes)

#ENRICHMENT PLEIO GENES
odd_ratio_enriched=np.zeros(len(phen))
p_value_enriched=np.zeros(len(phen))
n_genes_subset=len(high_pleio_genes)
phen_enriched_fisher_pleio=[]
p_val_pleio=[]
#Para cada uno de los fenotipos voy a tener un score asociado que me va a 
#indicar si está enriquecid
matrix_phen_pleio=[]
for i in range(len(high_pleio_genes)):
    ind_gene=np.where(genes==high_pleio_genes[i])[0]
    matrix_phen_pleio.append(phen_matrix[int(ind_gene), :])
matrix_phen_pleio=np.array(matrix_phen_pleio)

n_genes_phen=[]
n_genes_phen_subset=[]
for fen in range(len(high_pleio_genes)):
    phen_n_times_subset=np.sum(matrix_phen_pleio[:, fen])
    tabla=[[phen_n_times_subset, n_genes_subset-phen_n_times_subset],[phen_n_times_matching[fen], N_genes-phen_n_times_matching[fen]]]
    odd_ratio_enriched[fen], p_value_enriched[fen] = fisher_exact(tabla, alternative="greater") 
    if p_value_enriched[fen]<0.001:
        phen_enriched_fisher_pleio.append(phen[fen])
        p_val_pleio.append(p_value_enriched[fen])
        n_genes_phen.append(phen_n_times_matching[fen])
        n_genes_phen_subset.append(phen_n_times_subset)


p_val_pleio=np.array(p_val_pleio)
n_genes_phen=np.array(n_genes_phen)
n_genes_phen_subset=np.array(n_genes_phen_subset)
phen_enriched_fisher_pleio=np.array(phen_enriched_fisher_pleio)
ind_sorted=np.argsort(p_val_pleio)
p_val_pleio_sorted=p_val_pleio[ind_sorted]
phen_enriched_fisher_pleio_sorted=phen_enriched_fisher_pleio[ind_sorted]
n_genes_phen_sorted=n_genes_phen[ind_sorted]
n_genes_phen_subset_sorted=n_genes_phen_subset[ind_sorted]


df_pleio=pd.DataFrame()
df_pleio['phen']=phen_enriched_fisher_pleio_sorted
df_pleio['p-value']=p_val_pleio_sorted
df_pleio['n_genes_phen']=n_genes_phen_sorted
df_pleio['n_genes_subset_phen']=n_genes_phen_subset_sorted
df_pleio.to_csv(path_save_data+'enriched_phen_pleio_genes.csv', sep='\t')



#ENRICHMENT NON PLEIO GENES
odd_ratio_enriched=np.zeros(len(phen))
p_value_enriched=np.zeros(len(phen))
n_genes_subset=len(less_pleio_genes)
phen_enriched_fisher_non_pleio=[]
p_val_pleio=[]
#Para cada uno de los fenotipos voy a tener un score asociado que me va a 
#indicar si está enriquecid
matrix_phen_pleio=[]
for i in range(len(less_pleio_genes)):
    ind_gene=np.where(genes==less_pleio_genes[i])[0]
    matrix_phen_pleio.append(phen_matrix[int(ind_gene), :])
matrix_phen_pleio=np.array(matrix_phen_pleio)

n_genes_phen=[]
n_genes_phen_subset=[]
for fen in range(len(less_pleio_genes)):
    phen_n_times_subset=np.sum(matrix_phen_pleio[:, fen])
    tabla=[[phen_n_times_subset, n_genes_subset-phen_n_times_subset],[phen_n_times_matching[fen], N_genes-phen_n_times_matching[fen]]]
    odd_ratio_enriched[fen], p_value_enriched[fen] = fisher_exact(tabla, alternative="greater") 
    if p_value_enriched[fen]<0.001:
        phen_enriched_fisher_non_pleio.append(phen[fen])
        p_val_pleio.append(p_value_enriched[fen])
        n_genes_phen.append(phen_n_times_matching[fen])
        n_genes_phen_subset.append(phen_n_times_subset)


p_val_pleio=np.array(p_val_pleio)
n_genes_phen=np.array(n_genes_phen)
n_genes_phen_subset=np.array(n_genes_phen_subset)
phen_enriched_fisher_non_pleio=np.array(phen_enriched_fisher_non_pleio)
ind_sorted=np.argsort(p_val_pleio)
p_val_pleio_sorted=p_val_pleio[ind_sorted]
phen_enriched_fisher_non_pleio_sorted=phen_enriched_fisher_non_pleio[ind_sorted]
n_genes_phen_sorted=n_genes_phen[ind_sorted]
n_genes_phen_subset_sorted=n_genes_phen_subset[ind_sorted]


df_non_pleio=pd.DataFrame()
df_non_pleio['phen']=phen_enriched_fisher_non_pleio_sorted
df_non_pleio['p-value']=p_val_pleio_sorted
df_non_pleio['n_genes_phen']=n_genes_phen_sorted
df_non_pleio['n_genes_subset_phen']=n_genes_phen_subset_sorted
df_non_pleio.to_csv(path_save_data+'enriched_phen_non_pleio_genes.csv', sep='\t')



#5.) Pleio vs non pleio in development
#Number of developmental coordinates, times and cell types

#pleio
n_coord_pleio=np.zeros(len(high_pleio_genes))
n_times_pleio=np.zeros(len(high_pleio_genes))
n_cell_types_pleio=np.zeros(len(high_pleio_genes))
for i in range(len(high_pleio_genes)):
    ind_gene=np.where(genes==high_pleio_genes[i])[0]
    index_coord=np.where(dev_matrix_frac_cells[int(ind_gene), :]>0)[0]
    n_coord_pleio[i]=len(index_coord)/n_coord_with_cells
    n_times=np.zeros(len(time))
    n_cell_types=np.zeros(len(cell_types))
    for j in range(len(index_coord)):
        ind_t=np.trunc(index_coord[j]/len(cell_types))
        ind_cel_type=index_coord[j]%len(cell_types)
        n_times[int(ind_t)]=1
        n_cell_types[int(ind_cel_type)]=1
    n_times_pleio[i]=np.sum(n_times)/len(time)
    n_cell_types_pleio[i]=np.sum(n_cell_types)/len(cell_types)

#non pleio
n_coord_non_pleio=np.zeros(len(less_pleio_genes))
n_times_non_pleio=np.zeros(len(less_pleio_genes))
n_cell_types_non_pleio=np.zeros(len(less_pleio_genes))
for i in range(len(less_pleio_genes)):
    ind_gene=np.where(genes==less_pleio_genes[i])[0]
    index_coord=np.where(dev_matrix_frac_cells[int(ind_gene), :]>0)[0]
    n_coord_non_pleio[i]=len(index_coord)/n_coord_with_cells
    n_times=np.zeros(len(time))
    n_cell_types=np.zeros(len(cell_types))
    for j in range(len(index_coord)):
        ind_t=np.trunc(index_coord[j]/len(cell_types))
        ind_cel_type=index_coord[j]%len(cell_types)
        n_times[int(ind_t)]=1
        n_cell_types[int(ind_cel_type)]=1
    n_times_non_pleio[i]=np.sum(n_times)/len(time)
    n_cell_types_non_pleio[i]=np.sum(n_cell_types)/len(cell_types)


plt.figure(figsize=(4, 3), dpi=600)
sns.kdeplot(n_coord_pleio, color='mediumseagreen', fill=False, lw=2,  label='Pleio')
sns.kdeplot(n_coord_non_pleio, color='darkorange', fill=False, lw=2,  label='Non-Pleio')
plt.xlabel('Frac of developmental coord', fontsize=18)
plt.ylabel('Density of genes', fontsize=18)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.legend(fontsize=13)
plt.grid(False)
plt.tight_layout()
plt.xlim(0, 1)
plt.savefig(path_save_data+'distrib_frac_dev_coord_pleio_vs_non.png', dpi=600, bbox_inches='tight')
plt.show()



plt.figure(figsize=(4, 3.5), dpi=600)
plt.scatter(n_times_pleio, n_cell_types_pleio, color='mediumseagreen', s=10, alpha=0.8)
plt.xlabel('Frac of embryo times', fontsize=18)
plt.ylabel('Frac of cell types', fontsize=18)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.legend(fontsize=13)
plt.title('Pleio genes', fontsize=18, fontweight='bold')
plt.grid(False)
plt.tight_layout()
plt.savefig(path_save_data+'scatter_pleio_genes_cell_type_vs_times.png', dpi=600, bbox_inches='tight')
plt.show()


plt.figure(figsize=(4, 3.5), dpi=600)
plt.scatter(n_times_non_pleio, n_cell_types_non_pleio, color='darkorange', s=10, alpha=0.9)
plt.xlabel('Frac of embryo times', fontsize=18)
plt.ylabel('Frac of cell types', fontsize=18)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.legend(fontsize=13)
plt.title('Non-Pleio genes', fontsize=18, fontweight='bold')
plt.grid(False)
plt.tight_layout()
plt.savefig(path_save_data+'scatter_non_pleio_genes_cell_type_vs_times.png', dpi=600, bbox_inches='tight')
plt.show()


























