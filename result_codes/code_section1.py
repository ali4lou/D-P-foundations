
"""
Section 1: D-P rule
"""
import scanpy as sc
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import warnings; warnings.simplefilter('ignore')
from scipy.stats import mstats, kstest, ttest_ind, fisher_exact
import csv
import math
from scipy.stats import pearsonr
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics.pairwise import pairwise_distances

"""
path_save_data, path_dev, and path_phen 
are the path that you chose after download the needed files
"""

path_save_data='C:\\Users\\logslab\\Desktop\\D-P_rule_paper\\section1\\'
# path_save_data='D:\\DPrule\\D-P_rule_paper\\section1\\'


#1.) We are going to read all the data
# path_dev='D:\\DPrule\\D-P_rule_paper\\matrix_construction\\dev_space\\'
# path_phen='D:\\DPrule\\D-P_rule_paper\\matrix_construction\\phen_space\\'
path_dev='C:\\Users\\logslab\\Desktop\\D-P_rule_paper\\matrix_construction\\dev_space\\'
path_phen='C:\\Users\\logslab\\Desktop\\D-P_rule_paper\\matrix_construction\\phen_space\\'

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

#1.4.) We read developmental matrices
dev_matrix_binary=np.loadtxt(path_dev+'binarized_matrix.txt')
# dev_matrix_sum_cells=np.loadtxt(path_dev+'n_cells_per_coord_matrix.txt')
dev_matrix_sum_cells=np.loadtxt(path_dev+'dev_matrix_fraction_cells.txt')


m_types=np.loadtxt(path_dev+'m_types.txt')
n_coord_with_cells=len(np.where(m_types>0)[0])

m_types_array=m_types.flatten('F')

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

# #1.6.) We compute the fraction of cells developmental matrix
# dev_matrix_fraction_cells=np.zeros((len(genes), len(dev_matrix_sum_cells[0, :])))
# for i in range(len(genes)):
#     for j in range(len(m_types_array)):
#         if m_types_array[j]>0:
#             dev_matrix_fraction_cells[i][j]=dev_matrix_sum_cells[i][j]/m_types_array[j]

# np.savetxt(path_dev+'dev_matrix_fraction_cells.txt', dev_matrix_fraction_cells, fmt='%f')


#2) Similarities calculation
# # #2.1.) Calculation with binary matrices (jaccard))
# sim_dev_jaccard=1-pdist(dev_matrix_binary, metric='jaccard')
# sim_phen_jaccard=1-pdist(phen_matrix, metric='jaccard')

# np.savetxt(path_save_data+'sim_dev_binary_matrix_jaccard.txt', sim_dev_jaccard, fmt='%f')
# np.savetxt(path_save_data+'sim_phen_matrix_jaccard.txt', sim_phen_jaccard, fmt='%f')

# # #2.2.) Calculation using W phen matrix and sum_per_coord dev matrix with using braycurtis similarity
# sim_dev=1-pdist(dev_matrix_sum_cells, metric='braycurtis')
# sim_phen=1-pdist(W, metric='braycurtis')

# np.savetxt(path_save_data+'sim_dev_sum_cell_matrix_braycurtis.txt', sim_dev, fmt='%f')
# np.savetxt(path_save_data+'sim_W_matrix_braycurtis.txt', sim_phen, fmt='%f')

# #2.2.) Calculation using W phen matrix and sum_per_coord dev matrix with using cosine similarity
# sim_dev=1-pdist(dev_matrix_sum_cells, metric='cosine')
# sim_phen=1-pdist(W, metric='cosine')

# np.savetxt(path_save_data+'sim_dev_sum_cell_matrix_cosine.txt', sim_dev, fmt='%f')
# np.savetxt(path_save_data+'sim_W_matrix_cosine.txt', sim_phen, fmt='%f')

# # #2.1.) Calculation with fraction of cells expressing the gene (cosine))
# sim_dev_frac_cells_cosine=1-pdist(dev_matrix_fraction_cells, metric='cosine')

# np.savetxt(path_save_data+'sim_dev_frac_cells_cosine.txt', sim_dev_frac_cells_cosine, fmt='%f')



#2.3) We read the similarities txt
# sim_dev=np.loadtxt(path_save_data+'sim_dev_sum_cell_matrix_cosine.txt')
sim_dev=np.loadtxt(path_save_data+'sim_dev_frac_cells_cosine.txt')
sim_phen=np.loadtxt(path_save_data+'sim_W_matrix_cosine.txt')

sim_dev_jaccard=np.loadtxt(path_save_data+'sim_dev_binary_matrix_jaccard.txt')
sim_phen_jaccard=np.loadtxt(path_save_data+'sim_phen_matrix_jaccard.txt')


plt.figure(figsize=(4, 3),dpi=600)
plt.hist(sim_phen, bins=100, color='violet')
plt.xlabel('Phen pairwise similarity', fontsize=14, fontweight='bold')
plt.ylabel('# gene pairs', fontsize=14, fontweight='bold')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.title('Cosine similarity', fontsize=10)
plt.savefig(path_save_data+'phen_pairwise_sim.png', dpi=600, bbox_inches='tight')
plt.show()


plt.figure(figsize=(4, 3),dpi=600)
plt.hist(sim_dev, bins=100, color='violet')
plt.xlabel('Dev pairwise similarity', fontsize=14, fontweight='bold')
plt.ylabel('# gene pairs', fontsize=14, fontweight='bold')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.title('Cosine similarity', fontsize=10)
plt.savefig(path_save_data+'dev_pairwise_sim.png', dpi=600, bbox_inches='tight')
plt.show()




#2.4.) Pearson and spearman coefficient
pearsonr(sim_phen, sim_dev)
# spearmanr(sim_phen, sim_phen_jaccard)
# spearmanr(sim_dev, sim_dev_jaccard)
#spearmanr(sim_phen_jaccard, sim_dev_jaccard)


#3.) D-P rule analysis with Sliding Window
#3.1.) We compute the median of the sim_dev and sim_phen in each gene
sim_dev_matrix=squareform(sim_dev)
sim_phen_matrix=squareform(sim_phen)

average_sim_dev_per_gen=np.median(sim_dev_matrix, axis=1)
average_sim_phen_per_gen=np.median(sim_phen_matrix, axis=1)

#figure of distributions of individual genes (S_dev)
count=0
fig, axs = plt.subplots(3, 3, figsize=(16, 12))
for i in range(9):
    ax = axs[i // 3, i % 3]  
    ax.hist(sim_dev_matrix[(i+i), :], bins=100, color='plum')
    if i // 3==2:
        ax.set_xlabel('S-Dev', fontsize=40, fontweight='bold')
    if i % 3==0:
        count=count+1
        if count==2:
            ax.set_ylabel('# gene pairs', fontsize=40, fontweight='bold')
    ax.tick_params(axis='x', labelsize=30)    
    ax.tick_params(axis='y', labelsize=30)    
    ax.set_title(genes[i], fontsize=15)
    ax.axvline(x=np.median(sim_dev_matrix[i, :]), color='darkviolet', linestyle='--', lw=4)
    ax.grid(False)
plt.tight_layout()
plt.subplots_adjust(wspace=0.4, hspace=0.4)
plt.savefig(path_save_data+'ind_gene_distrib_Sdev.png', dpi=600, bbox_inches='tight')
plt.show()


#figure of distributions of individual genes (S_phen)
count=0
fig, axs = plt.subplots(3, 3, figsize=(16, 12))
for i in range(9):
    ax = axs[i // 3, i % 3]  
    ax.hist(sim_phen_matrix[(i+i), :], bins=100, color='plum')
    if i // 3==2:
        ax.set_xlabel('S-Phen', fontsize=40, fontweight='bold')
    if i % 3==0:
        count=count+1
        if count==2:
            ax.set_ylabel('# gene pairs', fontsize=40, fontweight='bold')
    ax.tick_params(axis='x', labelsize=30)    
    ax.tick_params(axis='y', labelsize=30)    
    ax.set_title(genes[i], fontsize=15)
    ax.axvline(x=np.median(sim_phen_matrix[i, :]), color='darkviolet', linestyle='--', lw=4)
    ax.grid(False)
plt.tight_layout()
plt.subplots_adjust(wspace=0.4, hspace=0.4)
plt.savefig(path_save_data+'ind_gene_distrib_Sphen.png', dpi=600, bbox_inches='tight')
plt.show()


#3.1.1.) We plot the distributions of the median of the similarities
p25_dev=np.percentile(average_sim_dev_per_gen, 25)
p75_dev=np.percentile(average_sim_dev_per_gen, 75)
p25_phen=np.percentile(average_sim_phen_per_gen, 25)
p75_phen=np.percentile(average_sim_phen_per_gen, 75)


plt.figure(figsize=(4, 3),dpi=600)
plt.hist(average_sim_dev_per_gen, bins=100, color='violet')
plt.xlabel('<sim-D>', fontsize=14, fontweight='bold')
plt.ylabel('# genes', fontsize=14, fontweight='bold')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.axvline(x=p75_dev, color='indigo', linestyle='--')
plt.axvline(x=p25_dev, color='indigo', linestyle='--')
plt.text(p75_dev+0.005, 175, 'P-75', fontweight='bold')
plt.text(p25_dev+0.005, 175, 'P-25', fontweight='bold')
plt.title('Fraction of cells expression \n Cosine similarity', fontsize=10)
plt.savefig(path_save_data+'median_Sdev_distrib_frac_cells_cosine.png', dpi=600, bbox_inches='tight')
plt.show()


plt.figure(figsize=(4, 3),dpi=600)
plt.hist(average_sim_phen_per_gen, bins=100, color='violet')
plt.xlabel('<sim-P>', fontsize=14, fontweight='bold')
plt.ylabel('# genes', fontsize=14, fontweight='bold')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.axvline(x=p75_phen, color='indigo', linestyle='--')
plt.axvline(x=p25_phen, color='indigo', linestyle='--')
plt.text(p75_phen+0.003, 405, 'P-75', fontweight='bold')
plt.text(p25_phen-0.028, 405, 'P-25', fontweight='bold')
plt.title('NMF profiles \n Cosine similarity', fontsize=10)
plt.savefig(path_save_data+'median_Sphen_distrib_NMF_comp_cosine.png', dpi=600, bbox_inches='tight')
plt.show()


#3.1.2.) We study the most the genes with the most repated sim_phen
sim_phen_unique, sim_phen_unique_times=np.unique(average_sim_phen_per_gen, return_counts=True)
sim_phen_unique_times_index=np.argsort(sim_phen_unique_times)[::-1]
sim_phen_unique_times=sim_phen_unique_times[sim_phen_unique_times_index]
sim_phen_unique=sim_phen_unique[sim_phen_unique_times_index]

#We analyze the genes with the most repeated sim_p
n_comp_x=np.linspace(0, 100-1, 100, dtype=int)
non_null_comp=[]
for i in range(3):
    ind_genes=np.where(average_sim_phen_per_gen==sim_phen_unique[i])[0]
    matrix_genes=np.zeros((len(ind_genes), len(W[0, :])))
    count=0
    for j in range(len(ind_genes)):
        matrix_genes[count, :]=W[int(ind_genes[j]), :]
        count=count+1
    
    plt.figure(figsize=(15, 5), dpi=600)
    plt.imshow(matrix_genes, cmap='plasma', aspect='auto')
    cbar=plt.colorbar(shrink=0.5, aspect=15)
    cbar.set_label('Weights', size=14)  # Aquí puedes ajustar el tamaño como desees
    plt.grid(False)
    plt.yticks(fontsize=12)
    plt.ylabel('Genes', fontsize=14)
    plt.xlabel('NMF phenotypes', fontsize=14)
    plt.xticks(np.arange(len(n_comp_x)), n_comp_x, fontsize=7, rotation=90)
    plt.savefig(path_save_data+'pelos_sim_phen_profiles%d.png' %i, dpi=600, bbox_inches='tight')  
    plt.show() 

#from those nmf profiles, we analyse the original phenotypes
#of nmf comp=30, 6, 97, 13, 59, 0
nmf_comp_analyse=[0, 30, 59, 97, 6, 13]
important_phen_sorted=[]
for i in range(len(nmf_comp_analyse)):
    ind_sorted_h=np.argsort(H[int(nmf_comp_analyse[i]), :])[::-1]
    important_phen_sorted.append(phen[ind_sorted_h])
    # h=H[int(nmf_comp_analyse[i]),:]
    # print(h[ind_sorted_h])
    print('\n')

#3.2.) We compute the sliding window calculation
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
# plt.text(0.02, 0.131, 'sw = 100', fontsize=16, color='white',
#          bbox=dict(facecolor='mediumslateblue', edgecolor='mediumslateblue', boxstyle='round,pad=0.3'))
# plt.text(0.02, 0.152, 'sw = 100', fontsize=16, color='white',
#          bbox=dict(facecolor='mediumslateblue', edgecolor='mediumslateblue', boxstyle='round,pad=0.3'))
plt.savefig(path_save_data+'DP_rule_frac_cells_nmf_final_good.png', dpi=600, bbox_inches='tight')
plt.show()


#4.) D-P rule analysis with matching genes from dev percentiles in phen percentiles

#We grouped genes in subsets by simD
#We take those genes from each simD subset and we search in which subsets of simP fall

#We defined the gruops of simD using percentiles
p20_dev=np.percentile(average_sim_dev_per_gen, 20)
p40_dev=np.percentile(average_sim_dev_per_gen, 40)
p60_dev=np.percentile(average_sim_dev_per_gen, 60)
p80_dev=np.percentile(average_sim_dev_per_gen, 80)

lista1=[]
lista2=[]
lista3=[]
lista4=[]
lista5=[]
for i in range(len(genes)):
    if average_sim_dev_per_gen[i]<p20_dev:
        lista1.append(i)
    if (average_sim_dev_per_gen[i]>=p20_dev) & (average_sim_dev_per_gen[i]<p40_dev):
        lista2.append(i)
    if (average_sim_dev_per_gen[i]>=p40_dev) & (average_sim_dev_per_gen[i]<p60_dev):
        lista3.append(i)
    if (average_sim_dev_per_gen[i]>=p60_dev) & (average_sim_dev_per_gen[i]<p80_dev):
        lista4.append(i)
    if average_sim_dev_per_gen[i]>=p80_dev:
        lista5.append(i)

set_dev=[lista1, lista2, lista3, lista4, lista5]


#We defined the gruops of simP using percentiles
p20_phen=np.percentile(average_sim_phen_per_gen, 20)
p40_phen=np.percentile(average_sim_phen_per_gen, 40)
p60_phen=np.percentile(average_sim_phen_per_gen, 60)
p80_phen=np.percentile(average_sim_phen_per_gen, 80)

lista1=[]
lista2=[]
lista3=[]
lista4=[]
lista5=[]
for i in range(len(genes)):
    if average_sim_phen_per_gen[i]<p20_phen:
        lista1.append(i)
    if (average_sim_phen_per_gen[i]>=p20_phen) & (average_sim_phen_per_gen[i]<p40_phen):
        lista2.append(i)
    if (average_sim_phen_per_gen[i]>=p40_phen) & (average_sim_phen_per_gen[i]<p60_phen):
        lista3.append(i)
    if (average_sim_phen_per_gen[i]>=p60_phen) & (average_sim_phen_per_gen[i]<p80_phen):
        lista4.append(i)
    if average_sim_phen_per_gen[i]>=p80_phen:
        lista5.append(i)

set_phen=[lista1, lista2, lista3, lista4, lista5]

matrix_sD_sP=np.zeros((5,5))
for i in range(len(set_dev)):
    for j in range(len(set_phen)):
        common=len(np.intersect1d(set_phen[j], set_dev[i]))/len(set_dev[i])
        matrix_sD_sP[i][j]=common

label=['', 'p20', 'p40', 'p60', 'p80']
matrix_sD_sP=np.transpose(matrix_sD_sP)
plt.figure(figsize=(5,5),  dpi=600)
plt.imshow(matrix_sD_sP, cmap='coolwarm', interpolation='none', origin='lower')
for i in range(matrix_sD_sP.shape[0]):  
    for j in range(matrix_sD_sP.shape[1]):  
        plt.text(j, i, "{:.2f}".format(matrix_sD_sP[i, j]), ha='center', va='center', color='black', fontsize=14)
plt.xlabel('<sim-D>', fontsize=22, fontweight='bold')
plt.ylabel('<sim-P>', fontsize=22, fontweight='bold')
plt.yticks(np.arange(-0.5, 4.5), label, rotation=0, fontsize=18)
plt.xticks(np.arange(-0.5, 4.5), label, rotation=0, fontsize=18)
plt.savefig(path_save_data+'percentiles_DP_rule_frac_cells_NMG.png', dpi=600, bbox_inches='tight')
plt.show()



#5.) SimD vs frac of active dev stages
#5.1.) Computation of active dev stages per gene
frac_dev_stages=np.sum(dev_matrix_binary, axis=1)/n_coord_with_cells

#5.2.) Sliding window
average_sim_dev_per_gen_sorted=np.sort(average_sim_dev_per_gen)
index_Dev_sort=np.argsort(average_sim_dev_per_gen)
frac_dev_stages_sorted=np.zeros(len(frac_dev_stages))
for i in range(len(index_Dev_sort)):
    frac_dev_stages_sorted[i]=frac_dev_stages[int(index_Dev_sort[i])]

serie1 = pd.Series(average_sim_dev_per_gen_sorted)
serie2 = pd.Series(frac_dev_stages_sorted)

sw_dev = serie1.rolling(window=100, center=False).mean()
sw_frac = serie2.rolling(window=100, center=False).mean()

# # figure
plt.figure(figsize=(5, 5), dpi=600)
plt.scatter(sw_dev, sw_frac, s=0.5, color='slateblue')
plt.xlabel('<sim-D>', fontsize=22, fontweight='bold')
plt.ylabel('# dev coord', fontsize=22, fontweight='bold')
plt.xticks(fontsize=18, rotation=90)
plt.yticks(fontsize=18)
plt.savefig(path_save_data+'frac_dev_coord_vs_Sdev.png', dpi=600, bbox_inches='tight')
plt.show()


#6.) SimP vs frac of associated phen
#6.1.) Computation of associated phen per gene
# frac_associated_phen=np.sum(phen_matrix, axis=1)/len(phen)
frac_associated_phen=np.sum(W, axis=1)/len(W[0, :])

pleiotropy_nmf=np.sum(W, axis=1)

#6.2.) Sliding window
average_sim_phen_per_gen_sorted=np.sort(average_sim_phen_per_gen)
index_phen_sort=np.argsort(average_sim_phen_per_gen)
frac_associated_phen_sorted=np.zeros(len(frac_associated_phen))
for i in range(len(index_phen_sort)):
    frac_associated_phen_sorted[i]=pleiotropy_nmf[int(index_phen_sort[i])]

serie1 = pd.Series(average_sim_phen_per_gen_sorted)
serie2 = pd.Series(frac_associated_phen_sorted)

sw_phen = serie1.rolling(window=100, center=False).mean()
sw_frac = serie2.rolling(window=100, center=False).mean()

# # figure
plt.figure(figsize=(5, 5), dpi=600)
plt.scatter(sw_phen, sw_frac, s=0.5, color='slateblue')
plt.xlabel('<sim-P>', fontsize=22, fontweight='bold')
plt.ylabel('Pleiotropy', fontsize=22, fontweight='bold')
plt.xticks(fontsize=18, rotation=90)
plt.yticks(fontsize=18)
plt.savefig(path_save_data+'pleiotropy_vs_Sphen.png', dpi=600, bbox_inches='tight')
# plt.savefig(path_save_data+'frac_weights_W_vs_Sphen.png', dpi=600, bbox_inches='tight')
plt.show()

