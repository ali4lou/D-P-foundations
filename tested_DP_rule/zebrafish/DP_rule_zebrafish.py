# -*- coding: utf-8 -*-
"""

D-P rule in zebrafish with anatomical pehnotype ontology

"""
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
from sklearn.decomposition import NMF
import statsmodels.api as sm


# Function to calculate smoothed_y (to create a function to predict any value of x)
def calculate_smoothed_y(x, loess_x, loess_y):
    smoothed_y = np.zeros_like(x)
    
    for i, xi in enumerate(x):
        # Encuentra los dos puntos en loess_x más cercanos a xi
        idx = np.searchsorted(loess_x, xi)
        
        if idx == 0:
            # Si xi es menor que el primer punto de loess_x
            smoothed_y[i] = loess_y[0]
        elif idx == len(loess_x):
            # Si xi es mayor que el último punto de loess_x
            smoothed_y[i] = loess_y[-1]
        else:
            # Interpolación lineal entre los dos puntos más cercanos
            x0, x1 = loess_x[idx - 1], loess_x[idx]
            y0, y1 = loess_y[idx - 1], loess_y[idx]
            smoothed_y[i] = y0 + (y1 - y0) * (xi - x0) / (x1 - x0)
    
    return smoothed_y



def count_stages(array_genes):
    frac_stages=np.zeros(len(array_genes))
    n_cells_per_stage=np.zeros(len(array_genes))
    for i in range(len(array_genes)):
        ind_gene=np.where(genes==array_genes[i])[0]
        frac_stages[i]=len(np.where(matrix_dev[int(ind_gene), :]>0)[0])/n_coord_with_cells
        n_cells_per_stage[i]=np.sum(matrix_dev[int(ind_gene), :])/n_coord_with_cells
        
    return frac_stages, n_cells_per_stage



def enrichement_phen(submatrix_gene_phen, gene_array):

    #numberof times that a phenotype is associated with a gene
    phen_n_times_all_genes=np.zeros(len(phen))
    for i in range(len(phen)):
        phen_n_times_all_genes[i]=np.sum(submatrix_phen[:, i])

    odd_ratio_enrich=np.zeros(len(phen))
    p_value_enrich=np.zeros(len(phen))
    n_genes_subset=len(gene_array)
    phen_enrich_fisher_genes_subset=[]
    n_genes_subset_associated_phen=[]
    p_val_subset=[]
    #For each phenotype we compute a score that indicates if the phenotypes is enriched
    for fen in range(len(phen)):
        phen_n_times_subset=np.sum(submatrix_gene_phen[:, fen])
        if phen_n_times_subset>0:
            tabla=[[phen_n_times_subset, n_genes_subset-phen_n_times_subset],[phen_n_times_all_genes[fen], len(genes)-phen_n_times_all_genes[fen]]]
            odd_ratio_enrich[fen], p_value_enrich[fen] = fisher_exact(tabla, alternative="greater") 
            if p_value_enrich[fen]<0.001:
                phen_enrich_fisher_genes_subset.append(phen[fen])
                n_genes_subset_associated_phen.append(phen_n_times_subset)
                p_val_subset.append(p_value_enrich[fen])

    return np.array(phen_enrich_fisher_genes_subset), np.array(n_genes_subset_associated_phen, dtype=int), np.array(p_val_subset, dtype=float)



path_save_data='PATH_TO_SAVE_YOUR_DATA'



#1.) We are going to read all the data

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


#1.4.) We read developmental matrices
# dev_matrix_pseudo_bulk=np.loadtxt(path_dev+'pseudo_bulk_matrix.txt')
dev_matrix_pseudo_bulk=np.loadtxt(path_save_data+'frac_cell_per_coord_dev_matrix.txt')

#1.5.) We read the genes od development
# f=open(path_dev+'genes_pseudo_bulk.txt', 'r')
# f=open(path_dev+'genes_pseudo_more_cell_types.txt', 'r')
f=open(path_save_data+'genes_frac_cell_matrix.txt', 'r')
txt = f.read()
genes_pseudo_bulk = txt.split('\n')
del txt, f
genes_pseudo_bulk=np.delete(genes_pseudo_bulk, len(genes_pseudo_bulk)-1)
genes_pseudo_bulk=np.array(genes_pseudo_bulk)



m_types=np.loadtxt(path_save_data+'m_types.txt')
n_coord_with_cells=len(np.where(m_types>0)[0])



#1.7.) We search the common genes and rebuild the matrices
genes, ind_phen, ind_dev =np.intersect1d(genes_associated_phen, genes_pseudo_bulk, return_indices=True)
matrix_dev=dev_matrix_pseudo_bulk[ind_dev, :]
# matrix_phen=phen_matrix[ind_phen, :]
submatrix_phen=phen_matrix[ind_phen, :]



# Initialize the NMF model with deterministic initialization using 'nndsvd'
n_components = 100  # The number of latent components (reduced dimensions)
model = NMF(n_components=n_components, init='nndsvd', random_state=42, max_iter=500)

# Fit the model to the data and perform the transformation
W = model.fit_transform(submatrix_phen)  # The reduced representation of the data
H = model.components_  # The latent components (patterns)

np.savetxt(path_save_data+'W_matrix.txt', W, fmt='%f')
np.savetxt(path_save_data+'H_matrix.txt', H, fmt='%f')
np.savetxt(path_save_data+'final_good_developmental_matrix.txt', matrix_dev, fmt='%f')
np.savetxt(path_save_data+'final_good_phen_matrix.txt', submatrix_phen, fmt='%f')
np.savetxt(path_save_data+'final_good_genes.txt', genes, fmt='%s')


# 2) Similarities calculation
dist_dev=pdist(matrix_dev, metric='cosine')
dist_phen=pdist(W, metric='cosine')

dist_dev_matrix=squareform(dist_dev)
sim_dev_matrix=1-dist_dev_matrix

dist_phen_matrix=squareform(dist_phen)
sim_phen_matrix=1-dist_phen_matrix

#2.1.) We compute the median of the sim_dev and sim_phen in each gene
average_sim_dev_per_gen=np.median(sim_dev_matrix, axis=1)
average_sim_phen_per_gen=np.median(sim_phen_matrix, axis=1)

sim_phen=1-dist_phen
sim_dev=1-dist_dev

print('Pearson: average similaties per gene:', pearsonr(average_sim_dev_per_gen, average_sim_phen_per_gen))
print('Spearman: average similaties per gene:', spearmanr(average_sim_dev_per_gen, average_sim_phen_per_gen))

print('Pearson between pairwise sim:', pearsonr(sim_dev, sim_phen))
print('Spearman between pairwise sim:', spearmanr(sim_dev, sim_phen))


np.savetxt(path_save_data+'sim_dev_cosine.txt', sim_dev, fmt='%f')
np.savetxt(path_save_data+'sim_phen_jaccard.txt', sim_phen, fmt='%f')



#3.) D-P rule analysis with Sliding Window
average_sim_dev_per_gen_sorted=np.sort(average_sim_dev_per_gen)
index_Dev_sort=np.argsort(average_sim_dev_per_gen)
average_sim_phen_per_gen_sorted=np.zeros(len(average_sim_dev_per_gen))
for i in range(len(index_Dev_sort)):
    average_sim_phen_per_gen_sorted[i]=average_sim_phen_per_gen[int(index_Dev_sort[i])]

serie1 = pd.Series(average_sim_dev_per_gen_sorted)
serie2 = pd.Series(average_sim_phen_per_gen_sorted)

sw_dev = serie1.rolling(window=100, center=False).mean()
sw_phen = serie2.rolling(window=100, center=False).mean()

# # figure
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
plt.savefig(path_save_data+'DP_rule.png', dpi=600, bbox_inches='tight')
plt.show()



#4.)percentiles
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


#We defined the groups of simP using percentiles
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
plt.savefig(path_phen+'percentiles_DP_rule_big_RNAi_W_norm.png', dpi=600, bbox_inches='tight')
plt.show()


#======================================================================
#======================================================================
#5.) DEVIATIONS to the RULE

#5.1.) LOESS CURVE
x=average_sim_dev_per_gen
y=average_sim_phen_per_gen


# Fit a loess model considering density
# 5.1.1.) Calculate the KDE
from scipy.stats import gaussian_kde
# Calculate the KDE
data = np.vstack([x, y])
kde = gaussian_kde(data)
density_values = kde(data)  # Get density values for the original data points

# Normalize the density values to be between 0 and 1 for weighting
density_values = density_values / np.max(density_values)

# Fit a loess model (without weights since lowess does not accept them)
loess_result = sm.nonparametric.lowess(y, x, frac=0.3)

# Extract smoothed values
loess_x = loess_result[:, 0]
loess_y = loess_result[:, 1]


#5.2.) DEVIATTIONS TO THE MODEL
smoothed_y = calculate_smoothed_y(x, loess_x, loess_y)

#5.2.1.) Compute the residuals to the loess curve 
residuals = y - smoothed_y


#5.2.2.) Take the extremes of the residuals
p_high=np.percentile(residuals, 90)
p_low=np.percentile(residuals, 10)
std_res=np.std(residuals)
media_res=np.mean(residuals)

plt.hist(residuals)

median_x=np.median(x)

#5.2.3.) We group the genes in subsets that fullfil and not the dp rule
genes_with_highp_res=[]
x_high_res=[]
y_high_res=[]
genes_with_lowp_res=[]
x_low_res=[]
y_low_res=[]
genes_dp_rule=[]
x_dp=[]
y_dp=[]

p75_dev=np.percentile(average_sim_dev_per_gen, 75)
p25_dev=np.percentile(average_sim_dev_per_gen, 25)


for i in range(len(genes)):
    #p_low are those from the right part of the plot
    if (residuals[i]<=p_low) & (x[i]>=p75_dev):
        genes_with_lowp_res.append(genes[i])
        x_low_res.append(x[i])
        y_low_res.append(y[i])
    #p_high are those from the left part of the plot
    if (residuals[i]>=p_high) & (x[i]<=p25_dev):
        genes_with_highp_res.append(genes[i])
        x_high_res.append(x[i])
        y_high_res.append(y[i])
    if (residuals[i]>(media_res-std_res)) & (residuals[i]<(media_res+std_res)) & (x[i]>=p75_dev):
        genes_dp_rule.append(genes[i])
        x_dp.append(x[i])
        y_dp.append(y[i])

genes_with_highp_res=np.array(genes_with_highp_res)
genes_with_lowp_res=np.array(genes_with_lowp_res)
x_high_res=np.array(x_high_res)
x_low_res=np.array(x_low_res)


np.savetxt(path_save_data+'genes_dp_rule.txt', genes_dp_rule, fmt='%s')
np.savetxt(path_save_data+'genes_dP_dev_deviation.txt', genes_with_highp_res, fmt='%s')
np.savetxt(path_save_data+'genes_Dp_phen_deviation.txt', genes_with_lowp_res, fmt='%s')


#5.2.4.)Plot the subgroups that deviate from the model
plt.figure(figsize=(5, 4), dpi=600)
plt.plot(loess_x, loess_y, color="blue", label="Loess Curve", lw=1.5)
plt.scatter(x, y, c=density_values, cmap='Grays', alpha=0.3, s=5)
plt.scatter(x_dp, y_dp, label="D-P rule", color="deepskyblue", alpha=0.2, s=5)
plt.scatter(x_low_res, y_low_res, label="D-p", color="blueviolet", alpha=0.4, s=5)
plt.scatter(x_high_res, y_high_res, label="d-P", color="deeppink", alpha=0.4, s=5)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', markerscale=5, fontsize=16)
plt.xlabel('<sim-D>', fontsize=22, fontweight='bold')
plt.ylabel('<sim-P>', fontsize=22, fontweight='bold')
plt.savefig(path_save_data+'deviations.png', dpi=600, bbox_inches='tight')
plt.show()



#5.3.) Fraction of stages (developmental profiles))
frac_stages_high_perc, n_cells_per_stage_high_perc=count_stages(genes_with_highp_res)
frac_stages_low_perc, n_cells_per_stage_low_perc=count_stages(genes_with_lowp_res)
frac_stages_dp, n_cells_per_stage_dp=count_stages(genes_dp_rule)

#figure
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(9, 3), dpi=600)
axes[0].hist(frac_stages_dp, bins=10, color='deepskyblue')
axes[0].set_title('D-P', fontsize=16, fontweight='bold')
axes[0].set_ylabel('# genes', fontsize=16)
axes[1].hist(frac_stages_low_perc, bins=10, color='blueviolet')
axes[1].set_title('D-p', fontsize=16, fontweight='bold')
axes[2].hist(frac_stages_high_perc, bins=30, color='deeppink')
axes[2].set_title('d-P', fontsize=16, fontweight='bold')
for ax in axes:
    ax.grid(False)
    ax.tick_params(labelsize=15)
fig.supxlabel('Fraction of developmental coordinates', fontsize=16)
plt.tight_layout()
plt.savefig(path_save_data + 'separate_x_distrib_frac_dev_coord.png',
            dpi=600, bbox_inches='tight')
plt.show()



pleio_score_nnmf=np.sum(W, axis=1)
#5.4.) D-P rule and pleiotropy
pleio_DP_rule=np.zeros(len(genes_dp_rule))
for i in range(len(genes_dp_rule)):
    ind_gene=np.where(genes==genes_dp_rule[i])[0]
    pleio_DP_rule[i]=pleio_score_nnmf[int(ind_gene)]


pleio_Dp_dev_P=np.zeros(len(genes_with_lowp_res))
for i in range(len(genes_with_lowp_res)):
    ind_gene=np.where(genes==genes_with_lowp_res[i])[0]
    pleio_Dp_dev_P[i]=pleio_score_nnmf[int(ind_gene)]

pleio_dP_Dev_D=np.zeros(len(genes_with_highp_res))
for i in range(len(genes_with_highp_res)):
    ind_gene=np.where(genes==genes_with_highp_res[i])[0]
    pleio_dP_Dev_D[i]=pleio_score_nnmf[int(ind_gene)]



#figure
plt.figure(figsize=(5, 4))

sb.kdeplot(pleio_Dp_dev_P, color='blueviolet', fill=False, lw=1.5,  label='(D-p) - P deviation')
median_Dp_dev_P = np.median(pleio_Dp_dev_P)
plt.axvline(median_Dp_dev_P, color='blueviolet', linestyle='--', linewidth=1)

sb.kdeplot(pleio_dP_Dev_D, color='deeppink', fill=False,lw=1.5, label='(d-P) - D deviation')
median_dP_Dev_D = np.median(pleio_dP_Dev_D)
plt.axvline(median_dP_Dev_D, color='deeppink', linestyle='--', linewidth=1)

sb.kdeplot(pleio_DP_rule, color='deepskyblue', fill=False, lw=1.5, label='D-P rule')
median_DP_rule = np.median(pleio_DP_rule)
plt.axvline(median_DP_rule, color='deepskyblue', linestyle='--', linewidth=1)

plt.xlabel('NMF pleiotropy', fontsize=20)
plt.ylabel('# genes', fontsize=20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.legend(fontsize=13)
plt.grid(False)

plt.tight_layout()
plt.savefig(path_save_data+'fig2C.png', dpi=600, bbox_inches='tight')
plt.show()


#6.) Enriched phen
g, ind_low_perc, i=np.intersect1d(genes, genes_with_lowp_res, return_indices=True)
mat_phen_low_perc=submatrix_phen[ind_low_perc, :]

g, ind_high_perc, i=np.intersect1d(genes, genes_with_highp_res, return_indices=True)
mat_phen_high_perc=submatrix_phen[ind_high_perc, :]

g, ind_dp, i=np.intersect1d(genes, genes_dp_rule, return_indices=True)
mat_phen_dp=submatrix_phen[ind_dp, :]


phenotipos_enrich_fisher_genes_with_lowp_res, n_genes_lowp_res_ass_phen, p_val_low=enrichement_phen(mat_phen_low_perc, genes_with_lowp_res)
phenotipos_enrich_fisher_genes_with_highp_res, n_genes_highp_res_ass_phen, p_val_high=enrichement_phen(mat_phen_high_perc, genes_with_highp_res)
phenotipos_enrich_fisher_genes_dp, n_genes_dp_ass_phen, p_val_dp=enrichement_phen(mat_phen_dp, genes_dp_rule)


com_dp_low=np.intersect1d(phenotipos_enrich_fisher_genes_dp, phenotipos_enrich_fisher_genes_with_lowp_res)
com_dp_high=np.intersect1d(phenotipos_enrich_fisher_genes_dp, phenotipos_enrich_fisher_genes_with_highp_res)
com_low_high=np.intersect1d(phenotipos_enrich_fisher_genes_with_lowp_res, phenotipos_enrich_fisher_genes_with_highp_res)


data_Dp_P_dev = np.array(list(zip(phenotipos_enrich_fisher_genes_with_lowp_res, n_genes_lowp_res_ass_phen, p_val_low)))
data_dP_D_dev = np.array(list(zip(phenotipos_enrich_fisher_genes_with_highp_res, n_genes_highp_res_ass_phen, p_val_high)))
data_DP = np.array(list(zip(phenotipos_enrich_fisher_genes_dp, n_genes_dp_ass_phen, p_val_dp)))

data_Dp_P_dev = pd.DataFrame(data_Dp_P_dev, columns=['Phenotype', '# Genes', 'p-Value'])
data_dP_D_dev = pd.DataFrame(data_dP_D_dev, columns=['Phenotype', '# Genes', 'p-Value'])
data_DP = pd.DataFrame(data_DP, columns=['Phenotype', '# Genes', 'p-Value'])

# Salvando em CSV
data_Dp_P_dev.to_csv(path_save_data+'phenotipos_enrich_fisher_genes_Dp_deviationP.csv', index=False, sep='\t')
data_dP_D_dev.to_csv(path_save_data+'phenotipos_enrich_fisher_genes_dP_deviationD.csv', index=False, sep='\t')
data_DP.to_csv(path_save_data+'phenotipos_enrich_fisher_gene_DP.csv', index=False, sep='\t')





