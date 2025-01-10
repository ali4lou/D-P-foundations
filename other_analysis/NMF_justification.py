# -*- coding: utf-8 -*-
"""
NNMF justification 
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
from scipy import stats
from sklearn.decomposition import NMF
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster


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



"""
path_save_data, path_dev and path_phen
are the paths that you chose after download the needed files
"""

path_save_data='C:\\Users\\logslab\\Desktop\\comprobacion_dp_rule\\all_files\\'

#1.) We are going to read all the data
path_dev='C:\\Users\\logslab\\Desktop\\comprobacion_dp_rule\\all_files\\'
path_phen='C:\\Users\\logslab\\Desktop\\comprobacion_dp_rule\\all_files\\'
path_matrices='C:\\Users\\logslab\\Desktop\\comprobacion_dp_rule\\all_files\\'

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

#1.4.) We read the phenotypes that belong to the base and have many (>25) ancestor phenotypes
f=open(path_phen+'base_phen_many_ancestors.txt', 'r')
txt = f.read()
base_phen_many_ancestors = txt.split('\n')
del txt, f
base_phen_many_ancestors=np.delete(base_phen_many_ancestors, len(base_phen_many_ancestors)-1)
base_phen_many_ancestors=np.array(base_phen_many_ancestors)

#2.) PLEIOTROPY CALCULATION
pleio_score_nnmf=np.zeros(len(genes))
for i in range(len(genes)):
    pleio_score_nnmf[i]=np.sum(W[i, :])
    
np.savetxt(path_save_data+'nmf_pleiotropy.txt', pleio_score_nnmf, fmt='%f')
      
pleio_all_ontology=np.zeros(len(genes))
for i in range(len(genes)):
    pleio_all_ontology[i]=np.sum(phen_matrix[i, :])
     

#3.) NMF first justification
#Pleiotropy is reduced for biased genes (those associated with base phen with many ancestors) when using NMF pleio
genes_biased=[]
for i in range(len(base_phen_many_ancestors)):
    ind_phen=np.where(phen==base_phen_many_ancestors[i])[0]
    if len(ind_phen)>0:
        print(ind_phen)
        ind_genes=np.where(phen_matrix[:, int(ind_phen)]==1)[0]
        for j in range(len(ind_genes)):
            genes_biased.append(genes[int(ind_genes[j])])

pleio_genes_biased_all_ont=[]
pleio_genes_biased_nnmf=[]
for i in range(len(genes_biased)):
    ind_gen=np.where(genes==genes_biased[i])[0]
    pleio_genes_biased_nnmf.append(pleio_score_nnmf[int(ind_gen)])#/max_pleio_nnmf)
    pleio_genes_biased_all_ont.append(pleio_all_ontology[int(ind_gen)])#♣/max_pleio_all_ont)
    
x=pleio_all_ontology    
    
plt.figure(figsize=(5.2,4), dpi=600)
plt.scatter(pleio_all_ontology, pleio_score_nnmf, s=0.5)
plt.scatter(pleio_genes_biased_all_ont, pleio_genes_biased_nnmf, s=1.2, label='Biased genes')
slope, intercept, r_value, p_value, std_err = stats.linregress(pleio_all_ontology, pleio_score_nnmf)
y_pred = intercept + slope * pleio_all_ontology
plt.plot(pleio_all_ontology, y_pred, color='violet', label='Linear regression', lw=0.5)
confidence_interval = 1.96 * std_err * np.sqrt(1/len(x) + (x - np.mean(x))**2 / np.sum((x - np.mean(x))**2))
lower_bound = y_pred - confidence_interval
upper_bound = y_pred + confidence_interval
plt.fill_between(x, lower_bound, upper_bound, color='deepskyblue', alpha=0.5, label='Confidence interval 95%')
plt.legend(fontsize=12, markerscale=4)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.ylabel('NMF pleiotropy', fontsize=18)
plt.xlabel('All ontology pleiotropy', fontsize=18)
plt.savefig(path_save_data+'biased_genes_improving_pleio_nnmf.png', dpi=600,  bbox_inches='tight')
plt.show()


# #Residuals calculation
# residuals=[]
# for i in range(len(genes_biased)):
#     ind_gen=np.where(genes==genes_biased[i])[0]
#     residuals.append(pleio_score_nnmf[int(ind_gen)]-y_pred[int(ind_gen)])


#TECHNICAL VERIFICATIONS

# #5.) We check which is the best number of components
# n_comp=[2, 10, 50, 100, 200, 300, 400]
# mse=np.zeros(len(n_comp))
# for i in range(len(n_comp)):
#     n_components = n_comp[i]  # The number of latent components (reduced dimensions)
#     model = NMF(n_components=n_components, init='nndsvd', random_state=42, max_iter=500)

#     # Fit the model to the data and perform the transformation
#     W = model.fit_transform(phen_matrix)  # The reduced representation of the data
#     H = model.components_  # The latent components (patterns)

#     new_phen_mat=np.dot(W, H)
#     mse[i]=np.mean((phen_matrix - new_phen_mat) ** 2)
    
#     print(i)

# plt.figure(figsize=(5,4), dpi=600)
# plt.scatter(n_comp, mse, s=50, color='darkturquoise')
# plt.xticks([2, 100, 200, 300, 400], fontsize=17)
# plt.yticks(fontsize=17)
# plt.ylabel('Mean Squared Error', fontsize=20)
# plt.xlabel('# new components', fontsize=20)
# plt.savefig(path_save_data+'n_components_nnmf.png', dpi=600,  bbox_inches='tight')
# plt.show()


n_run=5
n_components=100

# #6.) We create 5 new W and H matrices with 100 components. We save them
# path_save_matrices='C:\\Users\\logslab\\Desktop\\D-P_rule_paper\\NNMF_justification\\matrices_to_compare\\'
# for i in range(2, n_run):
#     model = NMF(n_components=n_components, init='nndsvd', max_iter=500)
#     # Fit the model to the data and perform the transformation
#     W = model.fit_transform(phen_matrix)  # The reduced representation of the data
#     H = model.components_  # The latent components (patterns)

#     np.savetxt(path_save_matrices+'W_%d.txt' %i, W, fmt='%f')
#     np.savetxt(path_save_matrices+'H_%d.txt' %i, H, fmt='%f')
    
#     print(i)


#7.) We read the matrices generated in different runs to analyze the stability
W_big=[]
H_big=[]
#voy a leer cada matriz de W y voy a calcular un valor pleiotrópico 
for i in range(n_run):
    W=np.loadtxt(path_matrices+'W_%d.txt' %i, dtype=float)
    W_big.append(W)
    H=np.loadtxt(path_matrices+'H_%d.txt' %i, dtype=float)
    H_big.append(H)
    
W_big=np.array(W_big)
H_big=np.array(H_big)


#8.) Stability of W and H -> computation of coeficient of variation 

#8.1.) Calculation of the coefficient of variation of the 5 W matrices by comparing the sum of the genes (pleiotropy) W[i, :]
sum_W_100comp=np.zeros((n_components, n_run))
for i in range(n_components):
    for j in range(n_run):
        sum_W_100comp[i][j]=np.mean(W_big[j][i, :])

coeficient=np.zeros(n_components)
for i in range(n_components):
    coeficient[i]=np.std(sum_W_100comp[i, :])/np.mean(sum_W_100comp[i, :])

plt.figure(figsize=(4, 3),dpi=600)
plt.hist(coeficient, bins=50, color='darkorange')
plt.ylabel('# genes', fontsize=15, fontweight='bold')
plt.xlabel('Coefficient of variation \n 5 W matrices', fontsize=15, fontweight='bold')
plt.xticks([0, 0.1, 0.2, 0.3], fontsize=14)
plt.yticks(fontsize=14)
plt.grid(False)
plt.savefig(path_save_data+'cv_W.png', dpi=600, bbox_inches='tight')
plt.show()

# #8.2.) Calculation of the coefficient of variation of the 5 H matrices by comparing the average of each new component H[i, :]
# average_H_100comp=np.zeros((n_components, n_run))
# for i in range(n_components):
#     for j in range(n_run):
#         average_H_100comp[i][j]=np.mean(H_big[j][i, :])

# coeficient=np.zeros(n_components)
# for i in range(n_components):
#     coeficient[i]=np.std(average_H_100comp[i, :])/np.mean(average_H_100comp[i, :])

# plt.figure(figsize=(4, 3),dpi=600)
# plt.hist(coeficient, bins=50, color='darkorange')
# plt.ylabel('# new components', fontsize=15, fontweight='bold')
# plt.xlabel('Coefficient of variation \n 5 runs', fontsize=15, fontweight='bold')
# plt.xticks(fontsize=14)
# plt.yticks([0, 5, 10, 15], fontsize=14)
# plt.grid(False)
# plt.savefig(path_save_data+'cv_H.png', dpi=600, bbox_inches='tight')
# plt.show()

# #8.2.) Calculation of most correlated components between the matrix that we are using and the matrices to check
# #Stability of W
# mean_value_for_each_comp=np.zeros((len(W[0, :]), n_run))
# for i in range(n_run):
#     pearson_comparison=np.zeros((n_components, n_components))
#     for j in range(n_components):
#         for k in range(n_components):
#             r, p_value=pearsonr(W[j, :], W_big[i][k, :])
#             if p_value<0.001:
#                 pearson_comparison[j][k]=r
#                 if (k==0) & (j==0):
#                     print(r)
#             else:
#                 pearson_comparison[j][k]=0
        
#         mean_value_for_each_comp[j][i]=np.max(pearson_comparison[j, :])
        
#     np.savetxt(path_save_data+'pearson_comparison_W_matrices\\pearson_run_%d.txt' %i,  pearson_comparison, fmt='%f')
    
#     plt.figure(figsize=(5, 5), dpi=600)
#     plt.imshow(pearson_comparison, cmap='plasma', aspect='auto')
#     cbar=plt.colorbar(shrink=0.5, aspect=15)
#     cbar.set_label('Weights', size=14)  # Aquí puedes ajustar el tamaño como desees
#     plt.grid(False)
#     plt.xticks(fontsize=12)
#     plt.yticks(fontsize=12)
#     plt.ylabel('NMF phenotypes, W', fontsize=14)
#     plt.xlabel('NMF phenotypes \n W comparison %d' %i, fontsize=14)
#     plt.savefig(path_save_data+'pearson_comparison_W_matrices\\pearson_run_%d.png' %i, dpi=600, bbox_inches='tight')  
#     plt.show()    
    
# #Plot of the maximun mean value for each component
# plt.figure(figsize=(4,3), dpi=600)
# plt.hist(np.mean(mean_value_for_each_comp, axis=1), bins=50, color='slateblue')
# plt.xlabel('<Max Pearson coeff> \n W matrices', fontsize=14, fontweight='bold')
# plt.ylabel('# genes?', fontsize=14, fontweight='bold')
# plt.xticks(fontsize=12)
# plt.yticks(fontsize=12)
# plt.grid(False)
# plt.savefig(path_save_data+'pearson_coef_W_matrices_stability.png', dpi=600, bbox_inches='tight')
# plt.show()


#8.3.) Calculation of most correlated components between the matrix that we are using and the matrices to check
#Stability of H
mean_value_for_each_comp_H=np.zeros((len(W[0, :]), n_run))
for i in range(n_run):
    pearson_comparison=np.zeros((n_components, n_components))
    for j in range(n_components):
        for k in range(n_components):
            r, p_value=pearsonr(H[j, :], H_big[i][k, :])
            if p_value<0.001:
                pearson_comparison[j][k]=r
                if (k==0) & (j==0):
                    print(r)
            else:
                pearson_comparison[j][k]=0
        
        mean_value_for_each_comp_H[j][i]=np.max(pearson_comparison[j, :])

#     np.savetxt(path_save_data+'pearson_comparison_H_matrices\\pearson_run_%d.txt' %i,  pearson_comparison, fmt='%f')
    
#     plt.figure(figsize=(5, 5), dpi=600)
#     plt.imshow(pearson_comparison, cmap='plasma', aspect='auto')
#     cbar=plt.colorbar(shrink=0.5, aspect=15)
#     cbar.set_label('Weights', size=14)  # Aquí puedes ajustar el tamaño como desees
#     plt.grid(False)
#     plt.xticks(fontsize=12)
#     plt.yticks(fontsize=12)
#     plt.ylabel('NMF phenotypes, H', fontsize=14)
#     plt.xlabel('NMF phenotypes \n H comparison %d' %i, fontsize=14)
#     plt.savefig(path_save_data+'pearson_comparison_H_matrices\\pearson_run_%d.png' %i, dpi=600, bbox_inches='tight')  
#     plt.show()    
        
plt.figure(figsize=(4,3), dpi=600)
plt.hist(np.mean(mean_value_for_each_comp_H, axis=1), bins=50, color='slateblue')
plt.xlabel('<Max Pearson coeff> \n 5 H matrices', fontsize=14, fontweight='bold')
plt.ylabel('# NMF phenotypes', fontsize=14, fontweight='bold')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(False)
plt.savefig(path_save_data+'pearson_coef_H_matrices_stability.png', dpi=600, bbox_inches='tight')
plt.show()

#9.) Independence of H components
#======================================================================
#9.1.) We just see the independence in one H matrix (the one that we are going to use for all the computations)
H=np.loadtxt(path_phen+'H.txt')

#9.2.)We are going to compute the correlation distances betweenn new components to check their independence
distH=pairwise_distances(H, metric='correlation')

#9.3.) We cluster the matrix to show it sorted
n_clust, opt_clust=n_clust(distH, 2)

inner_clust=[]
unique_clusters = np.unique(opt_clust)
for cluster in unique_clusters:
    cluster_points = np.where(opt_clust == cluster)[0]
    inner_clust.append(cluster_points)
    print(f"Cluster {cluster}: {len(cluster_points)} puntos, Índices: {cluster_points}")
    
corr_H=1-distH
del distH
distH=corr_H
del corr_H

#We sort the matrix
new_matrix2=np.zeros((len(distH), len(distH)))
new_matrix22=np.zeros((len(distH), len(distH)))

count=0
for i in range(len(inner_clust)):
    for j in range(len(inner_clust[i])):
        new_matrix2[count, :]=distH[int(inner_clust[i][j]), :]
        count=count+1

count=0
for i in range(len(inner_clust)):
    for j in range(len(inner_clust[i])):
        new_matrix22[count, :]=new_matrix2[:, int(inner_clust[i][j])]
        count=count+1

#figure
plt.figure(figsize=(5.5, 4.5), dpi=600)
plt.imshow(new_matrix22, cmap='Purples', aspect='auto', origin='lower')
cbar=plt.colorbar()
cbar.set_label('Correlation', fontsize=16)  # Aquí puedes ajustar el tamaño como desees
cbar.ax.tick_params(labelsize=14) 
plt.grid(False)
plt.yticks(fontsize=14)
plt.xticks(fontsize=14)
plt.ylabel('NMF phenotypes', fontsize=16)
plt.xlabel('NMF phenotypes', fontsize=16)
plt.savefig(path_save_data+'independence_new_phen_comp_H.png', dpi=600, bbox_inches='tight')
plt.show()   



#10.) We compute the nnmf pleio and plot the histogram
nmf_pleio_p95=np.percentile(pleio_score_nnmf, 95)
nmf_pleio_p5=np.percentile(pleio_score_nnmf, 5)



plt.figure(figsize=(4,3), dpi=600)
plt.hist(pleio_score_nnmf, bins=100, color='orchid', log=True)
plt.xlabel('Pleiotropy', fontsize=14, fontweight='bold')
plt.ylabel('# genes', fontsize=14, fontweight='bold')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.axvline(x=nmf_pleio_p95, color='indigo', linestyle='--')
plt.axvline(x=nmf_pleio_p5, color='indigo', linestyle='--')
plt.text(nmf_pleio_p95+0.05, 700, 'Pleio', fontweight='bold')
plt.text(nmf_pleio_p5+0.05, 1.5, 'Non-Pleio', fontweight='bold')
plt.grid(False)
plt.savefig(path_save_data+'pleiotropy_nmf.png', dpi=600, bbox_inches='tight')
plt.show()


#10.) Correlation between number of non null nmf components per gene and pleiotropy
W=np.loadtxt(path_phen+'W.txt')

non_null_comp=np.zeros(len(genes))
for i in range(len(genes)):
    non_null_comp[i]=len(np.where(W[i, :]>0)[0])

pearsonr(non_null_comp, pleio_score_nnmf)

plt.figure(figsize=(5,4), dpi=600)
plt.scatter(non_null_comp, pleio_score_nnmf, color='red', s=1.2)
plt.ylabel('Pleiotropy', fontsize=20, fontweight='bold')
plt.xlabel('# NMF components', fontsize=20, fontweight='bold')
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.grid(False)
plt.savefig(path_save_data+'nmf_non_null_comp_vs_pleiotropy.png', dpi=600, bbox_inches='tight')
plt.show()



