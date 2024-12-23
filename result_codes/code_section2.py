
"""
Section 2: D-P rule DEVIATIONS
"""
import scanpy as sc
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings; warnings.simplefilter('ignore')
from scipy.stats import mstats, kstest, ttest_ind, fisher_exact
import csv
import math
from scipy.stats import pearsonr
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics.pairwise import pairwise_distances
import statsmodels.api as sm
from scipy import spatial
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster, leaves_list, optimal_leaf_ordering
from scipy.stats import mstats, kstest, ttest_ind, t



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
        frac_stages[i]=len(np.where(dev_matrix_frac_cells[int(ind_gene), :]>0)[0])/n_coord_with_cells
        n_cells_per_stage[i]=np.sum(dev_matrix_frac_cells[int(ind_gene), :])/n_coord_with_cells
        
    return frac_stages, n_cells_per_stage

def n_relative_cell_types_per_time(array_genes):
    n_cell_type_per_time_submatrix=[]
    for i in range(len(array_genes)):
        n_cell_type_per_time_per_gene=np.zeros(len(time))
        ind_gene=np.where(genes==array_genes[i])[0]
        non_null_index=np.where(dev_matrix_frac_cells[int(ind_gene), :]>0)[0]
        for j in range(len(non_null_index)):
            ind_t=np.trunc(non_null_index[j]/len(cell_types))
            n_cell_type_per_time_per_gene[int(ind_t)]=n_cell_type_per_time_per_gene[int(ind_t)]+1
        n_cell_type_per_time_per_gene=n_cell_type_per_time_per_gene/n_cell_types_per_time
        n_cell_type_per_time_submatrix.append(n_cell_type_per_time_per_gene)
    return np.mean(np.array(n_cell_type_per_time_submatrix), axis=0)
    
def average_dev_profiles(array_genes, title_label):
    average_dev_profiles_colapsing_time_submatrix=[]
    for i in range(len(array_genes)):
        average_in_time_frac_cells_per_gene=np.zeros(len(time))
        ind_gene=np.where(genes==array_genes[i])[0]
        non_null_index=np.where(dev_matrix_frac_cells[int(ind_gene), :]>0)[0]
        for j in range(len(non_null_index)):
            ind_t=np.trunc(non_null_index[j]/len(cell_types))
            average_in_time_frac_cells_per_gene[int(ind_t)]=average_in_time_frac_cells_per_gene[int(ind_t)]+dev_matrix_frac_cells[int(ind_gene)][non_null_index[j]]
        average_in_time_frac_cells_per_gene=average_in_time_frac_cells_per_gene/n_cell_types_per_time
        average_dev_profiles_colapsing_time_submatrix.append(average_in_time_frac_cells_per_gene)
        
    average_dev_profiles_colapsing_time_submatrix=np.array(average_dev_profiles_colapsing_time_submatrix)
    

    dist=pairwise_distances(average_dev_profiles_colapsing_time_submatrix, metric='euclidean')
    n_clust_phen, opt_clust, ordered_indices, Z=n_clust_sorted_tree(dist, 12, 'Average dev profiles')

    inner_clust=[]
    unique_clusters = np.unique(opt_clust)
    for cluster in unique_clusters:
        cluster_points = np.where(opt_clust == cluster)[0]
        inner_clust.append(cluster_points)
        print(f"Cluster {cluster}: {len(cluster_points)} puntos, Índices: {cluster_points}")
        

    #8.3.) Sort matrix
    new_average_dev_profiles_colapsing_time_submatrix=np.zeros((len(average_dev_profiles_colapsing_time_submatrix[:, 0]), len(average_dev_profiles_colapsing_time_submatrix[0, :])))

    count=0
    for i in range(len(ordered_indices)):
        new_average_dev_profiles_colapsing_time_submatrix[count, :]=average_dev_profiles_colapsing_time_submatrix[int(ordered_indices[i]), :]
        count=count+1

    # average_dev_profiles_colapsing_time_submatrix_represent=np.where(new_average_dev_profiles_colapsing_time_submatrix==0, np.log(0), new_average_dev_profiles_colapsing_time_submatrix)

    
    #Figure of dev profiles
    #Figure of phen prof
    if title_label=='D-P':
        size_fig=20
    else:
        size_fig=6
    
    plt.figure(figsize=(8, size_fig), dpi=600)
    plt.imshow(new_average_dev_profiles_colapsing_time_submatrix, cmap='gnuplot', aspect='auto')
    cbar=plt.colorbar(shrink=0.5, aspect=15)
    cbar.set_label('<Fraction of cells> per embryo time', size=16)  # Aquí puedes ajustar el tamaño como desees
    plt.grid(False)
    plt.ylabel('Genes', fontsize=20)
    plt.xlabel('Embryo time', fontsize=20)
    plt.title(title_label, fontsize=18, fontweight='bold')
    plt.xticks(np.arange(len(time)), np.array(time, dtype=int), rotation=90, fontsize=4)
    # plt.yticks([])
    plt.savefig(path_save_data+'average_in_time_frac_cells_prof_%s.png' %title_label, dpi=600, bbox_inches='tight')  
    plt.show()       
            
    mean_rep=np.mean(np.array(average_dev_profiles_colapsing_time_submatrix), axis=0)
    plt.figure(figsize=(15, 5), dpi=600)
    plt.imshow(mean_rep.reshape(1, -1), cmap='gnuplot', aspect='auto')
    plt.yticks([])
    plt.ylabel('<Genes>', fontsize=20)
    plt.xticks(np.arange(len(time)), time, rotation=90, fontsize=7)
    plt.xlabel('Embryo time (min)', fontsize=20)
    cbar=plt.colorbar()
    cbar.set_label('<<Frac of cells> per t>', size=15)  # Aquí puedes ajustar el tamaño como desees
    plt.title(title_label, fontsize=18, fontweight='bold')
    plt.savefig(path_save_data+'average_of_the_average_in_time_frac_cells_prof_%s.png' %title_label, dpi=600, bbox_inches='tight')  
    plt.show()     
    
    return np.mean(np.array(average_dev_profiles_colapsing_time_submatrix), axis=0)
    # return np.array(average_dev_profiles_colapsing_time_submatrix)


def count_phen(array_genes, title_label):
    most_important_phen=[]
    genes_per_phen=np.dot(W, H)
    submatrix=[]
    submatrix_W=[]
    submatrix_phen=[]
    for i in range(len(array_genes)):
        ind_gene=np.where(genes==array_genes[i])[0]
        submatrix.append(genes_per_phen[int(ind_gene), :])
        submatrix_W.append(W[int(ind_gene), :])
        submatrix_phen.append(phen_matrix[int(ind_gene), :])
    submatrix=np.array(submatrix)
    mean_per_phen=np.mean(submatrix, axis=0)
    
    #We obtain the sorted indices
    sorted_indices = np.argsort(mean_per_phen)[::-1]
    
    # sorted indices
    mean_per_phen_sorted = mean_per_phen[sorted_indices]
    phen_sorted = phen[sorted_indices]
    
    submatrix_W=np.array(submatrix_W)
    
    # #Figure of phen prof
    # plt.figure(figsize=(6, 8), dpi=600)
    # plt.imshow(submatrix_W, cmap='gnuplot', aspect='auto')
    # cbar=plt.colorbar(shrink=0.5, aspect=15)
    # cbar.set_label('Weights', size=20)  # Aquí puedes ajustar el tamaño como desees
    # plt.grid(False)
    # plt.ylabel('Genes', fontsize=20)
    # plt.xlabel('NMF phenotypes', fontsize=20)
    # plt.title(title_label, fontsize=18, fontweight='bold')
    # # plt.xticks([])  # Set text labels.
    # # plt.yticks([])
    # plt.savefig(path_save_data+'nmf_phen_profiles_%s.png' %title_label, dpi=600, bbox_inches='tight')  
    # plt.show()     
    
    dist=pairwise_distances(submatrix_W, metric='euclidean')
    n_clust_phen, opt_clust, ordered_indices, Z=n_clust_sorted_tree(dist, 12, 'Average dev profiles')

    inner_clust=[]
    unique_clusters = np.unique(opt_clust)
    for cluster in unique_clusters:
        cluster_points = np.where(opt_clust == cluster)[0]
        inner_clust.append(cluster_points)
        print(f"Cluster {cluster}: {len(cluster_points)} puntos, Índices: {cluster_points}")
        

    #Sort matrix
    new_submatrix_W=np.zeros((len(submatrix_W[:, 0]), len(submatrix_W[0, :])))

    count=0
    for i in range(len(ordered_indices)):
        new_submatrix_W[count, :]=submatrix_W[int(ordered_indices[i]), :]
        count=count+1

    # submatrix_W_rep=np.where(new_submatrix_W==0, np.log(0), new_submatrix_W)

    #Figure of phen prof
    if title_label=='D-P':
        size_fig=20
    else:
        size_fig=8
    
    plt.figure(figsize=(15, size_fig), dpi=600)
    plt.imshow(new_submatrix_W, cmap='gnuplot', aspect='auto')
    cbar=plt.colorbar(shrink=0.5, aspect=15)
    cbar.set_label('Weights', size=20)  # Aquí puedes ajustar el tamaño como desees
    plt.grid(False)
    plt.ylabel('Genes', fontsize=20)
    plt.xlabel('NMF phenotypes', fontsize=20)
    plt.title(title_label, fontsize=18, fontweight='bold')
    plt.xticks(np.linspace(0,99, 100), rotation=90, fontsize=8)
    # plt.xticks([])  # Set text labels.
    # plt.yticks([])
    plt.savefig(path_save_data+'nmf_clustered_phen_profiles_%s.png' %title_label, dpi=600, bbox_inches='tight')  
    plt.show()     
    
    
    mean_rep=np.mean(np.array(submatrix_W), axis=0)
    plt.figure(figsize=(15, 5), dpi=600)
    plt.imshow(mean_rep.reshape(1, -1), cmap='gnuplot', aspect='auto')
    plt.yticks([])
    plt.ylabel('<Genes>', fontsize=20)
    plt.xlabel('NMF pehnotypes', fontsize=20)
    cbar=plt.colorbar()
    cbar.set_label('<Weights>', size=15)  # Aquí puedes ajustar el tamaño como desees
    plt.xticks(np.linspace(0,99, 100), rotation=90, fontsize=8)
    plt.title(title_label, fontsize=18, fontweight='bold')
    plt.savefig(path_save_data+'average_nmf_phen_profiles_%s.png' %title_label, dpi=600, bbox_inches='tight')  
    plt.show()  
        
    return mean_per_phen_sorted, phen_sorted, np.array(submatrix_W), np.array(submatrix_phen)


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
f=open(path_pleio+'nmf_pleiotropy.txt', 'r')
txt = f.read()
pleio = txt.split('\n')
del txt, f
pleio=np.delete(pleio, len(pleio)-1)
pleio=np.array(pleio)


#1.4.) We read developmental matrices
dev_matrix_binary=np.loadtxt(path_dev+'binarized_matrix.txt')
dev_matrix_sum_cells=np.loadtxt(path_dev+'n_cells_per_coord_matrix.txt')
dev_matrix_frac_cells=np.loadtxt(path_dev+'dev_matrix_fraction_cells.txt')


m_types=np.loadtxt(path_dev+'m_types.txt')
n_coord_with_cells=len(np.where(m_types>0)[0])

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


#1.6.) We read the similarities
sim_dev=np.loadtxt(path_sim+'sim_dev_frac_cells_cosine.txt')
sim_phen=np.loadtxt(path_sim+'sim_W_matrix_cosine.txt')

#1.6.1.) We compute the median of the sim_dev and sim_phen in each gene
sim_dev_matrix=squareform(sim_dev)
sim_phen_matrix=squareform(sim_phen)

average_sim_dev_per_gen=np.median(sim_dev_matrix, axis=1)
average_sim_phen_per_gen=np.median(sim_phen_matrix, axis=1)

del sim_dev_matrix, sim_phen_matrix, sim_dev, sim_phen


#1.7.) We read NMF pleio score
f=open(path_pleio+'nmf_pleiotropy.txt', 'r')
txt = f.read()
pleio_score_nnmf = txt.split('\n')
del txt, f
pleio_score_nnmf=np.delete(pleio_score_nnmf, len(pleio_score_nnmf)-1)
pleio_score_nnmf=np.array(pleio_score_nnmf, dtype='f')


#2.) LOESS CURVE
x=average_sim_dev_per_gen
y=average_sim_phen_per_gen


# Fit a loess model considering density
# 2.1. Calculate the KDE
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


#3.) DEVIATTIONS TO THE MODEL
smoothed_y = calculate_smoothed_y(x, loess_x, loess_y)

#3.1.) Compute the residuals to the loess curve 
residuals = y - smoothed_y


#3.2.) Take the extremes of the residuals
p_high=np.percentile(residuals, 90)
p_low=np.percentile(residuals, 10)
std_res=np.std(residuals)
media_res=np.mean(residuals)

plt.hist(residuals)

median_x=np.median(x)

#3.3.) We group the genes in subsets that fullfil and not the dp rule
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


#3.4.)Plot the subgroups that deviate from the model
plt.figure(figsize=(5, 4), dpi=600)
plt.plot(loess_x, loess_y, color="blue", label="Loess Curve", lw=1.5)
plt.scatter(x, y, c=density_values, cmap='Grays', alpha=0.3, s=5)
plt.scatter(x_dp, y_dp, label="D-P rule", color="deepskyblue", alpha=0.2, s=5)
plt.scatter(x_low_res, y_low_res, label="D-p", color="blueviolet", alpha=0.4, s=5)
plt.scatter(x_high_res, y_high_res, label="d-P", color="deeppink", alpha=0.4, s=5)
# plt.colorbar()
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', markerscale=5, fontsize=16)
plt.xlabel('<sim-D>', fontsize=22, fontweight='bold')
plt.ylabel('<sim-P>', fontsize=22, fontweight='bold')
plt.savefig(path_save_data+'deviations.png', dpi=600, bbox_inches='tight')
plt.show()


#4.) Analyse different variables
#4.1.) Developmental profiles 

#4.1.1.) Fraction of stages (developmental profiles))
frac_stages_high_perc, n_cells_per_stage_high_perc=count_stages(genes_with_highp_res)
frac_stages_low_perc, n_cells_per_stage_low_perc=count_stages(genes_with_lowp_res)
frac_stages_dp, n_cells_per_stage_dp=count_stages(genes_dp_rule)


#We plot the fractoin of active developmental coordinates
 
colors=['deepskyblue', 'blueviolet', 'deeppink']

pos=[0, 1, 2]
label=['D-P', 'D-p', 'd-P']
plt.figure(figsize=(5, 5), dpi=600)
sns.violinplot([frac_stages_dp, frac_stages_low_perc, frac_stages_high_perc], palette=colors, inner='quartile', edgecolor='white', linewidth=1.5, cut=0)
plt.xticks(pos, label, fontsize=20, fontweight='bold')
plt.yticks(fontsize=18)
plt.grid(False)
plt.ylabel('Developmetnal \n coordinates', fontsize=20, fontweight='bold')
plt.savefig(path_save_data+'deviation_developmental_coord.png', dpi=600, bbox_inches='tight')
plt.show()


# #4.1.2.) Fraction of cell types per time 
# n_cell_types_per_time=np.zeros(len(time))
# for i in range(len(time)):
#     n_cell_types_per_time[i]=len(np.where(m_types[:, i]>0)[0])

# np.savetxt(path_save_data+'n_cell_types_per_time.txt', n_cell_types_per_time, fmt='%f')

# n_relative_cell_types_per_time_high_perc=n_relative_cell_types_per_time(genes_with_highp_res)
# n_relative_cell_types_per_time_low_perc=n_relative_cell_types_per_time(genes_with_lowp_res)
# n_relative_cell_types_per_time_dp=n_relative_cell_types_per_time(genes_dp_rule)

# np.savetxt(path_save_data+'n_relative_cell_types_per_time_dP_deviationD.txt', n_relative_cell_types_per_time_high_perc, fmt='%f')
# np.savetxt(path_save_data+'n_relative_cell_types_per_time_Dp_deviationP.txt', n_relative_cell_types_per_time_low_perc, fmt='%f')
# np.savetxt(path_save_data+'n_relative_cell_types_per_time_DP.txt', n_relative_cell_types_per_time_dp, fmt='%f')


# #4.1.2.) Average of the developmental profiles (colapsing in time) per group
n_cell_types_per_time=np.zeros(len(time))
n_cells_sampled_per_time=np.zeros(len(time))
for i in range(len(time)):
    n_cell_types_per_time[i]=len(np.where(m_types[:, i]>0)[0])
    n_cells_sampled_per_time[i]=np.sum(m_types[:, i])

#We must take into account the number of cell types sampled in each embryo time and the number of cells sampled
plt.figure(figsize=(17,2), dpi=600)
plt.imshow(np.log(n_cell_types_per_time).reshape(1, -1), cmap='gnuplot', aspect='auto')
plt.yticks([])
plt.xticks(np.arange(len(time)), time, rotation=90, fontsize=7)
plt.xlabel('Embryo time (min)', fontsize=20)
cbar=plt.colorbar()
cbar.set_label('# cell types \n (log scale)', size=15)  # Aquí puedes ajustar el tamaño como desees
plt.savefig(path_save_data+'n_cell_types_per_time_log_scale.png', dpi=600, bbox_inches='tight')  
plt.show()

plt.figure(figsize=(17,2), dpi=600)
plt.imshow(np.log(n_cells_sampled_per_time).reshape(1, -1), cmap='gnuplot', aspect='auto')
plt.yticks([])
plt.xticks(np.arange(len(time)), time, rotation=90, fontsize=7)
plt.xlabel('Embryo time (min)', fontsize=20)
cbar=plt.colorbar()
cbar.set_label('# sampled cells \n (log scale)', size=15)  # Aquí puedes ajustar el tamaño como desees
plt.savefig(path_save_data+'n_sampled_cells_per_time_log_scale.png', dpi=600, bbox_inches='tight')  
plt.show()


average_dev_prof_high_perc=average_dev_profiles(genes_with_highp_res, 'd-P (D deviation)')
average_dev_prof_low_perc=average_dev_profiles(genes_with_lowp_res, 'D-p (P deviation)')
average_dev_prof_dp=average_dev_profiles(genes_dp_rule, 'D-P')

average_dev_prof_dp_rep=np.where(average_dev_prof_dp==0, np.log(0), average_dev_prof_dp)
average_dev_prof_low_perc_rep=np.where(average_dev_prof_low_perc==0, np.log(0), average_dev_prof_low_perc)
average_dev_prof_high_perc_rep=np.where(average_dev_prof_high_perc==0, np.log(0), average_dev_prof_high_perc)

average_dev_prof_dp_min=np.where(average_dev_prof_dp==0, 10, average_dev_prof_dp)
average_dev_prof_low_perc_min=np.where(average_dev_prof_low_perc==0, 10, average_dev_prof_low_perc)
average_dev_prof_high_perc_min=np.where(average_dev_prof_high_perc==0, 10, average_dev_prof_high_perc)


#Figure
vmin = min(
    np.min(average_dev_prof_dp_min),
    np.min(average_dev_prof_low_perc_min), 
    np.min(average_dev_prof_high_perc_min)
)
vmax = max(np.max(average_dev_prof_high_perc),
    np.max(average_dev_prof_low_perc), 
    np.max(average_dev_prof_dp)
)

tamaño1=20
tamaño2=18
tamaño3=14
paleta2='magma_r'
paleta='plasma_r'


plt.figure(figsize=(15, 5), dpi=600)

ax1 = plt.subplot(3, 1, 1)
plt.title('D-P', fontsize=tamaño1, fontweight='bold')
im1 = plt.imshow(average_dev_prof_dp_rep.reshape(1, -1), cmap=paleta2, aspect='auto', vmin=vmin, vmax=vmax)
plt.yticks([])
plt.ylabel('', fontsize=tamaño1)
plt.xticks([])

ax2 = plt.subplot(3, 1, 2)
plt.title('d-P', fontsize=tamaño1, fontweight='bold')
im2 = plt.imshow(average_dev_prof_high_perc_rep.reshape(1, -1), cmap=paleta2, aspect='auto', vmin=vmin, vmax=vmax)
plt.yticks([])
plt.ylabel('', fontsize=tamaño1)
plt.xticks([])

ax3 = plt.subplot(3, 1, 3)
plt.title('D-p', fontsize=tamaño1, fontweight='bold')
im3 = plt.imshow(average_dev_prof_low_perc_rep.reshape(1, -1), cmap=paleta2, aspect='auto', vmin=vmin, vmax=vmax)
plt.yticks([])
plt.ylabel('', fontsize=tamaño1)
plt.xticks(np.arange(len(time)), time, rotation=90, fontsize=7)
plt.xlabel('Embryo time (min)', fontsize=tamaño1)

plt.subplots_adjust(hspace=0.6, right=0.85)

cbar = plt.colorbar(im1, ax=[ax1, ax2, ax3], orientation='vertical', fraction=0.02, pad=0.04)
cbar.set_label('Fraction of cells', fontsize=tamaño2)
cbar.ax.tick_params(labelsize=tamaño2)

plt.savefig(path_save_data+'average_frac_cell_prof_in_embryo_time.png', dpi=600, bbox_inches='tight')

plt.show()


#4.2.) Phenotypic profiles
mean_per_phen_high_perc, phen_high_perc, W_high_perc, mat_phen_high_perc=count_phen(genes_with_highp_res, 'd-P (D deviation)')
mean_per_phen_low_perc, phen_low_perc, W_low_perc, mat_phen_low_perc=count_phen(genes_with_lowp_res, 'D-p (P deviation)')
mean_per_phen_dp, phen_dp, W_dp, mat_phen_dp=count_phen(genes_dp_rule, 'D-P')


#4.2.1.) We plot the distribution of weights associated to the genes
sum_weights_dp=np.sum(W_dp, axis=1)
sum_weights_W_low_perc=np.sum(W_low_perc, axis=1)
sum_weights_high_perc=np.sum(W_high_perc, axis=1)

colors=['deepskyblue', 'blueviolet', 'deeppink']

pos=[0, 1, 2]
label=['D-P', 'D-p', 'd-P']
plt.figure(figsize=(5, 5), dpi=600)
sns.violinplot([sum_weights_dp, sum_weights_W_low_perc, sum_weights_high_perc], palette=colors, inner='quartile', edgecolor='white', linewidth=1.5, cut=0)
plt.xticks(pos, label, fontsize=20, fontweight='bold')
plt.yticks(fontsize=18)
plt.grid(False)
plt.ylabel('Pleiotropy', fontsize=20, fontweight='bold')
plt.savefig(path_save_data+'pleiotropy_nmf_per_gene_dp_deviation.png', dpi=600, bbox_inches='tight')
plt.show()


#KS test of the associatd pleiotropy to each group
ks_pleio, p_value_ks = kstest(sum_weights_W_low_perc, sum_weights_high_perc, alternative='greater') 
# ks_pleio, p_value_ks = kstest(sum_weights_high_perc, sum_weights_dp, alternative='greater') 
# ks_pleio, p_value_ks = kstest(sum_weights_W_low_perc, sum_weights_dp, alternative='greater') 


#4.2.2) We plot the mean phenotypic profiles
mean_W_low_perc=np.mean(W_low_perc, axis=0)
mean_W_high_perc=np.mean(W_high_perc, axis=0)
mean_W_dp=np.mean(W_dp, axis=0)


np.savetxt(path_save_data+'mean_W_dP_deviationD.txt', mean_W_high_perc, fmt='%f')
np.savetxt(path_save_data+'mean_W_Dp_deviationP.txt', mean_W_low_perc, fmt='%f')
np.savetxt(path_save_data+'mean_W_DP.txt', mean_W_dp, fmt='%f')



mean_W_dp_rep=np.where(mean_W_dp==0, np.log(0), mean_W_dp)
mean_W_low_perc_rep=np.where(mean_W_low_perc==0, np.log(0), mean_W_low_perc)
mean_W_high_perc_rep=np.where(mean_W_high_perc==0, np.log(0), mean_W_high_perc)


mean_W_dp_min=np.where(mean_W_dp==0, 10, mean_W_dp)
mean_W_low_perc_min=np.where(mean_W_low_perc==0, 10, mean_W_low_perc)
mean_W_high_perc_min=np.where(mean_W_high_perc==0, 10, mean_W_high_perc)


#Figure
vmin = min(
    np.min(mean_W_low_perc_min),
    np.min(mean_W_high_perc_min), 
    np.min(mean_W_dp_min)
)
vmax = max(np.max(mean_W_low_perc),
    np.max(mean_W_high_perc), 
    np.max(mean_W_dp)
)

tamaño1=20
tamaño2=18
tamaño3=14
paleta2='magma_r'
paleta='plasma_r'


plt.figure(figsize=(15, 5), dpi=600)

ax1 = plt.subplot(3, 1, 1)
plt.title('D-P', fontsize=tamaño1, fontweight='bold')
im1 = plt.imshow(mean_W_dp_rep.reshape(1, -1), cmap=paleta2, aspect='auto', vmin=vmin, vmax=vmax)
plt.yticks([])
plt.ylabel('', fontsize=tamaño1)
plt.xticks([])

ax2 = plt.subplot(3, 1, 2)
plt.title('d-P', fontsize=tamaño1, fontweight='bold')
im2 = plt.imshow(mean_W_high_perc_rep.reshape(1, -1), cmap=paleta2, aspect='auto', vmin=vmin, vmax=vmax)
plt.yticks([])
plt.ylabel('', fontsize=tamaño1)
plt.xticks([])

ax3 = plt.subplot(3, 1, 3)
plt.title('D-p', fontsize=tamaño1, fontweight='bold')
im3 = plt.imshow(mean_W_low_perc_rep.reshape(1, -1), cmap=paleta2, aspect='auto', vmin=vmin, vmax=vmax)
plt.yticks([])
plt.ylabel('', fontsize=tamaño1)
plt.xticks(np.linspace(0,99, 100), rotation=90, fontsize=8)
plt.xlabel('NMF components', fontsize=tamaño1)

plt.subplots_adjust(hspace=0.6, right=0.85)

cbar = plt.colorbar(im1, ax=[ax1, ax2, ax3], orientation='vertical', fraction=0.02, pad=0.04)
cbar.set_label('<Weight>', fontsize=tamaño2)
cbar.ax.tick_params(labelsize=tamaño2)

plt.savefig(path_save_data+'deviation_NMF_components.png', dpi=600, bbox_inches='tight')

plt.show()


#4.2.3.) Phenotypic enrichement
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

# np.savetxt(path_save_data+'phenotipos_enrich_fisher_genes_dP_deviationD.csv', data_dP_D_dev, fmt="%s %s %s", delimiter="\t")
# np.savetxt(path_save_data+'phenotipos_enrich_fisher_genes_Dp_deviationP.csv', data_Dp_P_dev, fmt="%s %s %s", delimiter="\t")
# np.savetxt(path_save_data+'phenotipos_enrich_fisher_gene_DP.csv', data_DP, fmt="%s %s %s", delimiter="\t")


#5.) Quantitative differences of phen and dev profiles -> pearson and euclidean 
#5.1.) Phen space
phen_profile_dP_dev_deviation = pearsonr(mean_W_dp, mean_W_high_perc)
phen_profile_Dp_phen_deviation = pearsonr(mean_W_dp, mean_W_low_perc)

#5.1.1.) We search i the nmf profiles the components with the highest differences
phen_profile_Dp_phen_deviation_subtraction = mean_W_low_perc-mean_W_dp
ind_comp_high_Dp=np.where(phen_profile_Dp_phen_deviation_subtraction>0)[0]
ind_comp_high_Dp=np.array(ind_comp_high_Dp, dtype=int)
phen_profile_Dp_phen_deviation_subtraction=phen_profile_Dp_phen_deviation_subtraction[ind_comp_high_Dp]
ind_sorted=np.argsort(phen_profile_Dp_phen_deviation_subtraction)[::-1]
phen_profile_Dp_phen_deviation_subtraction=phen_profile_Dp_phen_deviation_subtraction[ind_sorted]
ind_comp_high_Dp=ind_comp_high_Dp[ind_sorted]

# for i in range(3):
#     h_subset_matrix=H[ind_comp_high_Dp[i], :]
#     print(ind_comp_high_Dp[i])
#     ind_mean_h_subset_group=np.argsort(h_subset_matrix)[::-1]
#     phen_sorted_mean_h_subset=phen[ind_mean_h_subset_group]
#     h_subset_matrix=h_subset_matrix[ind_mean_h_subset_group]
    
#     #Save dataframe
#     df_specific=pd.DataFrame()
#     df_specific['phen']=phen_sorted_mean_h_subset
#     df_specific['h']=h_subset_matrix
    
#     df_specific.to_csv(path_save_data+'phen_associted_with_comp_%d_substrac_dp_vs_P_dev.csv' %ind_comp_high_Dp[i], sep='\t')
#     # np.savetxt(path_save_data+'phen_associted_with_most_different_comp_substraction_dp_vs_phen_deviation.txt', phen_sorted_mean_h_subset, fmt='%s')

h_subset_matrix=H[87, :]
ind_mean_h_subset_group=np.argsort(h_subset_matrix)[::-1]
phen_sorted_mean_h_subset=phen[ind_mean_h_subset_group]
h_subset_matrix=h_subset_matrix[ind_mean_h_subset_group]

#Save dataframe
df_specific=pd.DataFrame()
df_specific['phen']=phen_sorted_mean_h_subset
df_specific['h']=h_subset_matrix

df_specific.to_csv(path_save_data+'phen_associted_with_comp_87_substrac_dp_vs_P_dev.csv', sep='\t')
# np.savetxt(path_save_data+'phen_associted_with_most_different_comp_substraction_dp_vs_phen_deviation.txt', phen_sorted_mean_h_subset, fmt='%s')


#5.2.) Dev space
dev_profile_dP_dev_deviation = spatial.distance.euclidean(average_dev_prof_dp, average_dev_prof_high_perc)
dev_profile_Dp_phen_deviation = spatial.distance.euclidean(average_dev_prof_dp, average_dev_prof_low_perc)

dev_profile_pearson_dP_dev_deviation = pearsonr(average_dev_prof_dp, average_dev_prof_high_perc)
dev_profile_pearson_Dp_phen_deviation = pearsonr(average_dev_prof_dp, average_dev_prof_low_perc)


#5.3.) We are going to plot the scatter of the NMF phen for each group and deviation (the typical profile)
plt.figure(figsize=(4,3), dpi=600)
plt.plot(mean_W_dp, mean_W_dp, color='deepskyblue', lw=1, label='D-P')
plt.scatter(mean_W_dp, mean_W_high_perc, s=8, color='deeppink', alpha=0.5, label='d-P')
plt.legend(markerscale=3, fontsize=13)
plt.xlabel('<D-P NMF phenotypes>', fontsize=14)
plt.ylabel('<d-P NMF phenotypes>', fontsize=14)
plt.savefig(path_save_data+'typical_phen_prof_values_D_dev.png', dpi=600, bbox_inches='tight')
plt.show()

plt.figure(figsize=(4,3), dpi=600)
plt.plot(mean_W_dp, mean_W_dp, color='deepskyblue', lw=1, label='D-P')
plt.scatter(mean_W_dp, mean_W_low_perc, s=8, color='blueviolet', alpha=0.5, label='D-p')
plt.legend(markerscale=3, fontsize=13)
plt.xlabel('<D-P NMF phenotypes>', fontsize=14)
plt.ylabel('<D-p NMF phenotypes>', fontsize=14)
plt.savefig(path_save_data+'typical_phen_prof_values_P_dev.png', dpi=600, bbox_inches='tight')
plt.show()


#We find an interesting component to analyse:

ind_interest=np.where(mean_W_high_perc>0.1)[0]

h_subset_matrix=H[int(ind_interest), :]
ind_mean_h_subset_group=np.argsort(h_subset_matrix)[::-1]
phen_sorted_mean_h_subset=phen[ind_mean_h_subset_group]
h_subset_matrix=h_subset_matrix[ind_mean_h_subset_group]

#Save dataframe
df_specific=pd.DataFrame()
df_specific['phen']=phen_sorted_mean_h_subset
df_specific['h']=h_subset_matrix

df_specific.to_csv(path_save_data+'phen_associted_with_comp_%d_substrac_dp_vs_D_dev.csv' %int(ind_interest), sep='\t')
# np.savetxt(path_save_data+'phen_associted_with_most_different_comp_substraction_dp_vs_phen_deviation.txt', phen_sorted_mean_h_subset, fmt='%s')


#5.4.) We are going to plot the scatter of the frac of cells per time for each group and deviation (the typical profile)
plt.figure(figsize=(4,3), dpi=600)
plt.plot(average_dev_prof_dp, average_dev_prof_dp, color='deepskyblue', lw=1, label='D-P')
plt.scatter(average_dev_prof_dp, average_dev_prof_high_perc, s=8, color='deeppink', alpha=0.5, label='d-P')
plt.legend(markerscale=3, fontsize=13)
plt.xlabel('<D-P fraction of cells>', fontsize=14)
plt.ylabel('<d-P fraction of cells>', fontsize=14)
plt.savefig(path_save_data+'typical_dev_prof_values_D_dev.png', dpi=600, bbox_inches='tight')
plt.show()

plt.figure(figsize=(4,3), dpi=600)
plt.plot(average_dev_prof_dp, average_dev_prof_dp, color='deepskyblue', lw=1, label='D-P')
plt.scatter(average_dev_prof_dp, average_dev_prof_low_perc, s=8, color='blueviolet', alpha=0.5, label='D-p')
plt.legend(markerscale=3, fontsize=13)
plt.xlabel('<D-P fraction of cells>', fontsize=14)
plt.ylabel('<D-p fraction of cells>', fontsize=14)
plt.savefig(path_save_data+'typical_dev_prof_values_P_dev.png', dpi=600, bbox_inches='tight')
plt.show()


#6.) D-P rule and pleiotropy
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

sns.kdeplot(pleio_Dp_dev_P, color='blueviolet', fill=False, lw=1.5,  label='(D-p) - P deviation')
median_Dp_dev_P = np.median(pleio_Dp_dev_P)
plt.axvline(median_Dp_dev_P, color='blueviolet', linestyle='--', linewidth=1)

sns.kdeplot(pleio_dP_Dev_D, color='deeppink', fill=False,lw=1.5, label='(d-P) - D deviation')
median_dP_Dev_D = np.median(pleio_dP_Dev_D)
plt.axvline(median_dP_Dev_D, color='deeppink', linestyle='--', linewidth=1)

sns.kdeplot(pleio_DP_rule, color='deepskyblue', fill=False, lw=1.5, label='D-P rule')
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


#violin plot
colors=['deepskyblue', 'blueviolet', 'deeppink']

pos=[0, 1, 2]
label=['D-P', 'D-p', 'd-P']
plt.figure(figsize=(5, 5), dpi=600)
sns.violinplot([pleio_DP_rule, pleio_Dp_dev_P, pleio_dP_Dev_D], palette=colors, inner='quartile', edgecolor='white', linewidth=1.5, cut=0)
plt.xticks(pos, label, fontsize=20, fontweight='bold')
plt.yticks(fontsize=18)
plt.grid(False)
plt.ylabel('Pleiotropy', fontsize=20, fontweight='bold')
plt.savefig(path_save_data+'pleiotropy_dev.png', dpi=600, bbox_inches='tight')
plt.show()


