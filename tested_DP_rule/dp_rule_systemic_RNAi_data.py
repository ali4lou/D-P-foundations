# -*- coding: utf-8 -*-
"""
D-P rule using just RNAi phen
"""


import obonet
import numpy as np
import os
import networkx as nx
import matplotlib.pyplot as plt 
import pandas as pd
import matplotlib as mpl
from scipy.cluster.hierarchy import dendrogram
import great_library_phenotypes as glp
from matplotlib import cm 
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import MDS
from scipy.stats import pearsonr
from sklearn.decomposition import NMF
from scipy.stats import fisher_exact, false_discovery_control
from scipy.stats import kstest



path_save_data_real='PATH_TO_SAVE_YOUR_DATA'

#1.1.) We read commmon genes
f=open(path_save_data_real+'genes_id.txt', 'r')
txt = f.read()
genes = txt.split('\n')
del txt, f
genes=np.delete(genes, len(genes)-1)
genes=np.array(genes)

#1.2.) We read commmon phenotypes
f=open(path_save_data_real+'phenotypes_description.txt', 'r')
txt = f.read()
phen = txt.split('\n')
del txt, f
phen=np.delete(phen, len(phen)-1)
phen=np.array(phen)

#1.3.) We read phenotype matrices
phen_matrix=np.loadtxt(path_save_data_real+'gene_phen_matrix.txt')
W=np.loadtxt(path_save_data_real+'W.txt')
H=np.loadtxt(path_save_data_real+'H.txt')

#1.4.) We read NMF pleio score
f=open(path_save_data_real+'nmf_pleiotropy.txt', 'r')
txt = f.read()
pleio_score_nnmf = txt.split('\n')
del txt, f
pleio_score_nnmf=np.delete(pleio_score_nnmf, len(pleio_score_nnmf)-1)
pleio_score_nnmf=np.array(pleio_score_nnmf, dtype='f')


#1.5.) Most pleio genes
f=open(path_save_data_real+'pleio_genes.txt', 'r')
txt = f.read()
pleio_genes = txt.split('\n')
del txt, f
pleio_genes=np.delete(pleio_genes, len(pleio_genes)-1)
pleio_genes=np.array(pleio_genes, dtype=str)



#1) ANALYSIS OF WORM PHENOTYPE ONTOLOGY (WPO)

#We read the data sheet with the Worm Phenotype Ontology
url=path_save_data_real+'phenotype_ontology.WS290.obo'
graph = obonet.read_obo(url)

#create mappings
id_to_name = {id_: data.get('name') for id_, data in graph.nodes(data=True)}
name_to_id = {data['name']: id_ for id_, data in graph.nodes(data=True) if 'name' in data}
phen_description = {data['def']: id_ for id_, data in graph.nodes(data=True) if 'def' in data}

del url


#We keep nodes -> id phenotype and name phenotype
phenotypes=glp.convert_dictionary_to_array(id_to_name)
phen_description=glp.convert_dictionary_to_array(phen_description)

#We find all the possible paths for each phenotype
pathlist=[]
for i in range(len(phenotypes)):
    start=id_to_name[phenotypes[i][0]]
    inner_path=glp.find_path(graph, start, name_to_id)
    pathlist.append(inner_path)
del inner_path, start
    




#2.) GENE-PHENOTYPE ASSOCIATION ANALYSIS 

#2.1) We read the gene-phenotype association dataset (version: WS290)
f=open(path_save_data_real+'phenotype_association.WS290.wb.txt', "r")
dat = f.read()
raw_association = dat.split('\n')
del dat, f

line=raw_association[10].split("\t")
matrix_genes=np.empty ((len(raw_association), len(line)), dtype=object) 
for i in range(len(raw_association)):
    line=raw_association[i].split('\t') 
    for j in range(len(line)): 
        matrix_genes[i][j] = line[j] 
del raw_association, line

#2.2.) We eliminate all the gene-phenotype associations with a 'NOT' label in column 3
matrix_genes=np.delete(matrix_genes, matrix_genes[:, 0]!= "WB", axis=0)

gene_id=matrix_genes[:, 1]
print(len(gene_id))

ind_n_genes_not_association=np.where(matrix_genes[:, 3]== "NOT")[0]
genes_not_association=gene_id[ind_n_genes_not_association]

print('not ass:', len(genes_not_association), len(genes_not_association)/len(gene_id))

genes_not_ass_unique, n_times_genes_not_ass=np.unique(genes_not_association, return_counts=True)
print('unique genes not ass:', len(genes_not_ass_unique))

matrix_genes=np.delete(matrix_genes, matrix_genes[:, 3]== "NOT", axis=0)

gene_id=matrix_genes[:, 1]
print('yes ass:', len(gene_id))
print('unique yes ass:', len(np.unique(gene_id)))

gene_name=matrix_genes[:, 2]
gene_phenotype=matrix_genes[:, 4]



#2.3.) Number of associations from mutations (genetic variation) or from RNAi
#2.3.1) We check that all the associations are Var type or RNAi type

#In gene_exp and gene_exp2 we keep the associations type
gene_exp=matrix_genes[:, 5]
gene_exp2=matrix_genes[:, 7]
gene_exp_unique=np.unique(gene_exp)
gene_exp2_unique=np.unique(gene_exp2)

count=0
for i in range(len(gene_exp_unique)):
    if 'Var' in gene_exp_unique[i]:
        count=count+1
    if 'REF' in gene_exp_unique[i]:
        count=count+1
        
print(len(gene_exp_unique), count)

#2.3.2) Analysis of association types
n_var=0
n_RNA=0
gene_var_ass=[]
phen_var_ass=[]
gene_RNA_ass=[]
phen_RNA_ass=[]
df_gene_paper=pd.DataFrame()
gene_check=[]
paper=[]
for i in range(len(gene_exp)):
    if 'Var' in gene_exp[i]:
        n_var=n_var+1
        gene_var_ass.append(gene_id[i])
        phen_var_ass.append(gene_phenotype[i])
    if 'Var' in gene_exp2[i]:
        gene_var_ass.append(gene_id[i])
        phen_var_ass.append(gene_phenotype[i])
        n_var=n_var+1
    if 'RNA' in gene_exp2[i]:
        n_RNA=n_RNA+1
        gene_RNA_ass.append(gene_id[i])
        phen_RNA_ass.append(gene_phenotype[i])
        gene_check.append(gene_id[i])
        paper.append(gene_exp[i])
        
#figure
pos=[0, 1]
label=['Allele \n (variation)', 'RNAi']
colors = ['cornflowerblue', 'mediumpurple']

plt.figure(figsize=(5, 5), dpi=600)
# barras=['sin clasificar', 'sin tipo - con lin', 'sin lin - con tipo', 'clasificadas', 'total']
cuenta=(n_var, n_RNA)
plt.bar(pos, cuenta, color=colors)
plt.xticks(pos, label, fontsize=25, fontweight='bold', rotation=0)
plt.yticks(fontsize=20)
plt.ylabel("# associations", fontsize=25, fontweight='bold')
plt.grid(False)
plt.savefig(path_save_data_real+'association_experiment_type.png', dpi=600, bbox_inches='tight')
plt.show()
    

#2.3.3.) Find the genes with both associations
df_gene_phen_RNAi = pd.DataFrame({'gene': gene_RNA_ass, 'phen': phen_RNA_ass})
df_gene_phen_var= pd.DataFrame({'gene': gene_var_ass, 'phen': phen_var_ass})


#3.) UNIQUE ASSOCIATIONS: associations that relate a gene with a different phenotype
#Distribution of the number of associations per gene
#We find the unique genes in the gene-phen association data sheet 
genes_id_type=np.array(list(set(gene_id)))
gene_name_type=np.array(list(set(gene_name)))


#but those assotiations are redundant
#In this loop we find the associatied phenotypes with each unique gene
phen_type=[]
inner_list=[]
#phenotipos corresponden a genes_id_tipo orden
for i in range(len(genes_id_type)):
    ind_genes=np.where(gene_id==genes_id_type[i])
    ind_genes=ind_genes[0]
    inner_list=[]
    for j in range(len(ind_genes)):
        index=ind_genes[j]
        inner_list.append(gene_phenotype[index])
    phen_type.append(inner_list)
del ind_genes, inner_list, index



#real number of assotiations (different assotiations-different phen)
n_times_unique_phen_ass_ini=[]
for i in range(len(phen_type)):
    control_phen=np.array(phen_type[i])
    unique_phen_ass_ini=np.unique(control_phen)
    n_times_unique_phen_ass_ini.append(len(unique_phen_ass_ini))
    

n_times_genes_ass=np.array(n_times_unique_phen_ass_ini)

#3.1.)  FEW vs. LOT ASSOTIATIONS - extremes of the distribution
ind_few_ass=np.where(n_times_genes_ass<=np.percentile(n_times_genes_ass, 25))[0]
ind_lot_ass=np.where(n_times_genes_ass>=np.percentile(n_times_genes_ass, 75))[0]
genes_few_ass=genes_id_type[ind_few_ass]
genes_lot_ass=genes_id_type[ind_lot_ass]
n_ass_lot=n_times_genes_ass[ind_lot_ass]
n_ass_few=n_times_genes_ass[ind_few_ass]

#figure
plt.figure(figsize=(4, 3),dpi=600)
plt.hist(n_times_genes_ass, bins=100, color='lightseagreen', log=True)
plt.ylabel('# genes', fontsize=14, fontweight='bold')
plt.xlabel('# associations\ndifferent initial phenotypes', fontsize=14, fontweight='bold')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(False)
plt.axvline(x=np.percentile(n_times_genes_ass, 25), color='royalblue', linestyle='--', lw=1)
plt.axvline(x=np.percentile(n_times_genes_ass, 75), color='tomato', linestyle='--', lw=1)
plt.savefig(path_save_data_real+'association_distribution_per_gene_unique_phen.png', dpi=600, bbox_inches='tight')
plt.show()



np.savetxt(path_save_data_real+'genes_few_ass.txt', genes_few_ass, fmt='%s')  
np.savetxt(path_save_data_real+'genes_lot_ass.txt', genes_lot_ass, fmt='%s')  

#3.1.1.) figure n_assotiations
data=[n_ass_few, n_ass_lot]
fig, ax = plt.subplots(figsize=(4,3), dpi=600)
box = ax.boxplot(data, labels=["< P25", "> P75"],  widths=0.45, patch_artist=True, 
                 flierprops=dict(marker='o', color='gray', markersize=0.5), 
                 boxprops=dict(edgecolor="none"))
colors = ["royalblue", "tomato"]
for patch, color in zip(box["boxes"], colors):
    patch.set_facecolor(color)

plt.setp(box["medians"], color="black", linewidth=1)
plt.setp(box["whiskers"], color="black", linestyle="--", linewidth=0.5)
plt.setp(box["caps"], color="black", linewidth=0.5)

plt.ylabel("# assotiations", fontsize=15)
plt.xlabel("Assotiations", fontsize=15)
plt.xticks(fontsize=18, fontweight='bold')
plt.yticks(fontsize=13)

plt.savefig(path_save_data_real+'n_ass_lot_few_genespng', dpi=600, bbox_inches='tight')
plt.show()


#3.1.2.) Genes with lot and few assotiations pleiotropy
g, ind_pleio_lot_ass, i=np.intersect1d(genes, genes_lot_ass, return_indices=True)
g, ind_pleio_few_ass, i=np.intersect1d(genes, genes_few_ass, return_indices=True)

pleio_few_ass=pleio_score_nnmf[ind_pleio_few_ass]
pleio_lot_ass=pleio_score_nnmf[ind_pleio_lot_ass]


#figure boxplot
data=[pleio_few_ass, pleio_lot_ass]
fig, ax = plt.subplots(figsize=(4,3), dpi=600)
box = ax.boxplot(data, labels=["< P25", "> P75"],  widths=0.45, patch_artist=True, 
                 flierprops=dict(marker='o', color='gray', markersize=0.5), 
                 boxprops=dict(edgecolor="none"))
colors = ["royalblue", "tomato"]
for patch, color in zip(box["boxes"], colors):
    patch.set_facecolor(color)
plt.setp(box["medians"], color="black", linewidth=1)
plt.setp(box["whiskers"], color="black", linestyle="--", linewidth=0.5)
plt.setp(box["caps"], color="black", linewidth=0.5)
plt.ylabel("NMF Pleiotropy", fontsize=15)
plt.xlabel("Assotiations", fontsize=15)
plt.xticks(fontsize=18, fontweight='bold')
plt.yticks(fontsize=13)
plt.savefig(path_save_data_real+'pleiotropy_genes_lot_few_ass.png', dpi=600, bbox_inches='tight')
plt.show()


print(len(n_ass_few), len(pleio_few_ass))
print(len(n_ass_lot), len(pleio_lot_ass))


print('mean lot vs few:', np.mean(pleio_lot_ass), np.mean(pleio_few_ass))
print('median lot vs few:', np.median(pleio_lot_ass), np.median(pleio_few_ass))

from scipy import stats
ks, pvalue = stats.ks_2samp(pleio_lot_ass, pleio_few_ass)
print('lot vs few ass: pleio', ks, pvalue)

ks, pvalue = stats.ks_2samp(n_ass_lot, n_ass_few)
print('lot vs few ass: n_ass', ks, pvalue)



#4. GENES and NUMBER OF PUBLICATIONS
publication=[]
gene_with_paper=[]
for i in range(len(gene_exp)):
    if 'WBPaper' in gene_exp[i]:
        publication.append(gene_exp[i])
        gene_with_paper.append(gene_id[i])
    if 'WBPaper' in gene_exp2[i]:
        publication.append(gene_exp2[i])
        gene_with_paper.append(gene_id[i])
        
publication=np.array(publication)

df_all_gene_publication=pd.DataFrame({'gene': gene_with_paper, 'publication': publication})

conteos_publi = df_all_gene_publication.value_counts().reset_index(name='counts')
gene_and_publication, n_publication_per_gene=np.unique(np.array(conteos_publi['gene']), return_counts=True)      




print('n curators (non publications):', len(gene_id)-len(gene_with_paper), (len(gene_id)-len(gene_with_paper))/len(gene_id))

#4.1.) Comparison betweeen publications per gene and pleiotropy
g_common, ind_pleio_publication, ind_n_publi=np.intersect1d(genes, gene_and_publication, return_indices=True)

pleio_gene_and_publication=pleio_score_nnmf[ind_pleio_publication]
n_publication_per_gene_final=n_publication_per_gene[ind_n_publi]

#n associations
g2, ind_ass_per_pub, i=np.intersect1d(genes_id_type, g_common, return_indices=True)
n_ass_per_gene_final=n_times_genes_ass[ind_ass_per_pub]

pearsonr(n_publication_per_gene_final, pleio_gene_and_publication)

p95=np.percentile(n_publication_per_gene_final, 95)

print(p95)
ind_gene_most_pub=np.where(n_publication_per_gene_final>50)[0]
gene_most_pub=g_common[ind_gene_most_pub]

len(gene_most_pub)/len(n_publication_per_gene_final)

pleio_most_publi=pleio_gene_and_publication[ind_gene_most_pub]
n_publication_most_pub_genes=n_publication_per_gene_final[ind_gene_most_pub]
print(gene_most_pub, pleio_most_publi, n_publication_most_pub_genes)

g2, ind_ass_most_pub, i=np.intersect1d(genes_id_type, gene_most_pub, return_indices=True)
n_ass_gene_most_pub=n_times_genes_ass[ind_ass_most_pub]
print('associations:', n_ass_gene_most_pub)


np.savetxt(path_save_data_real+'gene_most_pub.txt', gene_most_pub, fmt='%s')  

c_genes=np.intersect1d(genes, gene_most_pub)
print(len(c_genes), len(gene_most_pub))
most_pleio_genes=np.intersect1d(pleio_genes, gene_most_pub)
print(len(most_pleio_genes))

print('from our genes with the highest # of publications', len(most_pleio_genes)/len(gene_most_pub), 'are within most pleiotropic genes')

#figure
#log plot
x=n_publication_per_gene_final
y=pleio_gene_and_publication
# Transformación log-log
log_x = np.log10(x)
log_y = np.log10(y)
# Ajuste lineal
coef = np.polyfit(log_x, log_y, 1)  # coef[0] = pendiente (b), coef[1] = intercepto log(a)
b, log_a = coef
a = 10**log_a

print(f"Ajuste: y = {a:.3f} * x^{b:.3f}")

# Crear línea ajustada
x_fit = np.logspace(np.log10(x.min()), np.log10(x.max()), 100)
y_fit = a * x_fit**b

import matplotlib.colors as mcolors
# Normalización de los valores para el colormap
norm = mcolors.Normalize(vmin=n_ass_per_gene_final.min(), vmax=n_ass_per_gene_final.max())
# Selección del colormap tipo rainbow
colormap = cm.get_cmap('gnuplot_r')
# Asignar colores según los valores normalizados
# colores = colormap(norm(n_ass_per_gene_final))


#FIGURE
fig, ax=plt.subplots(figsize=(4, 3), dpi=600)
scatter=ax.scatter(n_publication_per_gene_final, pleio_gene_and_publication, s=4, alpha=0.7, c=n_ass_per_gene_final, cmap=colormap, norm=norm,)
ax.set_xlabel('# papers', fontsize=16, fontweight='bold')
ax.set_ylabel('NMF Pleiotropy', fontsize=16, fontweight='bold')
ax.tick_params(axis='x', labelsize=14, rotation=90)
ax.tick_params(axis='y', labelsize=14)

ax.plot(x_fit, y_fit, '-', label=f'y = {a:.2f} x^{b:.2f}', lw=1, color='lightblue')
ax.set_yscale('log')
ax.set_xscale('log')

cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('# associations', fontsize=11)

ax.legend()
plt.savefig(path_save_data_real + 'n_papers_vs_pleiotropy.png', dpi=600, bbox_inches='tight')
plt.show()


#figure mean and std
n_unique_papers=np.unique(n_publication_per_gene_final)
mean_pleio_per_unique_paper=np.zeros(len(n_unique_papers))
std_pleio_per_unique_paper=np.zeros(len(n_unique_papers))
for i in range(len(n_unique_papers)):
    ind_p=np.where(n_publication_per_gene_final==n_unique_papers[i])[0]
    mean_pleio_per_unique_paper[i]=np.mean(pleio_gene_and_publication[ind_p])
    std_pleio_per_unique_paper[i]=np.std(pleio_gene_and_publication[ind_p])


plt.figure(figsize=(4, 3), dpi=600)
plt.plot(n_unique_papers, mean_pleio_per_unique_paper, c='blueviolet', label='mean', marker='o', lw=0.7, markersize=2.3)
plt.fill_between(
    n_unique_papers,
    mean_pleio_per_unique_paper - std_pleio_per_unique_paper,
    mean_pleio_per_unique_paper + std_pleio_per_unique_paper,
    color='blueviolet',
    alpha=0.3,
    label='std dev')
plt.xlabel('# papers', fontsize=18, fontweight='bold')
plt.ylabel('NMF pleiotropy', fontsize=18, fontweight='bold')
plt.xticks(fontsize=16, rotation=90)
plt.yticks(fontsize=16)
plt.legend()
plt.yscale('log')
plt.xscale('log')
plt.savefig(path_save_data_real+'n_papers_vs_pleiotropy_mean_std.png', dpi=600, bbox_inches='tight')
plt.show()



#4.2.) Comparison of #phen original matrix and #phen W and number of papers
#or: omparison of #associations original and #phen W and number of papers
#have W matrix, phen_matrix, and number of publications and number of associaions
g_common, ind_use, ind_n_publi=np.intersect1d(genes, gene_and_publication, return_indices=True)

#n publications
n_publication_per_gene_final=n_publication_per_gene[ind_n_publi]

#n associations
g2, ind_ass_per_pub, i=np.intersect1d(genes_id_type, g_common, return_indices=True)
n_ass_per_gene_final=n_times_genes_ass[ind_ass_per_pub]

#n different phen in ortiginal matrix
submatrix=phen_matrix[ind_use, :]
n_dif_phen_original=np.sum(submatrix, axis=1)

#n different W components
sub_W=W[ind_use, :]
n_dif_phen_W=np.zeros(len(g_common))
for i in range(len(g_common)):
    n_dif_phen_W[i]=len(np.where(sub_W[i, :]>0)[0])
    
    
#scatter plot
x=n_dif_phen_original
y=n_dif_phen_W

# Ajuste lineal
coef = np.polyfit(x, y, 1)  # coef[0] = pendiente (b), coef[1] = intercepto log(a)
b, a = coef

print(f"Ajuste: y = {a:.3f} + {b:.3f}x")

# Crear línea ajustada
x_fit = np.linspace(x.min(), x.max(), 100)
y_fit = a + b*x_fit

#color
norm = mcolors.Normalize(vmin=n_publication_per_gene_final.min(), vmax=n_publication_per_gene_final.max())
colormap = cm.get_cmap('magma_r')

#figure
fig, ax=plt.subplots(figsize=(4, 3), dpi=600)
scatter=ax.scatter(n_dif_phen_original, n_dif_phen_W, s=4, alpha=0.8, c=n_publication_per_gene_final, cmap=colormap, norm=norm,)
ax.set_xlabel('# phen original', fontsize=16, fontweight='bold')
ax.set_ylabel('# NMF phen', fontsize=16, fontweight='bold')
ax.tick_params(axis='x', labelsize=14, rotation=90)
ax.tick_params(axis='y', labelsize=14)
# ax.set_yscale('log')
# ax.set_xscale('log')
ax.plot(x_fit, y_fit, '-', label=f'y = {a:.2f} + {b:.2f}x', lw=1, color='lightblue')
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('# publications', fontsize=11)
ax.legend() 
plt.savefig(path_save_data_real + 'original_phen_vs_W_phen.png', dpi=600, bbox_inches='tight')
plt.show()

#publications vs associations
#regression
x=n_publication_per_gene_final
y=n_ass_per_gene_final

# Ajuste lineal
coef = np.polyfit(x, y, 1)  # coef[0] = pendiente (b), coef[1] = intercepto log(a)
b, a = coef

print(f"Ajuste: y = {a:.3f} + {b:.3f}x")

# Crear línea ajustada
x_fit = np.linspace(x.min(), x.max(), 100)
y_fit = a + b*x_fit
#figure
fig, ax=plt.subplots(figsize=(3.5, 4), dpi=600)
scatter=ax.scatter(n_publication_per_gene_final, n_ass_per_gene_final, s=5, alpha=0.5, color='blueviolet')
ax.set_xlabel('# papers', fontsize=18, fontweight='bold')
ax.set_ylabel('# associations', fontsize=18, fontweight='bold')
ax.tick_params(axis='x', labelsize=16, rotation=90)
ax.tick_params(axis='y', labelsize=16)
ax.plot(x_fit, y_fit, '-', label=f'y = {a:.2f} + {b:.2f}x', lw=1, color='orange')
plt.legend()
plt.savefig(path_save_data_real + 'papers_vs_associations.png', dpi=600, bbox_inches='tight')
plt.show()


#=======================================================================================================
#5.) Genes classified by RNAi, variation and both associations type

gene_RNA_ass_unique=np.unique(gene_RNA_ass)
gene_var_ass_unique=np.unique(gene_var_ass)

gene_both_RNA_var_ass=np.intersect1d(gene_RNA_ass_unique, gene_var_ass_unique)

genes_only_var = list(set(gene_var_ass_unique) - set(gene_both_RNA_var_ass))
genes_only_RNA = list(set(gene_RNA_ass_unique) - set(gene_both_RNA_var_ass))
genes_only_var=np.array(genes_only_var)
genes_only_RNA=np.array(genes_only_RNA)

print(len(gene_RNA_ass_unique), len(gene_var_ass_unique), len(gene_both_RNA_var_ass), len(genes_only_var), len(genes_only_RNA))


#5.1.) We find the genes that are found in one assocition type: RNAi or variation
gene_only_RNA_ass=[]
gene_only_var_ass=[]

for i in range(len(gene_both_RNA_var_ass)):
    if len(np.where(gene_RNA_ass_unique==gene_both_RNA_var_ass[i])[0]):
        gene_only_RNA_ass.append(gene_both_RNA_var_ass[i])
    if len(np.where(gene_var_ass_unique==gene_both_RNA_var_ass[i])[0]):
        gene_only_var_ass.append(gene_both_RNA_var_ass[i])

gene_only_var_ass=np.array(gene_only_var_ass)
gene_RNA_ass_unique=np.array(gene_RNA_ass_unique)



#we are goind to compare the genes with just RNAi, just variations and both variations and RNAi
g, ind_ass, i=np.intersect1d(genes_id_type, genes, return_indices=True)
n_ass_per_gene=n_times_genes_ass[ind_ass]

total_genes_both_analyse, ind_use, g=np.intersect1d(genes, gene_both_RNA_var_ass, return_indices=True)
total_genes_both_analyse_n_ass=n_ass_per_gene[ind_use]
pleio_both_analyse=pleio_score_nnmf[ind_use]
print(len(total_genes_both_analyse))

genes_only_var_analyse, ind_use, g=np.intersect1d(genes, genes_only_var, return_indices=True)
genes_only_var_analyse_n_ass=n_ass_per_gene[ind_use]
pleio_only_var_analyse=pleio_score_nnmf[ind_use]
print(len(genes_only_var_analyse))

genes_only_RNA_analyse, ind_use, g=np.intersect1d(genes, genes_only_RNA, return_indices=True)
genes_only_RNA_analyse_n_ass=n_ass_per_gene[ind_use]
pleio_only_RNA_analyse=pleio_score_nnmf[ind_use]
print(len(genes_only_RNA_analyse))


#figure n_assotiations
data=[total_genes_both_analyse_n_ass, genes_only_RNA_analyse_n_ass, genes_only_var_analyse_n_ass]
fig, ax = plt.subplots(figsize=(4,3), dpi=600)
box = ax.boxplot(data, labels=["both", 'RNAi', "Variation"],  widths=0.45, patch_artist=True, 
                 flierprops=dict(marker='o', color='gray', markersize=0.5), 
                 boxprops=dict(edgecolor="none"))
colors = ["royalblue", "tomato", 'mediumturquoise']
for patch, color in zip(box["boxes"], colors):
    patch.set_facecolor(color)

plt.setp(box["medians"], color="black", linewidth=1)
plt.setp(box["whiskers"], color="black", linestyle="--", linewidth=0.5)
plt.setp(box["caps"], color="black", linewidth=0.5)

plt.ylabel("# assotiations", fontsize=15)
plt.xlabel("Assotiation type", fontsize=15)
plt.xticks(fontsize=14, fontweight='bold')
plt.yticks(fontsize=13)

plt.savefig(path_save_data_real+'n_ass_RNAi_variation_genes.png', dpi=600, bbox_inches='tight')
plt.show()


#we measure the pleiotropy
#NMF pleio
#figure boxplot
data=[pleio_both_analyse, pleio_only_RNA_analyse, pleio_only_var_analyse]
fig, ax = plt.subplots(figsize=(4,3), dpi=600)
box = ax.boxplot(data, labels=["both", 'RNAi', "Variation"],  widths=0.45, patch_artist=True, 
                 flierprops=dict(marker='o', color='gray', markersize=0.5), 
                 boxprops=dict(edgecolor="none"))
colors = ["royalblue", "tomato", 'mediumturquoise']
for patch, color in zip(box["boxes"], colors):
    patch.set_facecolor(color)

plt.setp(box["medians"], color="black", linewidth=1)
plt.setp(box["whiskers"], color="black", linestyle="--", linewidth=0.5)
plt.setp(box["caps"], color="black", linewidth=0.5)

plt.ylabel("NMF Pleiotropy", fontsize=15)
plt.xlabel("Assotiation type", fontsize=15)
plt.xticks(fontsize=14, fontweight='bold')
plt.yticks(fontsize=13)
plt.savefig(path_save_data_real+'pleiotropy_genes_Association_type.png', dpi=600, bbox_inches='tight')
plt.show()


#5.2.) RNAi associations and variation associations per gene (COMPARISON)
conteos_phen_RNAi = df_gene_phen_RNAi.value_counts().reset_index(name='counts')
unique_gene_RNAi, n_times_unique_gene_RNAi=np.unique(np.array(conteos_phen_RNAi['gene']), return_counts=True)      

conteos_phen_var = df_gene_phen_var.value_counts().reset_index(name='counts')
unique_gene_var, n_times_unique_gene_var=np.unique(np.array(conteos_phen_var['gene']), return_counts=True)      

#We search the genes with both RNAi and variation associations 
g, ind_use_RNAi, i=np.intersect1d(unique_gene_RNAi, total_genes_both_analyse, return_indices=True)
n_times_both_variations_RNAi=n_times_unique_gene_RNAi[ind_use_RNAi]

g, ind_use_var, i=np.intersect1d(unique_gene_var, total_genes_both_analyse, return_indices=True)
n_times_both_variations_var=n_times_unique_gene_var[ind_use_var]

len(np.where(n_times_both_variations_RNAi==n_times_both_variations_var)[0])

x = np.linspace(0, 110, 100)
y=x
pearsonr(n_times_both_variations_var, n_times_both_variations_RNAi)
#figure
plt.figure(figsize=(4, 4), dpi=600)
plt.plot(x, y, label='y = x', color='deepskyblue')
plt.scatter(n_times_both_variations_var, n_times_both_variations_RNAi, s=3, alpha=0.7, color='blue')
plt.xlabel('# variations', fontsize=22, fontweight='bold')
plt.ylabel('# RNAi', fontsize=22, fontweight='bold')
plt.xticks(fontsize=18, rotation=90)
plt.yticks(fontsize=18)
plt.savefig(path_save_data_real+'n_variations_vs_n_RNA.png', dpi=600, bbox_inches='tight')
plt.show()


#6.) GENES ASSOCIATED IN EACH RNAi PAPER 
df_gene_paper = pd.DataFrame({'gene': gene_check, 'paper': paper})
# paper_unique_RNA, n_times_unique_paper=np.unique(paper, return_counts=True)           
conteos = df_gene_paper.value_counts().reset_index(name='counts')
paper_unique_real_gene, n_times_unique_paper=np.unique(np.array(conteos['paper']), return_counts=True)      

#figure
plt.figure(figsize=(4, 3),dpi=600)
plt.hist(n_times_unique_paper, bins=100, color='violet', log=True)
plt.ylabel('# RNAi papers', fontsize=14, fontweight='bold')
plt.xlabel('# genes', fontsize=14, fontweight='bold')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(False)
plt.axvline(x=np.median(n_times_unique_paper), color='indigo', linestyle='--', lw=1)
plt.savefig(path_save_data_real+'distribution_genes_RNAi.png', dpi=600, bbox_inches='tight')
plt.show()

#We just want to keep those associations that came from RNAi papers and from systematic analysis
#We keep the associations linked to a RNAi paper with at least 100 genes (~percentil 98)
ind_papers_to_keep=np.where(n_times_unique_paper>100)[0]
papers_to_keep=paper_unique_real_gene[ind_papers_to_keep]


#6.1.) We just keep the gene-phen associations of RNAi related with those papers
gene_ass_big_RNA=[]
phen_ass_big_RNA=[]
for i in range(len(gene_exp)):
    if 'RNA' in gene_exp2[i]:
        if gene_exp[i] in papers_to_keep:
            phen_ass_big_RNA.append(gene_phenotype[i])
            gene_ass_big_RNA.append(gene_id[i])
             
gene_ass_big_RNA_unique=np.unique(gene_ass_big_RNA)
phen_ass_big_RNA_unique=np.unique(phen_ass_big_RNA)


#6.2.) We create the new gene-phenotype associations matrix

final_gene_id=np.array(gene_ass_big_RNA)
final_gene_phenotype=np.array(phen_ass_big_RNA)

unique_genes=np.unique(final_gene_id)
unique_phen=np.unique(final_gene_phenotype)

      
matriz=np.zeros((len(unique_genes), len(phenotypes[:,0])))
for i in range(len(unique_genes)):
    ind_gen=np.where(final_gene_id==unique_genes[i])[0]
    for j in range(len(ind_gen)):
        busca_ind=np.where(phenotypes[:,0]==final_gene_phenotype[int(ind_gen[j])])
        busca_ind=int(busca_ind[0])
        #ahora buscamos todos los fenotypes de la jerarquia
        for n in range(len(pathlist[busca_ind])):
            for dentro in range(len(pathlist[busca_ind][n])):
                ind_matrix=np.where(phenotypes[:,0]==pathlist[busca_ind][n][dentro])
                ind_matrix=int(ind_matrix[0])
                # print(ind_matrix)
                matriz[i][ind_matrix]=1
                
sum_col = matriz.sum(axis=0)
ind_nematode=np.where(sum_col==len(unique_genes))[0]
print(phenotypes[ind_nematode,1])

ind_not_null = np.where(sum_col > 0)[0]

RNAi_matrix=matriz[:, ind_not_null]
def_RNAi_phen_id=phenotypes[ind_not_null, 0]
def_RNAi_phen=phenotypes[ind_not_null, 1]
#unique_genes and embryo matrix are the important variables here

sum_col = RNAi_matrix.sum(axis=0)
ind_nematode=np.where(sum_col==len(unique_genes))[0]
def_RNAi_phen[int(ind_nematode)]


del matriz



np.savetxt(path_save_data_real+"big_RNAi_gene_phen_matrix.txt", RNAi_matrix, delimiter="\t")

with open(path_save_data_real+'big_RNAi_phen.txt', 'w') as file:
    for item in def_RNAi_phen:
        file.write(f"{item}\n")

with open(path_save_data_real+'big_RNAi_genes.txt', 'w') as file:
    for item in unique_genes:
        file.write(f"{item}\n")




#6.3.) We read the gene list and the developmental matrix
dev_matrix=np.loadtxt(path_save_data_real+'dev_matrix_fraction_cells.txt')

f=open(path_save_data_real+'genes_id.txt', 'r')
txt = f.read()
genes_dev = txt.split('\n')
del txt, f
genes_dev=np.delete(genes_dev, len(genes_dev)-1)
genes_dev=np.array(genes_dev)


#6.4.) We search the common genes and we create the pairwise matrix
from scipy.spatial.distance import squareform, pdist

ind_genes_found=[]
common_genes=[]
ind_genes_phen=[]
for i in range(len(unique_genes)):
    ind_genes=np.where(genes_dev==unique_genes[i])[0]
    if len(ind_genes)>0:
        ind_genes_found.append(int(ind_genes))
        common_genes.append(unique_genes[i])
        ind_genes_phen.append(i)
ind_genes_found=np.array(ind_genes_found, dtype=int)
ind_genes_phen=np.array(ind_genes_phen, dtype=int)
common_genes=np.array(common_genes)

submatrix_dev=dev_matrix[ind_genes_found, :]

dist_dev=pdist(submatrix_dev, metric='cosine')
dist_dev_matrix=squareform(dist_dev)
sim_dev_matrix=1-dist_dev_matrix

del dist_dev_matrix


#6.5.) We create the matrix of RNAi phen final

good_matrix_embryo_common_genes=RNAi_matrix[ind_genes_phen, :]

# Initialize the NMF model with deterministic initialization using 'nndsvd'
n_components = 100  # The number of latent components (reduced dimensions)
model = NMF(n_components=n_components, init='nndsvd', random_state=42, max_iter=500)

# Fit the model to the data and perform the transformation
W = model.fit_transform(good_matrix_embryo_common_genes)  # The reduced representation of the data
H = model.components_  # The latent components (patterns)

# from sklearn.preprocessing import normalize
# W_normalized = normalize(W, norm='l2', axis=1)

#6.5.1.) We compute the pairwise similarities
dist_phen=pdist(W, metric='cosine')

dist_phen_matrix=squareform(dist_phen)
sim_phen_matrix=1-dist_phen_matrix

#6.5.2.) We compute the median
average_sim_dev_per_gen=np.median(sim_dev_matrix, axis=1)
average_sim_phen_per_gen=np.median(sim_phen_matrix, axis=1)

from scipy.stats import spearmanr
sim_phen=1-dist_phen
sim_dev=1-dist_dev

print('Pearson: average similaties per gene:', pearsonr(average_sim_dev_per_gen, average_sim_phen_per_gen))
print('Spearman: average similaties per gene:', spearmanr(average_sim_dev_per_gen, average_sim_phen_per_gen))

print('Pearson between pairwise sim:', pearsonr(sim_dev, sim_phen))
print('Spearman between pairwise sim:', spearmanr(sim_dev, sim_phen))


#rule controling sparsity

n_dif_phen_W=np.zeros(len(common_genes))
for i in range(len(common_genes)):
    n_dif_phen_W[i]=len(np.where(W[i, :]>0)[0])


from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression

def partial_spearman(sim1, sim2, sparsity):
    # Convertir a rangos
    r1 = pd.Series(sim1).rank()
    r2 = pd.Series(sim2).rank()
    rz = pd.Series(sparsity).rank()

    def residual(x, z):
        model = LinearRegression().fit(z.values.reshape(-1, 1), x.values.reshape(-1, 1))
        return x - model.predict(z.values.reshape(-1, 1)).flatten()

    r1_resid = residual(r1, rz)
    r2_resid = residual(r2, rz)

    rho, p_val = spearmanr(r1_resid, r2_resid)
    return rho, p_val

rho, p_val=partial_spearman(average_sim_dev_per_gen, average_sim_phen_per_gen, n_dif_phen_W)

print(rho, p_val)


#rule with function
import pingouin as pg
import pandas as pd

# Supongamos que tus datos están en estas variables:
# average_sim_dev_per_gen, average_sim_phen_per_gen, n_dif_phen_W

# Convertimos los datos a un DataFrame para usar pingouin
df = pd.DataFrame({
    'sim1': average_sim_dev_per_gen,
    'sim2': average_sim_phen_per_gen,
    'sparsity': n_dif_phen_W
})

# Calculamos la correlación parcial de Spearman controlando por 'sparsity'
result = pg.partial_corr(data=df, x='sim1', y='sim2', covar='sparsity', method='spearman')

print(result[['r', 'p-val']])
print(result)

result = pg.partial_corr(data=df, x='sim1', y='sim2', covar='sparsity')

print(result[['r', 'p-val']])
print(result)

df.pcorr().round(3)



#6.6.) D-P rule
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

# # figure
plt.figure(figsize=(5, 5), dpi=600)
plt.scatter(sw_dev, sw_phen, s=0.5, color='slateblue')
plt.xlabel('<sim-D>', fontsize=22, fontweight='bold')
plt.ylabel('<sim-P>', fontsize=22, fontweight='bold')
plt.xticks(fontsize=18, rotation=90)
plt.yticks(fontsize=18)
plt.savefig(path_save_data_real+'DP_rule_big_RNAi_W_norm.png', dpi=600, bbox_inches='tight')
plt.show()


#6.6.1.)percentiles
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
for i in range(len(common_genes)):
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
for i in range(len(common_genes)):
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
plt.savefig(path_save_data_real+'percentiles_DP_rule_big_RNAi_W_norm.png', dpi=600, bbox_inches='tight')
plt.show()





#6.7.) PLEIO CHECK
g, ind_pleio_nmf, i=np.intersect1d(genes, common_genes, return_indices=True)
pleio_nmf=pleio_score_nnmf[ind_pleio_nmf]

pleio_big_RNAi=np.sum(W, axis=1)




plt.hist(pleio_big_RNAi, bins=100)
plt.hist(pleio_nmf, alpha=0.5, bins=100)

ks, pvalue = stats.ks_2samp(pleio_nmf, pleio_big_RNAi)
print('old vs new ass: pleio', ks, pvalue)

plt.scatter(pleio_nmf, pleio_big_RNAi, s=0.5)

data=[pleio_nmf, pleio_big_RNAi]
fig, ax = plt.subplots(figsize=(4,3), dpi=600)
box = ax.boxplot(data, labels=["old", "new"],  widths=0.45, patch_artist=True, 
                 flierprops=dict(marker='o', color='gray', markersize=0.5), 
                 boxprops=dict(edgecolor="none"))
colors = ["royalblue", "tomato"]
for patch, color in zip(box["boxes"], colors):
    patch.set_facecolor(color)
plt.setp(box["medians"], color="black", linewidth=1)
plt.setp(box["whiskers"], color="black", linestyle="--", linewidth=0.5)
plt.setp(box["caps"], color="black", linewidth=0.5)
plt.ylabel("NMF Pleiotropy", fontsize=15)
plt.xticks(fontsize=18, fontweight='bold')
plt.yticks(fontsize=13)
plt.savefig(path_save_data_real+'pleiotropy_RNAi_old_new.png', dpi=600, bbox_inches='tight')
plt.show()


pearsonr(pleio_nmf, pleio_big_RNAi)



#6.7.1.) Scatter plot new vs. old pleiotropy
m, b = np.polyfit(pleio_nmf, pleio_big_RNAi, 1)

print(f'Pendiente (m): {m}')
print(f'Intercepto (b): {b}')

x_vals = np.array(pleio_nmf)
y_vals = m * x_vals + b

residuals = np.array(pleio_big_RNAi) - y_vals

mean=np.mean(residuals)
std=np.std(residuals)

ind_res=np.where(residuals>(mean+2*std))[0]
ind_res_neg=np.where(residuals<(mean-2*std))[0]

colores = np.full(len(residuals), 'blue', dtype=object)
for i in range(len(residuals)):
    ind=np.where(ind_res==i)[0]
    if len(ind)>0:
        colores[i]='orange'
    ind2=np.where(ind_res_neg==i)[0]
    if len(ind2)>0:
        colores[i]='red'
    
print(len(ind_res)/len(residuals))
print(len(ind_res_neg)/len(residuals))


#figure scatter plot
plt.figure(figsize=(4, 4), dpi=600)
plt.scatter(pleio_nmf, pleio_big_RNAi, s=3, alpha=0.5, c=colores)
plt.plot(x_vals, y_vals, color='deepskyblue', linewidth=1.2, label=f'y = {m:.2f}x + {b:.2f}')
plt.xlabel('old pleio', fontsize=22, fontweight='bold')
plt.ylabel('new pleio', fontsize=22, fontweight='bold')
plt.xticks(fontsize=18, rotation=90)
plt.yticks(fontsize=18)
plt.savefig(path_save_data_real+'old_pleio_vs_new_pleio.png', dpi=600, bbox_inches='tight')
plt.show()

#residual distrib
plt.figure(figsize=(4, 3),dpi=600)
plt.hist(residuals, bins=100, color='chocolate')
plt.ylabel('# genes', fontsize=14, fontweight='bold')
plt.xlabel('Residual value', fontsize=14, fontweight='bold')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(False)
plt.axvline(x=np.mean(residuals), color='darkred', linestyle='--', lw=1)
plt.savefig(path_save_data_real+'residual_distrib_old_vs_new_pleio.png', dpi=600, bbox_inches='tight')
plt.show()


#6.7.2.) Analysis of outliers (genes with high and low residuals)
genes_res_lower=common_genes[ind_res]
genes_res_higher=common_genes[ind_res_neg]


#7.) PHENOTYPE ENRICHMENT?
#We search in the binary phen matrix the enriched phen
phen_n_times_matching=np.zeros(len(phen))
#Antes de calcular el enriquecimiento necesitamos saber cuántas veces sale el fenotipo
#en la matriz general de genes coincidentes
for i in range(len(phen)):
    phen_n_times_matching[i]=np.sum(phen_matrix[:, i])
N_genes=len(genes)

#ENRICHMENT GENES
odd_ratio_enriched=np.zeros(len(phen))
p_value_enriched=np.zeros(len(phen))
n_genes_subset=len(ind_lot_ass)
phen_enriched_fisher_pleio=[]
p_val_pleio=[]
#Para cada uno de los fenotipos voy a tener un score asociado que me va a 
#indicar si está enriquecid
matrix_phen_pleio=[]
for i in range(len(genes_lot_ass)):
    ind_gene=np.where(genes==genes_lot_ass[i])[0]
    if len(ind_gene)>0:
        matrix_phen_pleio.append(phen_matrix[int(ind_gene), :])
matrix_phen_pleio=np.array(matrix_phen_pleio)

n_genes_phen=[]
n_genes_phen_subset=[]
for fen in range(len(matrix_phen_pleio[0, :])):
    phen_n_times_subset=np.sum(matrix_phen_pleio[:, fen])
    tabla=[[phen_n_times_subset, n_genes_subset-phen_n_times_subset],[phen_n_times_matching[fen], N_genes-phen_n_times_matching[fen]]]
    odd_ratio_enriched[fen], p_value_enriched[fen] = fisher_exact(tabla, alternative="greater") 
    if p_value_enriched[fen]<0.0001:
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
df_pleio.to_csv(path_save_data_real+'enriched_phen_lot_ass_genes.csv', sep='\t')



#ENRICHMENT NON PLEIO GENES
odd_ratio_enriched=np.zeros(len(phen))
p_value_enriched=np.zeros(len(phen))
n_genes_subset=len(ind_few_ass)
phen_enriched_fisher_non_pleio=[]
p_val_pleio=[]
#Para cada uno de los fenotipos voy a tener un score asociado que me va a 
#indicar si está enriquecid
matrix_phen_pleio=[]
for i in range(len(genes_few_ass)):
    ind_gene=np.where(genes==genes_few_ass[i])[0]
    if len(ind_gene)>0:
        matrix_phen_pleio.append(phen_matrix[int(ind_gene), :])
matrix_phen_pleio=np.array(matrix_phen_pleio)

n_genes_phen=[]
n_genes_phen_subset=[]
for fen in range(len(phen)):
    phen_n_times_subset=np.sum(matrix_phen_pleio[:, fen])
    tabla=[[phen_n_times_subset, n_genes_subset-phen_n_times_subset],[phen_n_times_matching[fen], N_genes-phen_n_times_matching[fen]]]
    odd_ratio_enriched[fen], p_value_enriched[fen] = fisher_exact(tabla, alternative="greater") 
    if p_value_enriched[fen]<0.0001:
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
df_non_pleio.to_csv(path_save_data_real+'enriched_phen_few_ass_genes.csv', sep='\t')






