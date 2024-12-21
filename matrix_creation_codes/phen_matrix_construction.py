# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 10:26:54 2024

@author: Alicia

Phenotypic space -> matrix construction
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
import umap
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import MDS
from scipy.stats import pearsonr
from sklearn.decomposition import NMF



path_save_data='C:\\Users\\logslab\\Desktop\\D-P_rule_paper\\matrix_construction\\phen_space\\'   


#1) ANALYSIS OF WORM PHENOTYPE ONTOLOGY (WPO)

#We read the data sheet with the Worm Phenotype Ontology
# url='https://downloads.wormbase.org/releases/current-production-release/ONTOLOGY/phenotype_ontology.WS290.obo'
url='C:\\Users\\logslab\\Desktop\\COSAS NUEVAS MAYO - fenotipos, genotipo\\phenotype_ontology.WS290.obo'
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
    
#1.1.) Phenotypes from WPO linked to cell phenotype 
#CELL PHENOTYPES
#We find how many phenotypes reach 'cell phenotype' and 'cell physiology phenotype'
phen_individual='cell phenotype'
ind_phen=np.where(phenotypes[:, 1]==phen_individual)[0]
phen_individual_id=phenotypes[int(ind_phen)][0]

phen_individual='cell physiology phenotype'
ind_phen=np.where(phenotypes[:, 1]==phen_individual)[0]
phen_individual_id2=phenotypes[int(ind_phen)][0]

cell_phen=[]
cell_phys_phen=[]
for i in range(len(pathlist)):
    for j in range(len(pathlist[i])):
        for k in range(len(pathlist[i][j])):
            if pathlist[i][j][k]==phen_individual_id:
                cell_phen.append(phenotypes[i][1])
            if pathlist[i][j][k]==phen_individual_id2:
                cell_phys_phen.append(phenotypes[i][1])
                
           
total_set=np.hstack((cell_phys_phen, cell_phen))
total_set=np.unique(total_set)

# np.savetxt(path_save_data+'cell_phen_in_WPO.txt', total_set, fmt='%s', delimiter=' ')


#2.) GENE-PHENOTYPE ASSOCIATION ANALYSIS 

#2.1) We read the gene-phenotype association dataset (version: WS290)
# f = open( "D:\\atlas_gusanos_bien\\leee_phenotypes\\phenotype_association.WS290.wb.txt", "r")
f=open('C:\\Users\\logslab\\Desktop\\COSAS NUEVAS MAYO - fenotipos, genotipo\\phenotype_association.WS290.wb.txt', "r")
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
matrix_genes=np.delete(matrix_genes, matrix_genes[:, 3]== "NOT", axis=0)
gene_id=matrix_genes[:, 1]
gene_name=matrix_genes[:, 2]
gene_phenotype=matrix_genes[:, 4]


#2.3.) Number of associations from mutations (genetic variation) or from RNAi

#In gene_exp and gene_exp2 we keep the associations type
gene_exp=matrix_genes[:, 5]
gene_exp2=matrix_genes[:, 7]
gene_exp_unique=np.unique(gene_exp)
gene_exp2_unique=np.unique(gene_exp2)

# #2.3.1) We check that all the associations are Var type or RNAi type
# count=0
# for i in range(len(gene_exp_unique)):
#     if 'Var' in gene_exp_unique[i]:
#         count=count+1
#     if 'REF' in gene_exp_unique[i]:
#         count=count+1

# count=0
# for i in range(len(gene_exp2_unique)):
#     if 'Var' in gene_exp2_unique[i]:
#         count=count+1
#     if 'Person' in gene_exp2_unique[i]:
#         count=count+1
#     if 'RNA' in gene_exp2_unique[i]:
#         count=count+1
    
#2.3.2) Analysis of association types
n_var=0
n_RNA=0
for i in range(len(gene_exp)):
    if 'Var' in gene_exp[i]:
        n_var=n_var+1
    if 'Var' in gene_exp2[i]:
        n_var=n_var+1
    if 'RNA' in gene_exp2[i]:
        n_RNA=n_RNA+1

# #figure
# pos=[0, 1]
# label=['Allele \n (variation)', 'RNAi']
# colors = ['cornflowerblue', 'mediumpurple']

# plt.figure(figsize=(5, 5), dpi=600)
# # barras=['sin clasificar', 'sin tipo - con lin', 'sin lin - con tipo', 'clasificadas', 'total']
# cuenta=(n_var, n_RNA)
# plt.bar(pos, cuenta, color=colors)
# plt.xticks(pos, label, fontsize=25, fontweight='bold', rotation=0)
# plt.yticks(fontsize=20)
# plt.ylabel("# associations", fontsize=25, fontweight='bold')
# plt.grid(False)
# plt.savefig(path_save_data+'association_experiment_type.png', dpi=600, bbox_inches='tight')
# plt.show()
        
    
#2.3.3) Distribution of the number of associations per gene
#We find the unique genes in the gene-phen association data sheet 
genes_id_type=np.array(list(set(gene_id)))
gene_name_type=np.array(list(set(gene_name)))

gene_id=np.array(gene_id)
n_associations=np.zeros(len(genes_id_type))
#cuantas asociaciones hay por gen
for i in range(len(genes_id_type)):
    n_associations[i]=len(np.where(gene_id==genes_id_type[i])[0])
    if n_associations[i]>200:
        print(genes_id_type[i])

# #figure
# plt.figure(figsize=(4, 3),dpi=600)
# plt.hist(n_associations, bins=100, color='lightseagreen', log=True)
# plt.ylabel('# genes', fontsize=14, fontweight='bold')
# plt.xlabel('# associations', fontsize=14, fontweight='bold')
# plt.xticks(fontsize=12)
# plt.yticks(fontsize=12)
# plt.grid(False)
# plt.axvline(x=np.median(n_associations), color='darkslategrey', linestyle='--', lw=1)
# plt.savefig(path_save_data+'association_distribution.png', dpi=600, bbox_inches='tight')
# plt.show()





#3.) GENE-PHENOTYPE ASSOCIATION MATRIX CONSTRUCTION

#3.1.) In this loop we find the associatied phenotypes with each unique gene
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

# great_matrix=glp.create_gene_phenotype_matrix(genes_id_type, phenotypes, phen_type, pathlist)


#4.) ANALYSIS OF THE GENE-PHENOTYPE MATRIX 

#4.1) We find the parent of each phenotype
all_parent=glp.find_parent(phenotypes[:, 0], graph)

#4.2.) We find the minumun layer of each phenotype, i.e., the minimun number of steps to reach 'nematode phenotype'
dif_layers=[]
for i in range(len(phenotypes)):
    inner_list=[]
    for j in range(len(pathlist[i])):
        inner_list.append(len(pathlist[i][j]))
    dif_layers.append(inner_list)
del inner_list

min_layer=np.zeros(len(phenotypes))
for i in range(len(phenotypes)):
    min_layer[i]=np.min(dif_layers[i])


#4.3) I keep just the parent phenotypes which are closer to the nematode phenotype
important_parent=[]
for i in range(len(phenotypes[:, 0])):
    definitive_min_layer=1000
    for j in range(len(all_parent[i])):
        ind_phen=np.where(phenotypes[:, 0]==all_parent[i][j])[0]
        ind_phen=int(ind_phen)
        check_min_layer=min_layer[ind_phen]
        if check_min_layer<definitive_min_layer:
            definitive_min_layer=check_min_layer
            index_pa=int(j)
    if i!=849:
        important_parent.append(all_parent[i][index_pa])
    else: 
        important_parent.append([])
del index_pa, check_min_layer, definitive_min_layer, ind_phen



#4.4.) DataFrame to sort the phenotypes 
df_all_phen=pd.DataFrame()
df_all_phen['phenotypes'] = phenotypes[:, 0]
df_all_phen['description'] = phenotypes[:, 1]
df_all_phen['parent'] = all_parent
df_all_phen['min_layer'] = min_layer
df_all_phen['path'] = pathlist
df_all_phen['important parent'] = important_parent
df_all_phen=df_all_phen.sort_values(by=["min_layer"])

descrip_sorted=np.array(df_all_phen["description"])
all_parent_sorted=np.array(df_all_phen['parent'])
phenotype_sorted=np.array(df_all_phen['phenotypes'])
pathlist_sorted=list(df_all_phen['path'])
min_layer_sorted=np.array(df_all_phen["min_layer"])
important_parent_sorted=np.array(df_all_phen['important parent'])

#in each layer, we sorted the phenotypes by the important parent name
#In this way, we have sorted the phenotypes by minumun layer and, inside each layer, we have grouped them by the same parent phenotype
count=0
df_final=pd.DataFrame()
for i in range(int(np.max(min_layer_sorted)+1)):
    index=np.where(min_layer_sorted==i)[0]
    phen=[]
    desc=[]
    imp_parent=[]
    paths=[]
    layers=[]
    all_parent=[]
    for j in range(len(index)):
        phen.append(phenotype_sorted[count])
        desc.append(descrip_sorted[count])
        imp_parent.append(important_parent_sorted[count])
        layers.append(min_layer_sorted[count])
        paths.append(pathlist_sorted[count])
        all_parent.append(all_parent_sorted[count])
        count=count+1
    df=pd.DataFrame()
    df['phen']=phen
    df['desc']=desc
    df['imp_parent']=imp_parent
    df['layers']=layers
    df['paths']=paths
    df['all_parent']=all_parent
    df=df.sort_values(by=["imp_parent"])
    df_final = pd.concat([df_final, df])

descrip_sorted=np.array(df_final['desc'])
all_parent_sorted=np.array(df_final['all_parent'])
phenotype_sorted=np.array(df_final['phen'])
pathlist_sorted=list(df_final['paths'])
min_layer_sorted=np.array(df_final["layers"])
important_parent_sorted=np.array(df_final['imp_parent'])

#4.4.) Analysis of base phenotypes (justification of NMF use)
base=[]
path_base=[]
for i in range(len(phenotypes)):
    #find subterms
    a=sorted(id_to_name[subterm] for subterm in nx.ancestors(graph, phenotypes[i][0]))
    if (len(a)==0):
        base.append(phenotypes[i][0])
        path_base.append(pathlist[i])

base=np.array(base)
dad_base=[]
for i in range(len(base)):
    ind=np.where(phenotype_sorted==base[i])
    ind=ind[0]
    dad_base.append(all_parent_sorted[i])

#4.4.1.) We compute the number of ancestors of base phenotypes to justify the existance of redundancy
base_phen_many_ancestors=[]
n_dad_base_no_rep=np.zeros(len(base))
for i in range(len(base)):
    dad_inner=[]
    for j in range(len(path_base[i])):
        for r in range(len(path_base[i][j])):
            dad_inner.append(path_base[i][j][r])
    dad_inner=np.unique(dad_inner, axis=0)
    n_dad_base_no_rep[i]=len(dad_inner)
    if n_dad_base_no_rep[i]>25:
        ind_phen=np.where(phenotypes[:, 0]==base[i])[0]
        base_phen_many_ancestors.append(phenotypes[int(ind_phen)][1])
    
base_phen_many_ancestors=np.array(base_phen_many_ancestors)
np.savetxt(path_save_data+'base_phen_many_ancestors.txt', base_phen_many_ancestors, fmt='%s')

        
##figure
plt.figure(figsize=(4, 3),dpi=600)
plt.hist(n_dad_base_no_rep, bins=100, color='forestgreen', log=True)
plt.ylabel('# base phenotypes', fontsize=14, fontweight='bold')
plt.xlabel('# ancestors', fontsize=14, fontweight='bold')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(False)
plt.savefig(path_save_data+'n_dad_base_phen_distribution.png', dpi=600, bbox_inches='tight')
plt.show()


#4.5.) Relationship between parents and minimun layers

#In all_parent_sorted we find the unique irect parent phenotypes linked to each child phen
#We find the unique parents (not base phenotypes)
#We count how many children a father has
inner_dad=[]
for i in range(len(all_parent_sorted)):
    for j in range(len(all_parent_sorted[i])):
        inner_dad.append(all_parent_sorted[i][j])
unique_dad=np.unique(inner_dad)
n_times_dad=np.zeros(len(unique_dad))
for i  in range(len(unique_dad)):
    for j in range(len(inner_dad)):
        if inner_dad[j]==unique_dad[i]:
            n_times_dad[i]=n_times_dad[i]+1

#We find the description and the minimun layer of each parent phenotype
descrip_parent=[]
min_layer_parent=[]
for i in range(len(unique_dad)):
    ind_phen=np.where(phenotype_sorted==unique_dad[i])[0]
    descrip_parent.append(descrip_sorted[int(ind_phen)])
    min_layer_parent.append(min_layer_sorted[int(ind_phen)])
 
# mean_dad=np.mean(n_times_dad) 
# std1=np.std(n_times_dad)   
 
# #figure
# fig=plt.figure(figsize=(5, 4), dpi=600)
# plt.scatter(min_layer_parent, n_times_dad, s=5, color="deeppink", alpha=0.35)
# for i in range(len(n_times_dad)):
#     if (n_times_dad[i]>mean_dad+6*std1) or ((min_layer_parent[i]>6) & (n_times_dad[i]>mean_dad+4*std1)):
#         plt.text(min_layer_parent[i]+0.05, n_times_dad[i]+0.2, descrip_parent[i], fontsize=9, color='blue')
# plt.ylabel("# direct children", fontsize=15)
# plt.xlabel("Minimun layer", fontsize=15)
# plt.xticks(fontsize=12)
# plt.yticks(fontsize=12)
# plt.savefig(path_save_data+"min_layer_vs_n_children.png", dpi=600, bbox_inches='tight')
# plt.show()


# 4.7.) We create the final sorted matrix with the initial genes
great_matrix=glp.create_gene_phenotype_matrix(genes_id_type, phenotype_sorted, phen_type, pathlist_sorted)

#4.8.) We identify null phenotypes and null genes
#There are not null genes
#We eleiminate null phenotypes
ind_null_genes=np.where(np.sum(great_matrix, axis=1)==0)[0]
ind_null_phen=np.where(np.sum(great_matrix, axis=0)==0)[0]

phenotype_description_non_null=np.delete(descrip_sorted, ind_null_phen)
phenotype_id_non_null=np.delete(phenotype_sorted, ind_null_phen)

great_matrix_non_null=np.zeros((len(genes_id_type), len(phenotype_id_non_null)))
for i in range(len(phenotype_id_non_null)):
    ind_phen=np.where(phenotype_sorted==phenotype_id_non_null[i])[0]
    great_matrix_non_null[:, i]=great_matrix[:, int(ind_phen)]
    
great_matrix=np.array(great_matrix_non_null)
phenotype_sorted=phenotype_id_non_null
descrip_sorted=phenotype_description_non_null
del great_matrix_non_null, phenotype_id_non_null, phenotype_description_non_null

#4.9.) Dsitribution of the number of phenotypes linked to each gene
n_phen_per_gene=np.sum(great_matrix, axis=1)

# ##figure
# plt.figure(figsize=(4, 3),dpi=600)
# plt.hist(n_phen_per_gene, bins=100, color='darkorange', log=True)
# plt.ylabel('# genes', fontsize=14, fontweight='bold')
# plt.xlabel('# associated phenotypes', fontsize=14, fontweight='bold')
# plt.xticks(fontsize=12)
# plt.yticks(fontsize=12)
# plt.grid(False)
# plt.axvline(x=np.median(n_phen_per_gene), color='saddlebrown', linestyle='--', lw=1)
# plt.savefig(path_save_data+'n_phen_per_gene_distribution.png', dpi=600, bbox_inches='tight')
# plt.show()

#Correlation between n associations and n phenotypes per gene
pearsonr(n_phen_per_gene, n_associations)


# #5.) Copy genes that are in the WPO and have associations
# np.savetxt(path_save_data+'gene_id_WPO_associations.txt', genes_id_type, fmt='%s')

# #6.) Copy sorted (by layer and parent) pehnotypes
# np.savetxt(path_save_data+'phenotypes_id.txt', phenotype_sorted, fmt='%s')
# np.savetxt(path_save_data+'phenotypes_description.txt', descrip_sorted, fmt='%s')

#7.) We read the intersection of genes between developmental and phenotypic space
#We build a new phenotype matrix with the selected genes
#We save that phenotypic matrix
path_read='C:\\Users\\logslab\\Desktop\\D-P_rule_paper\\matrix_construction\\dev_space\\'   
f=open(path_read+'genes_id.txt', 'r')
txt = f.read()
gene_intersection = txt.split('\n')
del txt, f
gene_intersection=np.delete(gene_intersection, len(gene_intersection)-1)
gene_intersection=np.array(gene_intersection)

new_phenotypic_matrix=np.zeros((len(gene_intersection), len(phenotype_sorted)))
for i in range(len(gene_intersection)):
    ind_gene=np.where(genes_id_type==gene_intersection[i])[0]
    new_phenotypic_matrix[i, :]=great_matrix[int(ind_gene), :]

np.savetxt(path_save_data+'gene_phen_matrix.txt', new_phenotypic_matrix, fmt='%d')


#8.) NNMF SPACE CONSTRUCTION
#From the phenotypic data, we build the nnmf matrix

# Initialize the NMF model with deterministic initialization using 'nndsvd'
n_components = 100  # The number of latent components (reduced dimensions)
model = NMF(n_components=n_components, init='nndsvd', random_state=42, max_iter=500)

# Fit the model to the data and perform the transformation
W = model.fit_transform(new_phenotypic_matrix)  # The reduced representation of the data
H = model.components_  # The latent components (patterns)

#We save the W matrix and the H matrix.
#We are going to use them always
np.savetxt(path_save_data+'W.txt', W, fmt='%f')
np.savetxt(path_save_data+'H.txt', H, fmt='%f')


#9.) DataFrame with H and its relationships
H=np.loadtxt(path_save_data+'H.txt')

f=open(path_save_data+'phenotypes_description.txt', 'r')
txt = f.read()
phen = txt.split('\n')
del txt, f
phen=np.delete(phen, len(phen)-1)
phen=np.array(phen)

comp=np.linspace(1, 100, 100, dtype=int)

df_H = pd.DataFrame(np.transpose(H), columns=comp, index=phen)
df_H.to_csv(path_save_data+'H_data_frame.csv', sep='\t')


