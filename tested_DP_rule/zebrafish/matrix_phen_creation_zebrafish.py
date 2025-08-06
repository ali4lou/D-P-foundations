# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 14:22:31 2025

@author: logslab


Zebrafish ANATOMICAL ONTOLOGY
"""

#construcción de la matriz fenotípica 

import obonet
import numpy as np
import os
import networkx as nx
import matplotlib.pyplot as plt 
import pandas as pd
import matplotlib as mpl
from scipy.cluster.hierarchy import dendrogram
# import gran_libreria_phenotypes as glp
from matplotlib import cm 
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import MDS
from scipy.stats import pearsonr
from collections import defaultdict
from scipy.stats import mstats, kstest, ttest_ind, fisher_exact


def find_non_common_elements(arr1, arr2):
    set1 = set(arr1)
    set2 = set(arr2)
    non_common = list(set1.symmetric_difference(set2))  # Elements in either set1 or set2 but not both
    return non_common

def find_ancestors(reverse_graph, term):
    ancestors = set()
    stack = [term]
    while stack:
        node = stack.pop()
        for parent in reverse_graph[node]:
            if parent not in ancestors:
                ancestors.add(parent)
                stack.append(parent)
    return ancestors



hgggggggggggggggggggggggggggggggggggg¡
ç++++++++++++++++++++++                  h+'anatomy_phen_descriptions.txt', 'r')
txt = f.read()
zfin_to_desc = txt.split('\n')
del txt, f
zfin_to_desc=np.delete(zfin_to_desc, len(zfin_to_desc)-1)
zfin_to_desc=np.array(zfin_to_desc)

zfin_desc_all= []
desc_all=[]
ini_stage=[]
final_stage=[]

for elemento in zfin_to_desc:
    partes = elemento.split("\t")
    zfin_desc_all.append(partes[0])
    desc_all.append(partes[1])
    ini_stage.append(partes[2])
    final_stage.append(partes[3])

del zfin_to_desc


ini_stage=np.array(ini_stage, dtype=str)
final_stage=np.array(final_stage, dtype=str)

unique_ini_stage, n_times_ini=np.unique(ini_stage, return_counts=True)



#1.) Zfin gene related to ENS gene
f=open(path+'zfin_to_ENS.txt', 'r')
txt = f.read()
zfin_to_ENS = txt.split('\n')
del txt, f
zfin_to_ENS=np.delete(zfin_to_ENS, len(zfin_to_ENS)-1)
zfin_to_ENS=np.array(zfin_to_ENS)

zfin_all = []
ENS_all=[]

for elemento in zfin_to_ENS:
    partes = elemento.split("\t")

    for i in range(len(partes)):
    
        if partes[i].startswith("ZDB"):
            zfin_all.append(partes[i])
        
        if partes[i].startswith("ENS"):
            ENS_all.append(partes[i])

ENS_all=np.array(ENS_all)
zfin_all=np.array(zfin_all)

del zfin_to_ENS, partes, elemento



#2.) gene-phen associations
with open(path+'phen_gene_data.txt', 'r') as file:
    data = file.read()
rows = data.split('\n')
rows = [row.strip() for row in rows if row.strip()]
del data, file

zfin_associated = []
phen_associated_id=[]
phen_associated_desc=[]
phen_thing=[]

for elemento in rows:
    partes = elemento.split("\t")
    
    for i in range(len(partes)):
        if partes[i].startswith("ZFA"):
            zfin_associated.append(partes[2])
            phen_associated_id.append(partes[i])
            phen_associated_desc.append(partes[int(i+1)])
            phen_thing.append(partes[10])
    


phen_associated_desc=np.array(phen_associated_desc)
phen_associated_id=np.array(phen_associated_id)
zfin_associated=np.array(zfin_associated)












































uuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuij, partes


#2.1.) We search the ENS genes associted
ENS_associated=[]
phen_associated_id_final=[]
phen_associated_desc_final=[]
phen_thing_final=[]
count_rep_genes=0
for i in range(len(zfin_associated)):
    ind_gene=np.where(zfin_all==zfin_associated[i])[0]
    if len(ind_gene)==1:
        phen_associated_id_final.append(phen_associated_id[i])
        phen_associated_desc_final.append(phen_associated_desc[i])
        phen_thing_final.append(phen_thing[i])
        ENS_associated.append(ENS_all[int(ind_gene)])
    if len(ind_gene)>1:
        count_rep_genes=count_rep_genes+1
        for j in range(len(ind_gene)):
            phen_associated_id_final.append(phen_associated_id[i])
u           b            phen_associated_desc_final.append(phen_associated_desc[i])
            phen_thing_final.append(phen_thing[i])
            ENS_associated.append(ENS_all[int(ind_gene[j])])
    
ENS_associated=np.array(ENS_associated)
phen_associated_id_final=np.array(phen_associated_id_final)
phen_associated_desc_final=np.array(phen_associated_desc_final)
phen_thing_final=np.array(phen_thing_final)

del zfin_associated, phen_associated_desc, phen_associated_id, phen_thing


#2.2.) We delete the associations related to phen of very early stages
ind_phen_stage=np.where(phen_associated_id_final=='ZFA:0001175')[0]
len(ind_phen_stage)
phen_associated_desc_final[ind_phen_stage]
ind_to_delete=[ind_phen_stage]
#we eliminate those genes and associations

ind_phen_stage=np.where(phen_associated_id_final=='ZFA:0005772')[0]
len(ind_phen_stage)
ind_to_delete.append(ind_phen_stage)
phen_associated_desc_final[ind_phen_stage]

ind_phen_stage=np.where(phen_associated_id_final=='ZFA:0005604')[0]
len(ind_phen_stage)
ind_to_delete.append(ind_phen_stage)
phen_associated_desc_final[ind_phen_stage]

ind_to_delete=np.concatenate(ind_to_delete)
ENS_associated=np.delete(ENS_associated, ind_to_delete)
phen_associated_id_final=np.delete(phen_associated_id_final, ind_to_delete)
phen_associated_desc_final=np.delete(phen_associated_desc_final, ind_to_delete)
phen_thing_final=np.delete(phen_thing_final, ind_to_delete)

#3.) We search the ontology 
zfa_children=[]
zfa_parent=[]
with open(path+'anatomy_relationship.txt', 'r') as file:
    input_data = file.read()
rows = input_data.split('\n')
rows = [row.strip() for row in rows if row.strip()]
del file

for elemento in rows:
    partes = elemento.split("\t")
    if partes[2].startswith("is"):
        zfa_children.append(partes[1])
        zfa_parent.append(partes[0])
    if partes[2].startswith("part"):
        zfa_children.append(partes[1])
        zfa_parent.append(partes[0])

zfa_parent=np.array(zfa_parent)
zfa_children=np.array(zfa_children)


#exploratory analysis of the number of terms

unique_child=np.unique(zfa_children)
unique_parent=np.unique(zfa_parent)

child_inters_parent=np.intersect1d(unique_child, unique_parent)

specific_child=find_non_common_elements(unique_child, unique_parent)
real_specific_child=np.intersect1d(unique_child, specific_child)
top_ontology=np.intersect1d(unique_parent, specific_child)

n_phen_ont=len(real_specific_child)+len(unique_parent)
print('total zfa ontology', n_phen_ont)

print('We find %d specific pehnotypes' %len(real_specific_child))

zfa_unique=np.unique(phen_associated_id_final)
print('Total unique phenotypes associated:', len(zfa_unique))
print('Speficific phenotypes from that total:', len(np.intersect1d(real_specific_child, phen_associated_id_final)))
print('Parent phenotypes from that total:', len(np.intersect1d(unique_parent, phen_associated_id_final)))


# 3.1.) Build the graph and its reverse
graph = defaultdict(list)
reverse_graph = defaultdict(list)
for parent, child in zip(zfa_parent, zfa_children):
    graph[parent].append(child)
    reverse_graph[child].append(parent)

#3.2.) Find ancestors for a given term
results = {}
ancestors_list=[]
for term in zfa_unique:
    results[term] = find_ancestors(reverse_graph, term)
    ancestors = find_ancestors(reverse_graph, term)
    ancestors_list.append([term, list(ancestors)])

#3.2.1.) We find all the needed terms
all_redundant_terms=[]
for i in range(len(zfa_unique)):
    all_redundant_terms.append(ancestors_list[i][0])
    for j in range(len(ancestors_list[i][1])):
        all_redundant_terms.append(ancestors_list[i][1][j])

all_redundant_terms_unique=np.unique(np.array(all_redundant_terms))


#3.3.) We find the layers and the phentypes
layers=[]
layers.append(top_ontology)
count_phen=1
for i in range(17):
    inner_terms=[]
    for j in range(len(layers[i])):
        ind_zfa=np.where(zfa_parent==layers[i][j])[0]
        for k in range(len(ind_zfa)):
            inner_terms.append(zfa_children[int(ind_zfa[k])])
            count_phen=count_phen+1
    layers.append(inner_terms)
    

        

#3.3.1.) We sort the associated phenotypes by layer
unique_genes=np.unique(ENS_associated)

zfa_layer=np.zeros(len(all_redundant_terms_unique))
for i in range(len(all_redundant_terms_unique)):
    for j in range(len(layers)):
        for k in range(len(layers[j])):
            if layers[j][k]==all_redundant_terms_unique[i]:
                if (zfa_layer[i]!=0): 
                    if j<zfa_layer[i]:
                        zfa_layer[i]=j
                else:
                    zfa_layer[i]=j


zfin_desc_all.append(top_ontology[0])
desc_all.append('zebrafish anatomical entity')

zfin_desc_all=np.array(zfin_desc_all)
desc_all=np.array(desc_all)

zfa_unique_desc=[]
for i in range(len(all_redundant_terms_unique)):
    ind=np.where(zfin_desc_all==all_redundant_terms_unique[i])[0]
    zfa_unique_desc.append(desc_all[int(ind[0])])
    
df_zfa=pd.DataFrame()
df_zfa['zfa']=all_redundant_terms_unique
df_zfa['desc']=zfa_unique_desc
df_zfa['layer']=zfa_layer

df_zfa=df_zfa.sort_values(by=['layer'])

zfa_unique_matrix=np.array(list(df_zfa['zfa']))
zfa_unique_desc_matrix=np.array(list(df_zfa['desc']))


#4.) We built the gene-phenotype association matrix
gene_phen_association_matrix=np.zeros((len(unique_genes), len(zfa_unique_matrix)))
for i in range(len(unique_genes)):
    ind_association=np.where(ENS_associated==unique_genes[i])
    searched_phen=phen_associated_id_final[ind_association]
    for j in range(len(searched_phen)):
        ind_phen=np.where(zfa_unique==searched_phen[j])[0]
        ind_matrix=np.where(zfa_unique_matrix==zfa_unique[int(ind_phen)])[0]
        gene_phen_association_matrix[i][int(ind_matrix)]=1
        for k in range(len(ancestors_list[int(ind_phen)][1])):
            ind_matrix=np.where(zfa_unique_matrix==ancestors_list[int(ind_phen)][1][k])[0]
            gene_phen_association_matrix[i][int(ind_matrix)]=1

    print(i)

np.savetxt(path+'gene_phen_association_matrix.txt', gene_phen_association_matrix)
np.savetxt(path+'genes_associated_phen.txt', unique_genes, fmt='%s', delimiter=',')
np.savetxt(path+'phen_anatomy.txt', zfa_unique_desc_matrix, fmt='%s', delimiter=',')

