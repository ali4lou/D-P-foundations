# -*- coding: utf-8 -*-
"""
GO terms associated with pleio and non pleio genes
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
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import MDS
from scipy.stats import pearsonr
from sklearn.decomposition import NMF
from scipy.stats import fisher_exact, false_discovery_control
from scipy.stats import kstest




def flatten_and_unique(nested_list):
    result = []
    for sublist in nested_list:
        unique_items = set()
        for item in sublist:
            unique_items.update(item)  
        result.append(list(unique_items)) 
    return result


def create_association_matrix_gene_go_term(gene_subset, GO_specific_terms, pathlist_specific_terms, GO_specific_terms_descrip):
    
    common_genes=np.intersect1d(gene_subset, genes_id_type)
    print(len(common_genes))
    GO_specific_terms=np.array(GO_specific_terms)
    
    matrix=np.zeros((len(common_genes), len(GO_specific_terms)))
    
    for i in range(len(common_genes)):
        ind_gene=np.where(genes_id_type==common_genes[i])[0]
        for j in range(len(go_type[int(ind_gene)])):
            
            go_ind=np.where(GO_specific_terms==go_type[int(ind_gene)][j])[0]
            if len(go_ind)>0:
                ind_matrix = np.isin(GO_specific_terms, pathlist_specific_terms[int(go_ind)])
                matrix[i, ind_matrix] = 1
    return matrix
    
    # mean_association_gene_go_term=np.mean(matrix, axis=0)
    # ind_mean_association_sorted=np.argsort(mean_association_gene_go_term)[::-1]

    # GO_specific_terms_descrip_sorted=[]
    # mean_association_gene_go_term_sorted=np.zeros(len(mean_association_gene_go_term))
    # for i in range(len(ind_mean_association_sorted)):
    #     GO_specific_terms_descrip_sorted.append(GO_specific_terms_descrip[int(ind_mean_association_sorted[i])])
    #     mean_association_gene_go_term_sorted[i]=mean_association_gene_go_term[int(ind_mean_association_sorted[i])]
    # GO_specific_terms_descrip_sorted=np.array(GO_specific_terms_descrip_sorted)
    
    # df_mean_association_gene_go_term=pd.DataFrame()
    # df_mean_association_gene_go_term['GO term']=GO_specific_terms_descrip_sorted
    # df_mean_association_gene_go_term['frac associated genes']=mean_association_gene_go_term_sorted

    # return df_mean_association_gene_go_term, matrix


def enrichement_go(big_matrix_all_genes, submatrix_gene_go, go_list, go_list_term, label):

    #numberof times that a go_term is associated with a gene
    go_n_times_all_genes=np.zeros(len(go_list))
    for i in range(len(go_list)):
        go_n_times_all_genes[i]=np.sum(big_matrix_all_genes[:, i])

    odd_ratio_enrich=np.zeros(len(go_list))
    p_value_enrich=np.zeros(len(go_list))
    n_genes_subset=len(submatrix_gene_go[:, 0])
    go_enrich_fisher_genes_subset=[]
    go_term_enrich=[]
    p_value_enriched_go_term=[]
    n_genes=[]
    n_genes_subset_associated_go=[]
    subset_analyzed=[]
    #For each phenotype we compute a score that indicates if the phenotypes is enriched
    for fen in range(len(go_list)):
        go_n_times_subset=np.sum(submatrix_gene_go[:, fen])
        tabla=[[go_n_times_subset, n_genes_subset-go_n_times_subset],[go_n_times_all_genes[fen], len(big_matrix_all_genes[:, 0])-go_n_times_all_genes[fen]]]
        odd_ratio_enrich[fen], p_value_enrich[fen] = fisher_exact(tabla, alternative="greater") 
        if p_value_enrich[fen]<0.001:
            go_enrich_fisher_genes_subset.append(go_list[fen])
            go_term_enrich.append(go_list_term[fen])
            p_value_enriched_go_term.append(p_value_enrich[fen])
            n_genes.append(go_n_times_all_genes[fen])
            n_genes_subset_associated_go.append(go_n_times_subset)
            subset_analyzed.append(label)
            
    return np.array(subset_analyzed), np.array(go_enrich_fisher_genes_subset), np.array(go_term_enrich), np.array(p_value_enriched_go_term), np.array(n_genes), np.array(n_genes_subset_associated_go)



"""
path_save_data, path_dev, path_sim...
are the paths that you chose after download the needed files
"""

path_save_data='YOUR_PATH_TO_SAVE_DATA'
path_dev='PATH_WHERE_IS_DOWNLOADED_THE_DATA'
path_sec1_deviations='PATH_WHERE_IS_DOWNLOADED_THE_DATA'
path_sec2='PATH_WHERE_IS_DOWNLOADED_THE_DATA'
path_sim='PATH_WHERE_IS_DOWNLOADED_THE_DATA'


f=open(path_dev+'genes_id.txt', 'r')
txt = f.read()
genes = txt.split('\n')
del txt, f
genes=np.delete(genes, len(genes)-1)
genes=np.array(genes)


#1) ANALYSIS OF WORM PHENOTYPE ONTOLOGY (WPO)

#We read the data sheet with the Worm Phenotype Ontology
# url='https://downloads.wormbase.org/releases/current-production-release/ONTOLOGY/phenotype_ontology.WS290.obo'
url=path_save_data+'gene_ontology.WS294.obo'
graph = obonet.read_obo(url)

#create mappings
id_to_name = {id_: data.get('name') for id_, data in graph.nodes(data=True)}
name_to_id = {data['name']: id_ for id_, data in graph.nodes(data=True) if 'name' in data}
gene_description = {data['def']: id_ for id_, data in graph.nodes(data=True) if 'def' in data}

del url


#We keep nodes -> id phenotype and name phenotype
GO_term=glp.convert_dictionary_to_array(id_to_name)
go_description=glp.convert_dictionary_to_array(gene_description)

#We find all the possible paths for each phenotype
pathlist_bio_process=[]
GO_bio_process=[]
GO_bio_process_descrip=[]
for i in range(len(GO_term)):
    start=id_to_name[GO_term[i][0]]
    
    paths = nx.all_simple_paths(
        graph,
        source=name_to_id[start],
        target=name_to_id['biological_process']
    )
    innerlist = []
    for path in paths:
        innerlist.append(path)
    
    if len(innerlist)>0:
        pathlist_bio_process.append(innerlist)
        GO_bio_process.append(GO_term[i][0])
        GO_bio_process_descrip.append(GO_term[i][1])
        
pathlist_cell_comp=[]
GO_cell_comp=[]
GO_cell_comp_descrip=[]
for i in range(len(GO_term)):
    start=id_to_name[GO_term[i][0]]
    
    paths = nx.all_simple_paths(
        graph,
        source=name_to_id[start],
        target=name_to_id['cellular_component']
    )
    innerlist = []
    for path in paths:
        innerlist.append(path)
        
    if len(innerlist)>0:
        pathlist_cell_comp.append(innerlist)
        GO_cell_comp.append(GO_term[i][0])
        GO_cell_comp_descrip.append(GO_term[i][1])
    
pathlist_molecular=[]
GO_molecular=[]
GO_molecular_descrip=[]
for i in range(len(GO_term)):
    start=id_to_name[GO_term[i][0]]
    
    paths = nx.all_simple_paths(
        graph,
        source=name_to_id[start],
        target=name_to_id['molecular_function']
    )
    innerlist = []
    for path in paths:
        innerlist.append(path)
    
    if len(innerlist)>0:
        pathlist_molecular.append(innerlist)
        GO_molecular.append(GO_term[i][0])
        GO_molecular_descrip.append(GO_term[i][1])


#2.) GENE-GOterm ASSOCIATION ANALYSIS 

#2.1) We read the gene-GOterm association dataset (version: WS294)
f=open(path_save_data+'gene_association_nonnoctua.WS294.wb.txt', "r")
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


gene_id=np.array(matrix_genes[:, 1], dtype=str)
gene_name=matrix_genes[:, 2]
GOterm_association=np.array(list(matrix_genes[:, 4]), dtype=str)
type_association=matrix_genes[:, 3]


genes_id_type=np.unique(gene_id)
genes_id_type=np.delete(genes_id_type, 0)


#3.) In this loop we find the associatied go_terms with each unique gene
go_type=[]
#go_terms corresponing genes_id_tipo 
for i in range(len(genes_id_type)):
    ind_genes=np.where(gene_id==genes_id_type[i])
    ind_genes=ind_genes[0]
    # inner_list_bio_process=[]
    # inner_list_cell_comp=[]
    # inner_list_molecular=[]
    inner_list=[]
    for j in range(len(ind_genes)):
        index=ind_genes[j]
        inner_list.append(GOterm_association[index])
    go_type.append(inner_list)
del ind_genes, inner_list, index




#4.) We read each subset of genes to analyze
#4.1.) We read pleio and non pleio genes
pleio_genes=np.loadtxt(path_sec2+'pleio_genes.txt', dtype=str)
non_pleio_genes=np.loadtxt(path_sec2+'non_pleio_genes.txt', dtype=str)

#4.2.) Genes following the d-p rule and deviations
genes_dP_dev_deviation=np.loadtxt(path_sec1_deviations+'genes_dP_dev_deviation.txt', dtype=str)
genes_Dp_phen_deviation=np.loadtxt(path_sec1_deviations+'genes_Dp_phen_deviation.txt', dtype=str)
genes_dp_rule=np.loadtxt(path_sec1_deviations+'genes_dp_rule.txt', dtype=str)

#4.3.) We read dev and phen similarities and take two subsets (extreme percentiles)

#4.3.1.) We compute the median of the sim_dev and sim_phen in each gene
sim_dev=np.loadtxt(path_sim+'sim_dev_frac_cells_cosine.txt')
sim_phen=np.loadtxt(path_sim+'sim_W_matrix_cosine.txt')

sim_dev_matrix=squareform(sim_dev)
sim_phen_matrix=squareform(sim_phen)

average_sim_dev_per_gen=np.median(sim_dev_matrix, axis=1)
average_sim_phen_per_gen=np.median(sim_phen_matrix, axis=1)

del sim_dev, sim_phen, sim_dev_matrix, sim_phen_matrix

#Genes with high and low percentiles DEV SIM
ind_genes_high_sim_dev=np.where(average_sim_dev_per_gen>np.percentile(average_sim_dev_per_gen, 75))[0]
genes_high_sim_dev=genes[ind_genes_high_sim_dev]
high_sim_D=average_sim_dev_per_gen[ind_genes_high_sim_dev]

ind_genes_low_sim_dev=np.where(average_sim_dev_per_gen<np.percentile(average_sim_dev_per_gen, 25))[0]
genes_low_sim_dev=genes[ind_genes_low_sim_dev]
low_sim_D=average_sim_dev_per_gen[ind_genes_low_sim_dev]


#Genes with high and low percentiles PHEN SIM
ind_genes_high_sim_phen=np.where(average_sim_phen_per_gen>np.percentile(average_sim_phen_per_gen, 75))[0]
genes_high_sim_phen=genes[ind_genes_high_sim_phen]
high_sim_P=average_sim_phen_per_gen[ind_genes_high_sim_phen]

ind_genes_low_sim_phen=np.where(average_sim_phen_per_gen<np.percentile(average_sim_phen_per_gen, 25))[0]
genes_low_sim_phen=genes[ind_genes_low_sim_phen]
low_sim_P=average_sim_phen_per_gen[ind_genes_low_sim_phen]


# #We save the median similarities dataframe
# df1 = pd.DataFrame({'genes': genes, 'median_developmental_similarity': average_sim_dev_per_gen, 'median_phenotypic_similarity': average_sim_phen_per_gen})
# df2 = pd.DataFrame({'genes': genes_high_sim_dev, 'high_sim_D': high_sim_D})
# df3 = pd.DataFrame({'genes': genes_low_sim_dev, 'low_sim_D': low_sim_D})
# df4 = pd.DataFrame({'genes': genes_high_sim_phen, 'high_sim_P': high_sim_P})
# df5 = pd.DataFrame({'genes': genes_low_sim_phen, 'low_sim_P': low_sim_P})

# with pd.ExcelWriter(path_sim+'genes_median_similarities.xlsx') as writer:
#     df1.to_excel(writer, sheet_name='median similarity')
#     df2.to_excel(writer, sheet_name='genes_high_sim_D')
#     df3.to_excel(writer, sheet_name='genes_low_sim_D')
#     df4.to_excel(writer, sheet_name='genes_high_sim_P')
#     df5.to_excel(writer, sheet_name='genes_low_sim_P')



#ENRICHMENT
#5.) Gene-GO term associaciation matrix  and enrichment (for each subset of genes to analyze)
pathlist_cell_comp_unique = flatten_and_unique(pathlist_cell_comp)
pathlist_bio_process_unique=flatten_and_unique(pathlist_bio_process)
pathlist_molecular_unique=flatten_and_unique(pathlist_molecular)


GO_bio_process_descrip=np.array(GO_bio_process_descrip)
np.where(GO_bio_process_descrip=='biological_process')[0]


#To obtain all the GO terms enriched in each subset of genes, we have to match the genes_analyze variable to the subset of genes we want to analyze.
#The possible variables are: genes_high_sim_dev, genes_low_sim_dev, genes_high_sim_phen, genes_low_sim_phen, pleio_genes, non_pleio_genes, genes_dP_dev_deviation, genes_Dp_phen_deviation, genes_dp_rule
genes_analyze=genes_Dp_phen_deviation
genes_label='genes_Dp_phen_deviation'
label_analyzed='genes_Dp_P_dev'

# #5.1.) BIOLOGICAL PROCESS

big_matrix_bio_process=create_association_matrix_gene_go_term(genes, GO_bio_process, pathlist_bio_process_unique, GO_bio_process_descrip)


matrix_subset_bio_process=create_association_matrix_gene_go_term(genes_analyze, GO_bio_process, pathlist_bio_process_unique, GO_bio_process_descrip)
subset, enrich_subset_bio_process, enrich_go_term, p_value, n_genes, n_genes_subset=enrichement_go(big_matrix_bio_process, matrix_subset_bio_process, GO_bio_process_descrip, GO_bio_process,  label_analyzed)

index_sorted=np.argsort(p_value)

corrected_p_value=false_discovery_control(p_value, method='bh')
n_genes=np.array(n_genes, dtype=int)
n_genes_subset=np.array(n_genes_subset, dtype=int)


df_bio_process=pd.DataFrame()
df_bio_process['subset']=subset
df_bio_process['GO term']=enrich_go_term[index_sorted]
df_bio_process['GO description']=enrich_subset_bio_process[index_sorted]
df_bio_process['n genes']=n_genes[index_sorted]
df_bio_process['n genes subset']=n_genes_subset[index_sorted]
df_bio_process['p-value']=p_value[index_sorted]
df_bio_process['corected p-value']=corrected_p_value[index_sorted]

# np.savetxt(path_save_data+'enrich_go_bio_process_%s.txt' %genes_label, enrich_subset_bio_process, fmt='%s')
df_bio_process.to_csv(path_save_data+'enrich_go_bio_process_%s.csv' %genes_label, sep='\t')

del big_matrix_bio_process, matrix_subset_bio_process, enrich_subset_bio_process, df_bio_process

# #5.2.) MOLECULAR FUNCTION

big_matrix_molecular_function=create_association_matrix_gene_go_term(genes, GO_molecular, pathlist_molecular_unique, GO_molecular_descrip)


matrix_subset_molecular_func=create_association_matrix_gene_go_term(genes_analyze, GO_molecular, pathlist_molecular_unique, GO_molecular_descrip)
subset, enrich_subset_molecular_func, enrich_go_term, p_value, n_genes, n_genes_subset=enrichement_go(big_matrix_molecular_function, matrix_subset_molecular_func, GO_molecular_descrip, GO_molecular, label_analyzed)

index_sorted=np.argsort(p_value)

corrected_p_value=false_discovery_control(p_value, method='bh')
n_genes=np.array(n_genes, dtype=int)
n_genes_subset=np.array(n_genes_subset, dtype=int)


df_molecular=pd.DataFrame()
df_molecular['subset']=subset
df_molecular['GO term']=enrich_go_term[index_sorted]
df_molecular['GO description']=enrich_subset_molecular_func[index_sorted]
df_molecular['n genes']=n_genes[index_sorted]
df_molecular['n genes subset']=n_genes_subset[index_sorted]
df_molecular['p-value']=p_value[index_sorted]
df_molecular['corected p-value']=corrected_p_value[index_sorted]

df_molecular.to_csv(path_save_data+'enrich_go_molecular_func_%s.csv' %genes_label, sep='\t')
# np.savetxt(path_save_data+'enrich_go_molecular_func_%s.txt' %genes_label, enrich_subset_molecular_func, fmt='%s')

del big_matrix_molecular_function, matrix_subset_molecular_func, enrich_subset_molecular_func, df_molecular


# #5.3.) CELL COMPONENT

big_matrix_cell_comp=create_association_matrix_gene_go_term(genes, GO_cell_comp, pathlist_cell_comp_unique, GO_cell_comp_descrip)


matrix_subset_cell_comp=create_association_matrix_gene_go_term(genes_analyze, GO_cell_comp, pathlist_cell_comp_unique, GO_cell_comp_descrip)
subset, enrich_subset_cell_comp, enrich_go_term, p_value, n_genes, n_genes_subset=enrichement_go(big_matrix_cell_comp, matrix_subset_cell_comp, GO_cell_comp_descrip, GO_cell_comp, label_analyzed)

index_sorted=np.argsort(p_value)

corrected_p_value=false_discovery_control(p_value, method='bh')
n_genes=np.array(n_genes, dtype=int)
n_genes_subset=np.array(n_genes_subset, dtype=int)


df_cell_comp=pd.DataFrame()
df_cell_comp['subset']=subset
df_cell_comp['GO term']=enrich_go_term[index_sorted]
df_cell_comp['GO description']=enrich_subset_cell_comp[index_sorted]
df_cell_comp['n genes']=n_genes[index_sorted]
df_cell_comp['n genes subset']=n_genes_subset[index_sorted]
df_cell_comp['p-value']=p_value[index_sorted]
df_cell_comp['corected p-value']=corrected_p_value[index_sorted]

df_cell_comp.to_csv(path_save_data+'enrich_go_cell_comp_%s.csv' %genes_label, sep='\t')

# np.savetxt(path_save_data+'enrich_go_cell_comp_%s.txt' %genes_label, enrich_subset_cell_comp, fmt='%s')

del big_matrix_cell_comp, matrix_subset_cell_comp, enrich_subset_cell_comp, df_cell_comp















