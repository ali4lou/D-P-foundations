# -*- coding: utf-8 -*-
"""

Pleiotropy comparison with paper of 2024

"""

#Study of genes from paper automated profiling of gene function during embryonic developmental 

import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
from scipy.cluster.hierarchy import dendrogram
from matplotlib import cm 
import seaborn as sns
from scipy.stats import pearsonr
from sklearn.decomposition import NMF
from sklearn.metrics import mean_squared_error
from scipy.stats import fisher_exact
from sklearn.metrics.pairwise import pairwise_distances


"""
path_save_data, path_dev, path_phen, path_sim and path_pleio
are the path that you chose after download the needed files
"""

path_save_data='YOUR_PATH_TO_SAVE_DATA'

#1.) We are going to read all the data
path_dev='PATH_WHERE_IS_DOWNLOADED_THE_DATA'
path_phen='PATH_WHERE_IS_DOWNLOADED_THE_DATA'
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
# H=np.loadtxt(path_phen+'H.txt')

#1.4.) We read NMF pleio score
f=open(path_pleio+'nmf_pleiotropy.txt', 'r')
txt = f.read()
pleio_score_nnmf = txt.split('\n')
del txt, f
pleio_score_nnmf=np.delete(pleio_score_nnmf, len(pleio_score_nnmf)-1)
pleio_score_nnmf=np.array(pleio_score_nnmf, dtype='f')


# #1.5.) We read developmental matrices
# dev_matrix_sum_cells=np.loadtxt(path_dev+'n_cells_per_coord_matrix.txt')

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




#2.) We are going to read the genes o the paper
# We read the txt
name_arch = path_save_data+'c_elegans.PRJNA13758.WS292.geneIDs.txt'
df_genes_c_elegans = pd.read_csv(name_arch, delimiter=',', header=None)

sec_worm_base=np.array(df_genes_c_elegans[3])
genes_worm_base=np.array(df_genes_c_elegans[1])

#We read the excel
name_arch = path_save_data+'mmc1.xlsx'
df_excel_paper = pd.read_excel(name_arch, sheet_name='Table S1.6 Manual Scoring')


#We find there a column with the analyzed phenotypes
#A table reduced and annotated
sec_paper=np.array(df_excel_paper['Gene Target'])
sec_paper=np.delete(sec_paper, 0)
sec_paper=np.unique(sec_paper)


#We find the matching sec
sec_comunes=np.array(list(set(sec_paper)&set(sec_worm_base)))
sec_search_data_paper=[]
for i in range(len(sec_comunes)):
    ind_gen=np.where(sec_worm_base==sec_comunes[i])[0]
    if len(ind_gen)==1:
        sec_search_data_paper.append(sec_comunes[i])

# Changes in the dataframe
df_excel_paper.columns = df_excel_paper.iloc[0].astype(str)

df_excel_paper = df_excel_paper[1:]


df_excel_paper_germ_layer=df_excel_paper.iloc[:, :22]
df_excel_paper_morphogenesis = df_excel_paper.iloc[:, 22:]

#We obtain the total number of embryos
total_embryos = np.array(list(df_excel_paper_germ_layer.iloc[:, 5]))

#We select the columns where the name of the column starts with 'WB'
columnas_wb_germ_layer = [col for col in df_excel_paper_germ_layer.columns if str(col).startswith('WB')]
columnas_wb_morpho = [col for col in df_excel_paper_morphogenesis.columns if str(col).startswith('WB')]


# We create a dataframe with the filtered columns
df_excel_paper_filtered_germ_layer = df_excel_paper_germ_layer[columnas_wb_germ_layer]
df_excel_paper_filtered_morphogenesis=df_excel_paper_morphogenesis[columnas_wb_morpho]

#We build the matrix 
phen_paper_germ_layer=np.array(df_excel_paper_filtered_germ_layer.columns.tolist())
phen_paper_morpho=np.array(df_excel_paper_filtered_morphogenesis.columns.tolist())

sec_mirar_data_paper=np.array(sec_search_data_paper, dtype=str)
sec_final_common, index_paper, index_data_paper = np.intersect1d(sec_paper, sec_search_data_paper, return_indices=True)


matrix_germ_layer=np.array(df_excel_paper_filtered_germ_layer)
matrix_germ_layer=np.delete(matrix_germ_layer, 0, axis=1)
matrix_morpho=np.array(df_excel_paper_filtered_morphogenesis)



genes_common=[]
for i in range(len(sec_final_common)):
    ind_gen=np.where(sec_worm_base==sec_final_common[i])[0]
    if len(ind_gen)==1:
        genes_common.append(genes_worm_base[int(ind_gen)])
genes_common=np.array(genes_common)


genes_common_total_paper=np.array(list(set(genes) & set(genes_common)))


matrix_morpho_final=np.zeros((len(genes_common), len(matrix_morpho[0, :])))
matrix_germ_layer_final=np.zeros((len(genes_common), len(matrix_germ_layer[0, :])))
total_embryos_final=np.zeros(len(genes_common))
for i in range(len(index_paper)):
    matrix_germ_layer_final[i, :]=matrix_germ_layer[int(index_paper[i]), :]
    matrix_morpho_final[i, :]=matrix_morpho[int(index_paper[i]), :]
    total_embryos_final[i]=total_embryos[i]



#3. Pleiotropy comparison (nnmf, germ layer, morpho)
#The associated genes are keep in 
pleio_nnmf_paper=np.zeros(len(genes_common_total_paper))
pleio_germ_layer=np.zeros(len(genes_common_total_paper))
pleio_morpho=np.zeros(len(genes_common_total_paper))

for i in range(len(genes_common_total_paper)):
    ind_gen_sec=np.where(genes_common==genes_common_total_paper[i])[0]
    ind_gen_total=np.where(genes==genes_common_total_paper[i])[0]
    print(ind_gen_sec)
    pleio_germ_layer[i]=np.sum(matrix_germ_layer[int(ind_gen_sec), :])/total_embryos_final[int(ind_gen_sec)]
    pleio_morpho[i]=np.sum(matrix_morpho[int(ind_gen_sec), :])/total_embryos_final[int(ind_gen_sec)]
    pleio_nnmf_paper[i]=pleio_score_nnmf[int(ind_gen_total)]
    
pearsonr(pleio_germ_layer,pleio_morpho)

print('NMF pleio vs. paper morpho pleio:', pearsonr(pleio_nnmf_paper,pleio_morpho), '\n')
print('NMF pleio vs. paper germ layer pleio:',pearsonr(pleio_nnmf_paper, pleio_germ_layer), '\n')


plt.scatter(pleio_morpho, pleio_germ_layer, s=0.9)    
    
plt.scatter(pleio_nnmf_paper, pleio_germ_layer, s=0.9)    


plt.hist(pleio_nnmf_paper, bins=100)
    


#4. Pleiotropy comparison using the phen matrix of the WormBase taking into account the associations
#Just with the phenotypes they analyze 

import obonet
import great_library_phenotypes as glp

#leemos archivo de phenotypes
# url='https://downloads.wormbase.org/releases/current-production-release/ONTOLOGY/phenotype_ontology.WS290.obo'
url='C:\\Users\\logslab\\Desktop\\COSAS NUEVAS MAYO - fenotipos, genotipo\\phenotype_ontology.WS290.obo'
graph = obonet.read_obo(url)

#create mappings
id_to_name = {id_: data.get('name') for id_, data in graph.nodes(data=True)}
name_to_id = {data['name']: id_ for id_, data in graph.nodes(data=True) if 'name' in data}
phen_description = {data['def']: id_ for id_, data in graph.nodes(data=True) if 'def' in data}

del url



#We keep the nodes -> id phenotype and name phenotype
phenotypes=glp.convert_dictionary_to_array(id_to_name)
phen_description=glp.convert_dictionary_to_array(phen_description)

#We search if just taking 14 or 16 associated phen to the paper and the pleiotropy values correlate
phen_common=np.union1d(phen_paper_morpho, phen_paper_germ_layer)
phen_common=np.delete(phen_common, 0)

phen_comunes_descrip=[]
for i in range(len(phen_common)):
    ind_phen=np.where(np.array(phenotypes[:,0])==phen_common[i])[0]
    if len(ind_phen)>0:
        phen_comunes_descrip.append(phenotypes[int(ind_phen)][1])

phen_comunes_descrip=np.array(phen_comunes_descrip)

new_matrix_gene_phen=np.zeros((len(genes_common_total_paper), len(phen_common)))
new_pleio=[]
for i in range(len(genes_common_total_paper)):
    index_gen=np.where(genes==genes_common_total_paper[i])[0]
    for j in range(len(phen_comunes_descrip)):
        index_phen=np.where(phen==phen_comunes_descrip[j])[0]
        # print(index_phen)
        new_matrix_gene_phen[i][j]=phen_matrix[int(index_gen)][int(index_phen)]
    new_pleio.append(np.sum(new_matrix_gene_phen[i,:]))
    
pearsonr(pleio_germ_layer,pleio_morpho)
pearsonr(new_pleio,pleio_morpho)
pearsonr(new_pleio, pleio_germ_layer)
pearsonr(new_pleio, pleio_nnmf_paper)


# plt.scatter(pleio_morpho, pleio_germ_layer, s=0.9)    
    
# plt.scatter(new_pleio, pleio_germ_layer, s=0.9)    

# plt.hist(new_pleio, bins=100)
    










