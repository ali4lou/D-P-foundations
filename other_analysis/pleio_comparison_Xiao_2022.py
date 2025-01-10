# -*- coding: utf-8 -*-
"""
Pleiotropy comparison: defect buffering
"""


import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
from scipy.cluster.hierarchy import dendrogram
from matplotlib import cm 
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
from sklearn.decomposition import NMF
from sklearn.metrics import mean_squared_error
from scipy.stats import fisher_exact
from sklearn.metrics.pairwise import pairwise_distances


"""
path_save_data, path_dev, path_phen, path_sim and path_pleio
are the path that you chose after download the needed files
"""

path_save_data='C:\\Users\\logslab\\Desktop\\comprobacion_dp_rule\\all_files\\'

path_dev='PATH_WHERE_IS_DOWNLOADED_THE_DATA'
path_phen='PATH_WHERE_IS_DOWNLOADED_THE_DATA'
path_pleio='PATH_WHERE_IS_DOWNLOADED_THE_DATA'


#0.) Data from the paper
df = pd.read_csv(path_save_data+'phen_defect_buffering_cellular.txt',  delimiter='\t')
# Eliminar las filas donde hay NaN en la columna 'phenotypes'
df_clean = df.dropna(subset=['Phenotypes '])

del df

genes_paper=np.array(df_clean['Wormbase ID '], dtype=str)
genes_name_paper=np.array(df_clean['Gene name '], dtype=str)
pleio_paper=np.array(df_clean['Phenotypes '])

genes_paper_without_spaces = [elemento.rstrip() for elemento in genes_paper]
genes_name_paper_without_spaces = [elemento.rstrip() for elemento in genes_name_paper]
genes_name_paper_without_spaces=np.array(genes_name_paper_without_spaces, dtype=str)

genes_name_paper=genes_name_paper_without_spaces
del genes_name_paper_without_spaces
genes_paper=genes_paper_without_spaces
del genes_paper_without_spaces



#1.) We are going to read all the data
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


#2.) Pleiotropy comparison
#We search for the matching genes between the paper and our data
pleio_nnmf_comparing_paper=[]
genes_paper_found=[]
pleio_util_paper=[]
for i in range(len(genes_paper)):
    ind=np.where(genes==genes_paper[i])[0]
    if len(ind)>0:
        genes_paper_found.append(genes_paper[i])
        pleio_nnmf_comparing_paper.append(pleio_score_nnmf[int(ind)])
        pleio_util_paper.append(pleio_paper[i])

genes_paper_encuentro=np.array(genes_paper_found, dtype=str)
pleio_nnmf_comparing_paper=np.array(pleio_nnmf_comparing_paper, dtype=float)
pleio_util_paper=np.array(pleio_util_paper, dtype=int)

print('nmf pleio vs. pleio paper:', spearmanr(pleio_nnmf_comparing_paper, pleio_util_paper), '\n')
plt.scatter(pleio_nnmf_comparing_paper, pleio_util_paper)


#3.) We search if the 'lethal' phenotypes in the paper are also lethal in the WormBase
#We just keep the lethal phenotypes
df_filtered = df_clean[df_clean['Required for cellular function'] == 'Yes, embryonic lethal']

genes_paper=np.array(df_filtered['Wormbase ID '], dtype=str)
genes_name_paper=np.array(df_filtered['Gene name '], dtype=str)
pleio_paper=np.array(df_filtered['Phenotypes '])

genes_paper_without_spaces = [elemento.rstrip() for elemento in genes_paper]
genes_name_paper_without_spaces  = [elemento.rstrip() for elemento in genes_name_paper]
genes_name_paper_without_spaces =np.array(genes_name_paper_without_spaces , dtype=str)

genes_name_paper=genes_name_paper_without_spaces 
del genes_name_paper_without_spaces 
genes_paper=genes_paper_without_spaces 
del genes_paper_without_spaces 


#miro de esos genes cunatos están 
pleio_nnmf_compara_paper=[]
genes_paper_found=[]
pleio_util_paper=[]
for i in range(len(genes_paper)):
    ind=np.where(genes==genes_paper[i])[0]
    if len(ind)>0:
        genes_paper_found.append(genes_paper[i])
        pleio_nnmf_compara_paper.append(pleio_score_nnmf[int(ind)])
        pleio_util_paper.append(pleio_paper[i])

genes_paper_found=np.array(genes_paper_found, dtype=str)



ind_lethal=np.where(phen=='lethal')[0]

si_letales=[]
no_letales=[]
for i in range(len(genes_paper_found)):
    ind_gen=np.where(genes==genes_paper_found[i])[0]
    if phen_matrix[int(ind_gen)][int(ind_lethal)]==1.0:
        si_letales.append(genes_paper_found[i])
    else:
        no_letales.append(genes_paper_found[i])











